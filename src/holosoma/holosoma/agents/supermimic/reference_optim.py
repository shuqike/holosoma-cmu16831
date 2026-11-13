"""Public-release Muon/Adam partition, compatible with the Isaac Gym runtime.

The Muon equations and constants follow PyTorch v2.10.0 torch/optim/_muon.py
(BSD-3-Clause), as pinned by the public SuperMimic release. Runtime PyTorch 2.4
does not expose torch.optim.Muon. CPU equivalence checks execute the downloaded
upstream functional implementation, including its default weight decay.
"""
import math
from holosoma.utils.safe_torch_import import nn, optim, torch


class ReferenceMuon(optim.Optimizer):
    def __init__(self, parameters, lr=2e-4):
        super().__init__(parameters, dict(lr=lr,weight_decay=.1,momentum=.95))
        if any(p.ndim!=2 for group in self.param_groups for p in group['params']):
            raise ValueError('Muon is restricted to middle Linear weight matrices')

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('The reference adapter uses explicit losses, not optimizer closures')
        for group in self.param_groups:
            lr,momentum=group['lr'],group['momentum']
            for parameter in group['params']:
                gradient=parameter.grad
                if gradient is None:
                    continue
                state=self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer']=torch.zeros_like(parameter)
                buffer=state['momentum_buffer']
                buffer.lerp_(gradient,1-momentum)
                update=gradient.lerp(buffer,momentum).bfloat16()
                transposed=update.shape[0]>update.shape[1]
                if transposed:
                    update=update.T
                update.div_(update.norm().clamp(min=1e-7))
                for _ in range(5):
                    gram=update@update.T
                    polynomial=torch.addmm(gram,gram,gram,beta=-4.7750,alpha=2.0315)
                    update=torch.addmm(update,polynomial,update,beta=3.4445)
                if transposed:
                    update=update.T
                adjusted_lr=lr*.2*math.sqrt(max(parameter.shape))
                parameter.mul_(1-lr*group['weight_decay'])
                parameter.add_(update,alpha=-adjusted_lr)


class OptimizerBundle:
    """One training-step interface, with separate optimizer state dictionaries."""
    def __init__(self, optimizers):
        self.optimizers=optimizers

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    @property
    def state(self):
        return {parameter:state for optimizer in self.optimizers for parameter,state in optimizer.state.items()}

    def zero_grad(self,set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return dict(kind='pytorch_2_10_muon_adam',optimizers=[o.state_dict() for o in self.optimizers])

    def load_state_dict(self,state):
        if state.get('kind')!='pytorch_2_10_muon_adam' or len(state['optimizers'])!=len(self.optimizers):
            raise ValueError('Optimizer variant differs from the saved checkpoint')
        for optimizer,saved in zip(self.optimizers,state['optimizers']):
            optimizer.load_state_dict(saved)


def build_reference_optimizer(module,lr,betas=(.9,.999)):
    middle=[]
    for sequence in module.modules():
        if isinstance(sequence,nn.Sequential):
            linears=[layer for layer in sequence if isinstance(layer,nn.Linear)]
            middle.extend(layer.weight for layer in linears[1:-1])
    ids={id(parameter) for parameter in middle}
    rest=[parameter for parameter in module.parameters() if id(parameter) not in ids]
    cuda=rest[0].is_cuda
    adam=optim.Adam(rest,lr=lr,betas=betas,fused=cuda)
    return OptimizerBundle([adam,ReferenceMuon(middle,lr=lr)] if middle else [adam])
