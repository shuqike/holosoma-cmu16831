"""Execute downloaded official functions and compare the G1 numerical adaptation."""
import argparse
import hashlib
import json
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from train_supermimic_isaacgym import make_config
make_config(Path('/tmp/supermimic-reference-check'),64,1,1)
from holosoma.utils.safe_torch_import import torch, nn
from holosoma.agents.supermimic.reference_components import (
    RunningMoments, quaternion_product, quaternion_rotate, integrate_generalized,
    advance_root_pose, rotation_matrix, weighted_channel_mean)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--official-source", type=Path, required=True)
parser.add_argument("--muon-source", type=Path, required=True)
parser.add_argument("--normalization-source", type=Path, required=True)
args = parser.parse_args()
ROOT = args.official_source.resolve()
EXTERNAL = {"dependencies/torch_muon.py": args.muon_source,
            "dependencies/rsl_normalization.py": args.normalization_source}
EXPECTED = json.loads((Path(__file__).resolve().parents[1] / "configs/supermimic/reference_sources.json").read_text())["sha256"]
for relative, expected in EXPECTED.items():
    path = EXTERNAL.get(relative, ROOT / relative)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Reference source does not match the audited version: " + str(path))


def source_object(path,name,namespace,parent=None):
    tree=ast.parse(EXTERNAL.get(path, ROOT/path).read_text())
    body=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==parent).body if parent else tree.body
    node=next(node for node in body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(ROOT/path),'exec'),namespace)
    return namespace[name]


torch.manual_seed(17)
namespace=dict(torch=torch,nn=nn)
OfficialMoments=source_object('dependencies/rsl_normalization.py','EmpiricalNormalization',namespace)
ours,reference=RunningMoments(7).double(),OfficialMoments(7).double()
for size in (3,19,1,45,6):
    values=torch.randn(size,7,dtype=torch.float64)*2+4
    ours.update(values)
    reference.update(values)
    torch.testing.assert_close(ours(values),reference(values),atol=1e-12,rtol=1e-12)
before={k:v.clone() for k,v in ours.state_dict().items()}
ours.eval()
ours.update(torch.randn(99,7,dtype=torch.float64))
assert all(torch.equal(v,ours.state_dict()[k]) for k,v in before.items())

namespace=dict(torch=torch,quat_mul=quaternion_product,
    quat_apply=quaternion_rotate,quat_apply_inverse=lambda q,v:quaternion_rotate(q,v,inverse=True))
source_object('src/supermimic/utils/gencoord_helpers.py','rep_state_offsets',namespace)
source_object('src/supermimic/utils/gencoord_helpers.py','decompose_rep_state',namespace)
source_object('src/supermimic/utils/helpers.py','quat_integrate',namespace)
official_step=source_object('src/supermimic/representations.py','apply_prediction',namespace,'GenCoordRepresentation')
max_error=0.
for decimation in (1,2,4,10):
    quat=torch.nn.functional.normalize(torch.randn(16,4,dtype=torch.float64),dim=-1)
    pose=torch.cat((torch.randn(16,3,dtype=torch.float64),quat),-1)
    down=torch.zeros(16,3,dtype=torch.float64)
    down[:,2]=-1
    gravity=quaternion_rotate(quat,down,inverse=True)
    linear,angular,joints,velocity=[torch.randn(16,n,dtype=torch.float64) for n in (3,3,29,29)]
    cfg=SimpleNamespace(num_joints=29,substep_coeff=(decimation+1)/(2*decimation))
    for step in range(32):
        increments=torch.randn(16,35,dtype=torch.float64,requires_grad=True)*.1
        rep=torch.cat((pose[:,2:3],gravity,linear,angular,joints,velocity),-1)
        ref_pose,ref_state=official_step(cfg,pose,rep,increments,.02)
        integrated=integrate_generalized(linear,angular,gravity,joints,velocity,increments,.02,decimation)
        our_pose=advance_root_pose(pose,integrated,.02)
        our_state=torch.cat((our_pose[:,2:3],integrated['gravity'],integrated['linear'],integrated['angular'],
                             integrated['joints'],integrated['joint_velocity']),-1)
        max_error=max(max_error,float((ref_state-our_state).abs().max()),float((ref_pose-our_pose).abs().max()))
        torch.testing.assert_close(ref_state,our_state,atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(ref_pose,our_pose,atol=1e-10,rtol=1e-10)
        ref_gradient,=torch.autograd.grad(ref_state.square().sum()+ref_pose.square().sum(),increments)
        our_gradient,=torch.autograd.grad(our_state.square().sum()+our_pose.square().sum(),increments)
        torch.testing.assert_close(ref_gradient,our_gradient,atol=1e-10,rtol=1e-10)
        pose,our_state=our_pose.detach(),our_state.detach()
        gravity,linear,angular,joints,velocity=our_state[:,1:4],our_state[:,4:7],our_state[:,7:10],our_state[:,10:39],our_state[:,39:68]

identity=torch.eye(3,dtype=torch.float64).expand(16,3,3)
matrix=rotation_matrix(quat)
torch.testing.assert_close(matrix.transpose(-1,-2)@matrix,identity,atol=1e-12,rtol=1e-12)
pred={'position':torch.randn(5,8,3,dtype=torch.float64),'rotation':torch.randn(5,8,9,dtype=torch.float64)}
target={k:torch.randn_like(v) for k,v in pred.items()}
nominal={'position':.01,'rotation':.02}
actual,_=weighted_channel_mean(pred,target,nominal)
expected=torch.cat([(pred[k]-target[k])/nominal[k] for k in pred],-1).square().mean()
torch.testing.assert_close(actual,expected)
print('PASS: public rsl_rl normalization; official gencoord 32-step states, root poses and Jacobians at 4 decimations; channel-weighted loss. Max state error:',max_error)

from holosoma.agents.supermimic.reference_optim import ReferenceMuon,build_reference_optimizer
namespace=dict(torch=torch,Tensor=torch.Tensor,math=math,_to_scalar=lambda value:float(value))
for name in ('_zeropower_via_newtonschulz','_adjust_lr','_single_tensor_muon'):
    source_object('dependencies/torch_muon.py',name,namespace)
for shape in ((5,9),(9,5),(8,8)):
    parameter=nn.Parameter(torch.randn(*shape))
    reference=parameter.detach().clone()
    momentum=torch.zeros_like(reference)
    optimizer=ReferenceMuon([parameter],lr=2e-4)
    for step in range(7):
        gradient=torch.randn_like(parameter)
        parameter.grad=gradient.clone()
        optimizer.step()
        namespace['_single_tensor_muon']([reference],[gradient],[momentum],lr=2e-4,weight_decay=.1,
            momentum=.95,nesterov=True,ns_coefficients=(3.4445,-4.7750,2.0315),ns_steps=5,
            eps=1e-7,adjust_lr_fn='match_rms_adamw',has_complex=False)
        assert torch.equal(parameter,reference)
        assert torch.equal(optimizer.state[parameter]['momentum_buffer'],momentum)
module=nn.Sequential(nn.Linear(5,8),nn.LayerNorm(8),nn.GELU(),nn.Linear(8,8),nn.GELU(),nn.Linear(8,3))
bundle=build_reference_optimizer(module,2e-4)
assert bundle.optimizers[1].param_groups[0]['params']==[module[3].weight]
print('PASS: Muon adapter matches PyTorch 2.10 functional updates and momentum bitwise on three matrix shapes over seven steps; official middle-layer partition')

from holosoma.agents.supermimic.world_model import SequenceReplay
replay=SequenceReplay(9)
for step in range(21):
    replay.append(dict(state=torch.stack((torch.full((4,),step),torch.arange(4)),-1),
        done=torch.tensor([step%3==0,step%7==0,False,True])))
batch=replay.sample(40000,4,("state","done"),complete=True)
assert not batch['done'][:,:-1].any()
assert (batch['state'][:,1:,0]-batch['state'][:,:-1,0]==1).all()
assert (batch['state'][:,:,1]==batch['state'][:,:1,1]).all()
assert batch['state'][:,:,0].min()>=12
# World 0 has no length-four window; world 3 terminates every step. The other
# two have unequal valid-window counts but must each receive half the draws.
worlds,counts=torch.unique(batch['state'][:,0,1],return_counts=True)
assert worlds.tolist()==[1,2]
assert ((counts.float()/counts.sum()-.5).abs()<.015).all()
one=replay.sample(1000,1,("done",),torch.tensor([3]),complete=True)
assert one['done'].all()  # Terminal last transition is observable, hence valid.
replay.append(dict(state=torch.stack((torch.full((4,),21),torch.arange(4)),-1),done=torch.zeros(4,dtype=torch.bool)))
assert not replay._complete_cache
print('PASS: complete replay windows across ring wrap, boundary exclusion, terminal-last inclusion, uniform eligible worlds and cache invalidation')

from holosoma.agents.supermimic.world_model import lambda_returns
reference_gae=source_object('src/supermimic/learners/policy_learner.py','gae_returns',dict(torch=torch))
values=torch.randn(33,128,1,dtype=torch.float64)*100
rewards=torch.randn(32,128,1,dtype=torch.float64)
dones=torch.rand(32,128,1,dtype=torch.float64)
for lam in (0.,.95,1.):
    expected=reference_gae(values[:-1],values[-1],rewards,dones,.99,lam).clamp_max(7.)
    actual=lambda_returns(rewards,.99*(1-dones),values[1:],lam).clamp_max(7.)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
print('PASS: fixed GAE targets and final-only ceiling match official code, including soft termination and lambda endpoints')

from holosoma.agents.supermimic.supermimic import Actor
build_official_mlp=source_object('src/supermimic/models.py','build_mlp',dict(torch=torch,nn=nn))
actor=Actor(11,32,torch.linspace(2.,12.,29),reference=True,raw_actions=True).double()
decoder=build_official_mlp(11,(32,32),29).double()
decoder.load_state_dict(actor.network.state_dict())
observation=torch.randn(7,11,dtype=torch.float64,requires_grad=True)
actual=actor(observation)
expected=decoder(actor.normalizer(observation))
torch.testing.assert_close(actual,expected,atol=0,rtol=0)
actual_grad,=torch.autograd.grad(actual.square().sum(),observation)
expected_grad,=torch.autograd.grad(expected.square().sum(),observation)
torch.testing.assert_close(actual_grad,expected_grad,atol=0,rtol=0)
# Bounds are storage scaling only in the reference actor; they must never
# amplify decoder output or its Jacobian. Simulator processing owns clipping.
actor.action_bound.mul_(5.)
torch.testing.assert_close(actor(observation),expected,atol=0,rtol=0)
with torch.no_grad():
    actor.network[-1].weight.zero_()
    actor.network[-1].bias.fill_(20.)
torch.testing.assert_close(actor(observation),torch.full((7,29),20.,dtype=torch.float64),atol=0,rtol=0)
print('PASS: raw actor output and observation Jacobian match official GELU decoder; no tanh or joint-bound amplification')

from holosoma.agents.supermimic.reference_actor import VelocityActor
import builtins
def compatible_zip(*iterables,strict=False):
    if strict:
        assert len({len(value) for value in iterables})==1
    return builtins.zip(*iterables)
namespace_actor=dict(torch=torch,nn=nn,_CudaGraphNormalization=OfficialMoments,
    build_mlp=build_official_mlp,zip=compatible_zip)
OfficialActor=source_object('src/supermimic/models.py','MultiEncoderAE',namespace_actor)
goal_indices=[4,7,2]
scales=torch.linspace(.25,2.,11,dtype=torch.float64)
ours_actor=VelocityActor(11,goal_indices,scales,torch.ones(29),(64,32,16,8),32).double()
official_actor=OfficialActor(8,29,[SimpleNamespace(name='velocity',goal_dim=3)],
    {'velocity':1.},32,(64,32,16,8),0.,1).double()
official_actor.encoders['velocity'].load_state_dict(ours_actor.encoder.state_dict())
official_actor.decoder.load_state_dict(ours_actor.decoder.state_dict())
class ObservationDict(dict):
    @property
    def device(self):
        return self['history'].device
    @property
    def shape(self):
        return (self['history'].shape[0],)
for count in (17,5,31):
    obs=torch.randn(count,11,dtype=torch.float64,requires_grad=True)
    sensor,goal=ours_actor.normalizer.split(obs)
    td=ObservationDict(history=sensor[:,None],goal_velocity=goal)
    ours_actor.normalizer.update(obs)
    # The runner supplies detached real observations; differentiating through
    # statistics computed from the probe itself would test a different policy.
    with torch.no_grad():
        official_actor.update_normalization(td)
    ours_output,official_output=ours_actor(obs),official_actor(td)
    torch.testing.assert_close(ours_output,official_output,atol=1e-12,rtol=1e-12)
    ours_gradient,=torch.autograd.grad(ours_output.square().sum(),obs)
    official_gradient,=torch.autograd.grad(official_output.square().sum(),obs)
    torch.testing.assert_close(ours_gradient,official_gradient,atol=1e-12,rtol=1e-12)
print('PASS: full single-slot velocity encoder/latent/decoder, split normalizers, outputs and observation Jacobians match official MultiEncoderAE')

# Execute the exact upstream functional update on CUDA as well: bfloat16 GEMM
# and fused Adam are the actual training path, unlike a CPU-only smoke check.
if torch.cuda.is_available():
    for shape in ((64,64),(256,128),(128,256)):
        parameter=nn.Parameter(torch.randn(*shape,device='cuda'))
        reference=parameter.detach().clone()
        momentum=torch.zeros_like(reference)
        optimizer=ReferenceMuon([parameter])
        for step in range(3):
            gradient=torch.randn_like(parameter)
            parameter.grad=gradient.clone()
            optimizer.step()
            namespace['_single_tensor_muon']([reference],[gradient],[momentum],lr=2e-4,weight_decay=.1,
                momentum=.95,nesterov=True,ns_coefficients=(3.4445,-4.7750,2.0315),ns_steps=5,
                eps=1e-7,adjust_lr_fn='match_rms_adamw',has_complex=False)
            torch.testing.assert_close(parameter,reference,rtol=0,atol=0)
            torch.testing.assert_close(optimizer.state[parameter]['momentum_buffer'],momentum,rtol=0,atol=0)
    print('PASS: CUDA Muon parameters and momentum are bitwise equal to upstream functional updates')
