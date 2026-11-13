"""Single velocity slot of the public encoder/latent/decoder policy.

Keeps the G1 observation tensor unchanged: command channels become the goal,
and remaining channels form the one-frame sensor input. No reference motions,
extra history, new commands or multitask imitation targets are introduced.
"""
from holosoma.utils.safe_torch_import import nn, torch
from holosoma.agents.supermimic.reference_components import RunningMoments


def build_mlp(input_dim,hidden_dims,output_dim):
    layers=[]
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim,width),nn.LayerNorm(width),nn.GELU()))
        input_dim=width
    return nn.Sequential(*layers,nn.Linear(input_dim,output_dim))


class VelocityNormalization(nn.Module):
    def __init__(self,obs_dim,goal_indices,obs_scale):
        super().__init__()
        self.register_buffer("goal_indices",torch.tensor(goal_indices,dtype=torch.long,device=obs_scale.device))
        self.register_buffer("sensor_indices",torch.tensor([i for i in range(obs_dim) if i not in goal_indices],
                                                          dtype=torch.long,device=obs_scale.device))
        self.register_buffer("goal_scale",obs_scale[self.goal_indices].clone())
        self.sensor=RunningMoments(obs_dim-len(goal_indices))
        self.goal=RunningMoments(len(goal_indices))

    def split(self,observation):
        return observation[...,self.sensor_indices],observation[...,self.goal_indices]/self.goal_scale

    def forward(self,observation):
        sensor,goal=self.split(observation)
        return self.sensor(sensor),self.goal(goal)

    @torch.no_grad()
    def update(self,observation):
        sensor,goal=self.split(observation)
        self.sensor.update(sensor)
        self.goal.update(goal)


class VelocityActor(nn.Module):
    def __init__(self,obs_dim,goal_indices,obs_scale,action_bound,hidden_dims,latent_dim=32):
        super().__init__()
        self.normalizer=VelocityNormalization(obs_dim,goal_indices,obs_scale)
        self.encoder=build_mlp(len(goal_indices),hidden_dims,latent_dim)
        self.decoder=build_mlp(obs_dim-len(goal_indices)+latent_dim,hidden_dims,action_bound.numel())
        self.register_buffer("action_bound",action_bound.clone())
        self.register_buffer("obs_action_scale",None,persistent=False)

    def forward(self,observation):
        sensor,goal=self.normalizer(observation)
        latent=self.encoder(goal).tanh()
        return self.decoder(torch.cat((latent,sensor),-1))
