"""Public SuperMimic generalized integration with G1 rough-terrain features.

The 35 velocity increments and physical pose loss follow the pinned public
release. Ground-height changes and contact termination are explicit G1 task
extensions: copying motion-tracking termination would change the task protocol.
"""
from holosoma.utils.safe_torch_import import F, nn, torch
from holosoma.agents.supermimic.reference_components import (
    RunningMoments, integrate_generalized, advance_root_pose,
    physical_loss_blocks,
)
from holosoma.agents.supermimic.world_model import mlp


class ReferenceDynamics(nn.Module):
    def __init__(self, state_scale, action_scale, action_offset, context_dim, hidden_dim,
                 geometry, dt, decimation):
        super().__init__()
        if action_scale.numel() != 29 or decimation < 1:
            raise ValueError("Expected G1 joints and positive control decimation")
        self.register_buffer("state_scale", state_scale.clone())
        self.register_buffer("action_scale", action_scale.clone())
        self.register_buffer("action_offset", action_offset.clone())
        self.state_normalizer = RunningMoments(state_scale.numel())
        self.action_normalizer = RunningMoments(action_scale.numel())
        self.dt, self.decimation = dt, decimation
        self.joint_dt_scaled = dt / .05  # diagnostic dataset metadata
        self.geometry = geometry
        self.terrain_dim = state_scale.numel()-71
        self.network = nn.Sequential(
            *list(mlp(state_scale.numel()+29+context_dim,hidden_dim,hidden_dim,nn.GELU).children()),
            nn.LayerNorm(hidden_dim),nn.GELU(),nn.Linear(hidden_dim,35+2+self.terrain_dim+1))
        # Public dynamics uses standard Linear initialization. Only the contact
        # adapter bias encodes the infrequent one-control-step fall prior.
        with torch.no_grad():
            self.network[-1].bias[-1] = -4.

    @torch.no_grad()
    def update_normalizers(self, state, action):
        self.state_normalizer.update(state/self.state_scale)
        self.action_normalizer.update(action*self.action_scale+self.action_offset)

    def predict(self, state, action, context):
        physical = state/self.state_scale
        prediction = self.network(torch.cat((self.state_normalizer(physical),
            self.action_normalizer(action*self.action_scale+self.action_offset),context),-1))
        integration = integrate_generalized(physical[..., :3],physical[..., 3:6],physical[..., 6:9],
            physical[..., 9:38],physical[..., 38:67],prediction[..., :35],self.dt,self.decimation)
        separation, orientation, vertical = self.geometry(integration["joints"],integration["gravity"])
        _, _, old_vertical = self.geometry(physical[..., 9:38],physical[..., 6:9])
        ground_delta = prediction[..., 35:-1]
        dz = integration["vertical_displacement"]
        clearance = physical[..., 67:69]+dz+vertical-old_vertical-ground_delta[..., :2]
        terrain = physical[..., 71:]+dz-ground_delta[..., 2:]
        successor = torch.cat((integration["linear"],integration["angular"],integration["gravity"],
            integration["joints"],integration["joint_velocity"],clearance,separation,orientation,terrain),-1)
        return successor*self.state_scale,prediction[..., -1],integration

    def forward(self,state,action,context):
        state,fall,integration = self.predict(state,action,context)
        return state,fall

    def physical_error(self,state,pose,target_state,target_pose):
        predicted = physical_loss_blocks(state,pose,self.state_scale)
        target = physical_loss_blocks(target_state,target_pose,self.state_scale)
        predicted["terrain"] = torch.cat((state[...,67:69],state[...,71:]),-1)/5.
        target["terrain"] = torch.cat((target_state[...,67:69],target_state[...,71:]),-1)/5.
        tolerances = dict(root_pos=.01,root_ori=.02,joint_pos=.1,joint_vel=3.14,terrain=.01)
        width = sum(value.shape[-1] for value in predicted.values())
        return sum(((predicted[key]-target[key])/tolerances[key]).square().sum(-1) for key in predicted)/width

    def rollout_loss(self,batch):
        state, pose = batch["state"][:,0], batch["root_pose"][:,0]
        valid = torch.ones_like(batch["done"][:,0],dtype=torch.bool)
        weighted_sum, fall_sum, valid_count = state.new_zeros(()),state.new_zeros(()),state.new_zeros(())
        # Root and joint tolerances are the release's GenCoord Bryson weights.
        # Terrain channels are distances, using its same 1 cm position tolerance.
        for step in range(batch["done"].shape[1]):
            state,fall,integration = self.predict(state,batch["applied_action"][:,step],batch["context"][:,step])
            pose = advance_root_pose(pose,integration,self.dt)
            # Masking supports independent long diagnostics; training samples
            # complete windows so every scored transition is valid.
            per_sample = self.physical_error(state,pose,batch["next_state"][:,step],batch["next_root_pose"][:,step])
            weighted_sum = weighted_sum+(per_sample*valid).sum()
            fall_sum = fall_sum+(F.binary_cross_entropy_with_logits(fall,batch["terminated"][:,step].float(),reduction="none")*valid).sum()
            valid_count = valid_count+valid.sum()
            valid = valid & ~batch["done"][:,step]
            state = torch.where(valid[:,None],state,batch["next_state"][:,step])
            pose = torch.where(valid[:,None],pose,batch["next_root_pose"][:,step])
        return weighted_sum/valid_count,fall_sum/valid_count
