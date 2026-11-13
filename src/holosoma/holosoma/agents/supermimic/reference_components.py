"""Numerical components for the public SuperMimic G1 adaptation.

Reference: HybridRobotics/supermimic, commit 5efbf114e11f725c31d83e217be987c13c0e4ce2.
Quaternion convention here is wxyz, explicitly converted at the simulator boundary.
These functions do not alter the task, rewards, observations or termination rules.
"""
from holosoma.utils.safe_torch_import import nn, torch


class RunningMoments(nn.Module):
    """Explicit pooled-moment updates; forwarding never changes statistics.

    Uses the public implementation's population variance and std + 1e-2
    convention. Updates are restricted to real training observations by callers.
    """

    def __init__(self, dimension, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(1, dimension))
        self.register_buffer("variance", torch.ones(1, dimension))
        self.register_buffer("std", torch.ones(1, dimension))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, values):
        return (values-self.mean)/(self.std+self.epsilon)

    @torch.no_grad()
    def update(self, values):
        if not self.training:
            return
        values = values.reshape(-1, self.mean.shape[-1])
        self.count.add_(values.shape[0])
        fraction = values.shape[0]/self.count
        batch_mean = values.mean(0, keepdim=True)
        batch_variance = values.var(0, unbiased=False, keepdim=True)
        difference = batch_mean-self.mean
        self.mean.add_(fraction*difference)
        self.variance.add_(fraction*(batch_variance-self.variance+difference*(batch_mean-self.mean)))
        self.std.copy_(self.variance.clamp_min(0).sqrt())


def quaternion_product(left, right):
    w1, v1 = left[..., :1], left[..., 1:]
    w2, v2 = right[..., :1], right[..., 1:]
    return torch.cat((w1*w2-(v1*v2).sum(-1, keepdim=True),
                      w1*v2+w2*v1+torch.cross(v1,v2,dim=-1)), -1)


def quaternion_rotate(quaternion, vector, inverse=False):
    imaginary = -quaternion[..., 1:] if inverse else quaternion[..., 1:]
    cross = 2*torch.cross(imaginary,vector,dim=-1)
    return vector+quaternion[..., :1]*cross+torch.cross(imaginary,cross,dim=-1)


def rotation_increment(angular_velocity, dt):
    """The reference axis-angle integration, finite at zero angular velocity."""
    omega = angular_velocity*dt
    angle = (omega.square().sum(-1, keepdim=True)+1e-8).sqrt()
    result = torch.cat(((angle*.5).cos(), (angle*.5).sin()*omega/angle), -1)
    return result/result.norm(dim=-1, keepdim=True)


def integrate_generalized(linear, angular, gravity, joints, joint_velocity,
                          increments, dt, decimation):
    """Velocity-only prediction with the reference semi-implicit substep rule.

    No independently learned joint-position or orientation residual is allowed.
    All vectors are in the current body frame; successors use the new frame.
    """
    coefficient = (decimation+1)/(2*decimation)
    dv, dw, dqdot = increments.split((3,3,joints.shape[-1]),dim=-1)
    pose_linear, pose_angular = linear+coefficient*dv, angular+coefficient*dw
    delta = rotation_increment(pose_angular,dt)
    return dict(linear=quaternion_rotate(delta,linear+dv,inverse=True),
        angular=quaternion_rotate(delta,angular+dw,inverse=True),
        gravity=quaternion_rotate(delta,gravity,inverse=True),
        joints=joints+(joint_velocity+coefficient*dqdot)*dt,
        joint_velocity=joint_velocity+dqdot,
        vertical_displacement=-(gravity*pose_linear).sum(-1,keepdim=True)*dt,
        pose_linear=pose_linear,rotation_increment=delta)


def advance_root_pose(root_pose, integration, dt):
    orientation = quaternion_product(root_pose[..., 3:7],integration["rotation_increment"])
    orientation = orientation/orientation.norm(dim=-1,keepdim=True)
    position = root_pose[..., :3]+quaternion_rotate(root_pose[..., 3:7],integration["pose_linear"])*dt
    return torch.cat((position,orientation),-1)


def rotation_matrix(quaternion):
    identity = torch.eye(3,dtype=quaternion.dtype,device=quaternion.device)
    axes = identity.expand(*quaternion.shape[:-1],3,3)
    rotated = quaternion_rotate(quaternion.unsqueeze(-2).expand(*quaternion.shape[:-1],3,4),axes)
    return rotated.transpose(-1,-2)


def physical_loss_blocks(state, root_pose, state_scale):
    physical = state/state_scale
    return dict(root_pos=root_pose[..., :3],
        root_ori=rotation_matrix(root_pose[..., 3:7]).flatten(-2)*(.5**.5),
        joint_pos=physical[..., 9:38],joint_vel=physical[..., 38:67])


def weighted_channel_mean(prediction, target, nominal_errors):
    """Reference Bryson weighting followed by a mean over all scored channels."""
    squared = {name:(prediction[name]-target[name]).square() for name in prediction}
    width = sum(value.shape[-1] for value in squared.values())
    loss = sum(value.mean()*value.shape[-1]/nominal_errors[name]**2 for name,value in squared.items())/width
    return loss, {name:value.mean().detach() for name,value in squared.items()}
