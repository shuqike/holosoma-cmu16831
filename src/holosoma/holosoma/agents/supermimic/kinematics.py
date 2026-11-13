"""Differentiable leg geometry read from the task's actual URDF.

Body geometry is determined by joint angles and base orientation. Terrain clearance
also depends on terrain and base translation, so it is intentionally kept separate.
"""
import math
import xml.etree.ElementTree as ET

from holosoma.utils.safe_torch_import import F, nn, torch


def _rpy(values):
    r, p, y = values
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


class LegGeometry(nn.Module):
    def __init__(self, urdf, joint_names, nominal, foot_names):
        super().__init__()
        joints = {j.find("child").get("link"): j for j in ET.fromstring(urdf).findall("joint")}
        chains = []
        for foot in foot_names:
            chain, link = [], foot
            while link in joints:
                joint = joints[link]
                chain.append(joint)
                link = joint.find("parent").get("link")
            chains.append(chain[::-1])
        if len(chains) != 2 or len(chains[0]) != len(chains[1]):
            raise ValueError("Expected two G1 leg chains of equal length")
        offsets, rotations, axes, indices = [], [], [], []
        for pair in zip(*chains):
            xyz, rpy, axis, index = [], [], [], []
            for joint in pair:
                origin = joint.find("origin")
                xyz.append([float(x) for x in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
                rpy.append(_rpy([float(x) for x in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]))
                fixed = joint.get("type") == "fixed"
                axis.append([0., 0., 0.] if fixed else [float(x) for x in joint.find("axis").get("xyz").split()])
                index.append(-1 if fixed else joint_names.index(joint.get("name")))
            offsets.append(xyz)
            rotations.append(rpy)
            axes.append(axis)
            indices.append(index)
        for name, value in (("offsets", offsets), ("rotations", rotations), ("axes", axes)):
            self.register_buffer(name, nominal.new_tensor(value))
        self.register_buffer("indices", torch.tensor(indices, device=nominal.device))
        self.register_buffer("nominal", nominal.clone())

    def forward(self, joint_offsets, gravity):
        q = joint_offsets + self.nominal
        shape = (*q.shape[:-1], 2, 3, 3)
        eye = torch.eye(3, device=q.device, dtype=q.dtype)
        rotation = eye.expand(shape)
        position = q.new_zeros(*q.shape[:-1], 2, 3)
        for offset, fixed_rotation, axis, indices in zip(self.offsets, self.rotations, self.axes, self.indices):
            position = position + (rotation @ offset[..., None]).squeeze(-1)
            rotation = rotation @ fixed_rotation
            angle = q[..., indices.clamp_min(0)] * (indices >= 0)
            x, y, z = axis.unbind(-1)
            zero = torch.zeros_like(x)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(2, 3, 3)
            joint_rotation = eye + angle.sin()[..., None, None] * skew + (1-angle.cos())[..., None, None] * (skew @ skew)
            rotation = rotation @ joint_rotation
        gravity = F.normalize(gravity, dim=-1)
        feet_gravity = (rotation.transpose(-1, -2) @ gravity[..., None, :, None]).squeeze(-1)
        orientation = feet_gravity[..., :2].norm(dim=-1).sum(dim=-1, keepdim=True)
        # Horizontal left direction in the base frame, using its projected forward axis.
        lateral = F.normalize(torch.stack((torch.zeros_like(gravity[..., 0]), -gravity[..., 2], gravity[..., 1]), -1), dim=-1)
        separation = ((position[..., 0, :] - position[..., 1, :]) * lateral).sum(-1, keepdim=True).abs()
        vertical = -(position * gravity[..., None, :]).sum(-1)
        return separation, orientation, vertical
