import torch

from .light import Light

__all__ = [
    'Paraboloid',
]


class Paraboloid:
    def __init__(self, optical_axis, focus, focal_length, diameter, device, dtype):
        self.device = device
        self.dtype = dtype
        self.focus = torch.tensor(focus, dtype=dtype, device=device)
        self.optical_axis = torch.tensor(
            optical_axis, dtype=dtype, device=device)
        self.focal_length = torch.tensor(
            focal_length, dtype=dtype, device=device)
        self.diameter = torch.tensor(
            diameter, dtype=dtype, device=device)
        self.directrix_center = self.focus - self.focal_length * self.optical_axis * 2

    def reflection(self, light):
        """
            light:
                Light
            return:
                Light
        """
        # 1. get reflection from
        # F: focus, C: directrix_center, s: optical_axis
        # |start + x * dir - F| = (start + x * dir - C) * s
        dir_optical_axis = torch.matmul(
            self.optical_axis.reshape(1, 1, 3), light.dir.reshape(light.count(), 3, 1)).reshape(light.count())
        start_optical_axis = torch.matmul(
            self.optical_axis.reshape(1, 1, 3), light.start.reshape(light.count(), 3, 1)).reshape(light.count())
        directrix_center_optical_axis = torch.matmul(
            self.optical_axis.reshape(1, 1, 3), self.directrix_center.reshape(1, 3, 1)).reshape(1)
        start_start = torch.matmul(
            light.start.reshape(light.count(), 1, 3), light.start.reshape(light.count(), 3, 1)).reshape(light.count())
        start_dir = torch.matmul(
            light.start.reshape(light.count(), 1, 3), light.dir.reshape(light.count(), 3, 1)).reshape(light.count())
        focus_focus = torch.matmul(
            self.focus.reshape(1, 1, 3), self.focus.reshape(1, 3, 1)).reshape(1)
        dir_focus = torch.matmul(
            self.focus.reshape(1, 1, 3), light.dir.reshape(light.count(), 3, 1)).reshape(light.count())
        start_focus = torch.matmul(
            self.focus.reshape(1, 1, 3), light.start.reshape(light.count(), 3, 1)).reshape(light.count())
        x_2_coeff = 1 - dir_optical_axis ** 2
        x_1_coeff = 2 * start_dir - 2 * dir_focus - 2 * \
            dir_optical_axis * \
            (start_optical_axis - directrix_center_optical_axis)
        x_0_coeff = start_start + focus_focus - 2 * start_focus - \
            (start_optical_axis - directrix_center_optical_axis) ** 2
        valid = torch.abs(x_0_coeff) > 1e-6
        light.filter(valid)
        x_2_coeff = x_2_coeff[valid]
        x_1_coeff = x_1_coeff[valid]
        x_0_coeff = x_0_coeff[valid]

        problem = torch.zeros(
            light.count(), 2, 2, device=self.device, dtype=self.dtype)
        problem[:, 0, 0] = -x_1_coeff / x_0_coeff
        problem[:, 0, 1] = -x_2_coeff / x_0_coeff
        problem[:, 1, 0] = 1
        result = 1 / torch.linalg.eigvals(problem)
        valid = torch.logical_and(result.imag == 0, result.real > 0).to(
            torch.int32).sum(dim=1) == 1
        result = result[valid]
        result = result[torch.logical_and(
            result.imag == 0, result.real > 0)].real
        light.filter(valid)
        reflection_start = light.start + \
            result.reshape(light.count(), 1) * light.dir

        # 2. remove out of diameter
        reflection_start_to_focus = self.focus.reshape(1, 3) - reflection_start
        proj = torch.matmul(reflection_start_to_focus.reshape(
            light.count(), 1, 3), self.optical_axis.reshape(1, 3, 1)).reshape(light.count())
        valid = torch.linalg.norm(
            reflection_start_to_focus, dim=1) ** 2 - proj ** 2 < (self.diameter * 0.5) ** 2
        light.filter(valid)
        reflection_start = reflection_start[valid]

        # 3. get reflection dir
        reflection_start_to_focus = self.focus.reshape(1, 3) - reflection_start
        reflection_start_to_focus = reflection_start_to_focus / \
            torch.linalg.norm(reflection_start_to_focus, dim=1, keepdim=True)
        normal = reflection_start_to_focus + self.optical_axis
        normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True)
        normal_part = normal * torch.matmul(normal.reshape(
            light.count(), 1, 3), -light.dir.reshape(light.count(), 3, 1)).reshape(light.count(), 1)
        reflection_dir = light.dir + 2 * normal_part

        return Light(reflection_start, reflection_dir, light.intensity)
