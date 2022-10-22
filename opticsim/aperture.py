import torch

__all__ = [
    'Circle',
]


class Circle:
    def __init__(self, optical_axis, center, diameter, device, dtype):
        self.device = device
        self.dtype = dtype
        self.diameter = diameter
        self.optical_axis = torch.tensor(
            optical_axis, dtype=dtype, device=device)
        self.center = torch.tensor(center, dtype=dtype, device=device)
        self.y_axis = torch.cross(self.optical_axis, torch.randn(
            3, dtype=dtype, device=device), dim=0)
        self.y_axis = self.y_axis / torch.linalg.norm(self.y_axis, dim=0)
        self.x_axis = torch.cross(self.y_axis, self.optical_axis)

    def sample(self, num):
        """
            num:
                int, maximum ramdon point count
            return:
                random_point:
                    Tensor, shape = (actual_num, 3), float32
        """
        x = (torch.rand(num, dtype=self.dtype,
             device=self.device) - 0.5) * self.diameter
        y = (torch.rand(num, dtype=self.dtype,
             device=self.device) - 0.5) * self.diameter
        valid = (x ** 2 + y ** 2) < ((self.diameter * 0.5) ** 2)
        x = x[valid]
        y = y[valid]
        num = x.shape[0]
        return self.center.reshape(1, 3) \
            + x.reshape(num, 1) * self.x_axis.reshape(1, 3) \
            + y.reshape(num, 1) * self.y_axis.reshape(1, 3)
