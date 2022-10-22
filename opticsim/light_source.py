import torch

from .light import Light

__all__ = [
    'Point',
]


class Point:
    def __init__(self, position, intensity, device, dtype):
        self.device = device
        self.dtype = dtype
        self.position = torch.tensor(position, dtype=dtype, device=device)
        self.intensity = intensity

    def sample(self, aperture, num):
        """
            aperture:
                Aperture
            num:
                int, maximum ramdon light count
            return:
                Light
        """
        dir = aperture.sample(num) - self.position.reshape(1, 3)
        dir = dir / \
            torch.linalg.norm(dir, dim=1, keepdim=True)
        num = dir.shape[0]
        start = self.position.reshape(1, 3).expand(num, -1)
        intensity = torch.ones(
            num, dtype=self.dtype, device=self.device) * self.intensity
        return Light(start, dir, intensity)
