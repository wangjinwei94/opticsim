import torch

from .light import Light

__all__ = [
    'PixelArray',
]


class PixelArray:
    def __init__(self, center, x_axis, y_axis, x_size, y_size, pixel_size, over_exposure, device, dtype):
        self.device = device
        self.dtype = dtype
        self.x_axis = torch.tensor(x_axis, dtype=dtype, device=device)
        self.y_axis = torch.tensor(y_axis, dtype=dtype, device=device)
        self.optical_axis = torch.cross(self.x_axis, self.y_axis, dim=0)
        self.center = torch.tensor(center, dtype=dtype, device=device)
        self.x_size = x_size
        self.y_size = y_size
        self.pixel_size = pixel_size
        self.x_count = round(x_size / pixel_size)
        self.y_count = round(y_size / pixel_size)
        self.over_exposure = over_exposure
        self.clear_image()

    def clear_image(self):
        self.image = torch.zeros(self.y_count * self.x_count,
                                 dtype=self.dtype, device=self.device)

    def add_rays(self, light):
        """
            light:
                Light
        """
        start_to_center = self.center.reshape(1, 3) - light.start
        start_to_center_optical_axis = torch.matmul(
            self.optical_axis.reshape(1, 1, 3), start_to_center.reshape(light.count(), 3, 1)).reshape(light.count())
        dir_optical_axis = torch.matmul(
            self.optical_axis.reshape(1, 1, 3), light.dir.reshape(light.count(), 3, 1)).reshape(light.count())
        hit_offset = light.start \
            + (start_to_center_optical_axis / dir_optical_axis).reshape(light.count(), 1) * light.dir \
            - self.center.reshape(1, 3)
        x_coord = (torch.matmul(hit_offset.reshape(light.count(), 1, 3),
                                self.x_axis.reshape(1, 3, 1)).reshape(light.count()) + self.x_size * 0.5) / self.pixel_size
        y_coord = (torch.matmul(hit_offset.reshape(light.count(), 1, 3),
                                self.y_axis.reshape(1, 3, 1)).reshape(light.count()) + self.y_size * 0.5) / self.pixel_size
        x_coord = torch.floor(x_coord).to(torch.int64)
        y_coord = torch.floor(y_coord).to(torch.int64)
        valid = torch.logical_and(x_coord >= 0, x_coord < self.x_count)
        valid = torch.logical_and(valid, y_coord >= 0)
        valid = torch.logical_and(valid, y_coord < self.y_count)
        self.image.index_add_(
            0, y_coord[valid] * self.x_count + x_coord[valid], light.intensity[valid])

    def get_image(self):
        """
            return:
                image:
                    Tensor, shape = (y_count, x_count), float32, 0 ~ 1
        """
        max_val = torch.max(self.image).item()
        if max_val <= 0:
            max_val = 1
        bins = 10000000
        hist = torch.histc(self.image / max_val, bins=bins,
                           min=0, max=1) / self.image.numel()
        max_val = torch.argmax(
            (torch.torch.cumsum(hist, dim=0) > (1 - self.over_exposure)).to(torch.int32)) / bins * max_val
        if max_val <= 0:
            max_val = 1
        return torch.minimum(self.image.reshape(self.y_count, self.x_count) / max_val, self.image.new_tensor(1))
