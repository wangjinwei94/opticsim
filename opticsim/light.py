__all__ = [
    'Light',
]


class Light:
    def __init__(self, start, dir, intensity):
        self.start = start
        self.dir = dir
        self.intensity = intensity

    def count(self):
        return self.start.shape[0]

    def filter(self, valid):
        self.start = self.start[valid]
        self.dir = self.dir[valid]
        self.intensity = self.intensity[valid]
