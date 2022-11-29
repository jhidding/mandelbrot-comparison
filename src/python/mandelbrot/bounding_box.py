# ~\~ language=Python filename=src/python/mandelbrot/bounding_box.py
# ~\~ begin <<docs/src/python.md|src/python/mandelbrot/bounding_box.py>>[init]
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ~\~ begin <<docs/src/python.md|complex-from-string-py>>[init]
def complex_from_pair(s: list[float]) -> complex:
    return complex(s[0], s[1])
# ~\~ end
# ~\~ begin <<docs/src/python.md|bounding-box-py>>[init]
@dataclass
class BoundingBox:
    width: int
    height: int
    center: complex
    extent: complex
    _scale: Optional[float] = None

    @property
    def scale(self):
        if self._scale is None:
            self._scale = max(self.extent.real / self.width,
                              self.extent.imag / self.height)
        return self._scale

    # ~\~ begin <<docs/src/python.md|bounding-box-methods-py>>[init]
    @staticmethod
    def from_config(cfg):
        w = int(cfg["width"])
        h = int(cfg["height"])
        c = complex_from_pair(cfg["center"])
        e = complex_from_pair(cfg["extent"])
        return BoundingBox(w, h, c, e)
    # ~\~ end
    # ~\~ begin <<docs/src/python.md|bounding-box-methods-py>>[1]
    def split(self, n):
        """Split the domain in nxn subdomains, and return a grid of BoundingBoxes."""
        w = self.width // n
        h = self.height // n
        e = self.scale * w + self.scale * h * 1j
        x0 = self.center - e * (n / 2 - 0.5)
        return [[BoundingBox(w, h, x0 + i * e.real + j * e.imag * 1j, e)
                 for i in range(n)]
                for j in range(n)]
    # ~\~ end
    # ~\~ begin <<docs/src/python.md|bounding-box-methods-py>>[2]
    def grid(self):
        """Return the complex values on the grid in a 2d array."""
        x0 = self.center - self.extent / 2
        x1 = self.center + self.extent / 2
        g = np.mgrid[x0.imag:x1.imag:self.height*1j,
                     x0.real:x1.real:self.width*1j]
        return g[1] + g[0]*1j
    # ~\~ end
# ~\~ end
# ~\~ end
