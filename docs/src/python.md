# Python
The main approach with Python will be: use Numba to make this fast. Then there are two ways to parallelize: let Numba parallelize the function, or do a manual domain decomposition and use one of many ways in Python to run things multi-threaded. There is a third way: create a vectorized function and parallelize using `dask.array`. This last option is almost always slower than `@njit(parallel=True)` or domain decomposition.

## Bounding box
We start with a naive implementation. We have a `BoundingBox`:

``` {.python #bounding-box-py}
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

    <<bounding-box-methods-py>>
```


### Reading from config

``` {.python #complex-from-string-py}
def complex_from_pair(s: list[float]) -> complex:
    return complex(s[0], s[1])
```

``` {.python #bounding-box-methods-py}
@staticmethod
def from_config(cfg):
    w = int(cfg["width"])
    h = int(cfg["height"])
    c = complex_from_pair(cfg["center"])
    e = complex_from_pair(cfg["extent"])
    return BoundingBox(w, h, c, e)
```

### Create a sub-module
To make this code into a module we have to create a file `__init__.py`. This file can be left empty.

``` {.python file=src/python/mandelbrot/__init__.py}

```

To make the type checker work properly, we have to create an emtpy file `py.typed`

``` {.python file=src/python/mandelbrot/py.typed}

```

Now we can put the `BoundingBox` class in a module.

``` {.python file=src/python/mandelbrot/bounding_box.py}
from dataclasses import dataclass
from typing import Optional
import numpy as np

<<complex-from-string-py>>
<<bounding-box-py>>
```

## Plotting the fractal

``` {.python file=src/python/mandelbrot/plotting.py}
import matplotlib  # type:ignore
matplotlib.use(backend="Agg")
from matplotlib import pyplot as plt
import numpy as np

from .bounding_box import BoundingBox

def plot_fractal(box: BoundingBox, values: np.ndarray, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = None
    plot_extent = (box.width + 1j * box.height) * box.scale
    z1 = box.center - plot_extent / 2
    z2 = z1 + plot_extent
    ax.imshow(values, origin='lower', extent=(z1.real, z2.real, z1.imag, z2.imag),
              cmap=matplotlib.colormaps["jet"])
    ax.set_xlabel("$\Re(c)$")
    ax.set_ylabel("$\Im(c)$")
    return fig, ax
```

## Reading config

``` {.python file=src/python/mandelbrot/config.py}
from typing import Optional
from pathlib import Path
import toml

default_config = """
[default]
max_iter = 256
width = 256
height = 256
center = [-0.8, 0.0]
extent = [3.0, 3.0]
"""

def read_config(input_path: Optional[Path] = None):
    if input_path is None:
        return toml.loads(default_config)
    else:
        return toml.load(input_path)
```

## Timing
Python ships with a `sqlite3` module that we can use to store timing results the database.

``` {.python file=src/python/mandelbrot/timeit.py}
from contextlib import contextmanager
import logging
import time
from typing import Optional
import sqlite3

def create_table(db: sqlite3.Connection):
    db.execute("""
        create table if not exists "timings" (
            "program" text not null,
            "language" text not null,
            "case" text not null,
            "parallel" integer not null,
            "runtime" real not null)""")

@contextmanager
def timeit(program: str, what: str, parallel: int, db: Optional[sqlite3.Connection]):
    start = time.monotonic()
    try:
        yield
    finally:
        t = time.monotonic() - start
        if db is not None:
            db.execute(
                "insert into \"timings\" values (?, 'python', ?, ?, ?)",
                (program, what, parallel, t))
        logging.info(f"It took {t}s to run '{what}'.")
```

## Numba

``` {.python file=src/python/mandelbrot/numba.py}
import numba  # type:ignore
from numba import prange  # type:ignore
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging
import sqlite3

from numba_progress import ProgressBar  # type:ignore
from argh import arg, dispatch_command  # type:ignore

from .config import read_config
from .bounding_box import BoundingBox
from .plotting import plot_fractal
from .timeit import timeit, create_table

@numba.njit(nogil=True, parallel=True)
def compute_mandelbrot(width: int, height: int, center: complex, scale: complex, max_iter:int, progress):
    result = np.zeros((height, width), np.int64)
    for j in prange(height):
        for i in prange(width):
            c = center + (i - width // 2 + (j - height // 2) * 1j) * scale
            z = 0.0+0.0j
            for k in range(max_iter):
                z = z**2 + c
                if (z*z.conjugate()).real >= 4.0:
                    break
            result[j, i] = k
        progress.update(width)
    return result

def mangle_name(s: str) -> str:
    return s.replace(" ", "_").lower()

@arg("--input_config", "-i", help="Input parameters ini file")
@arg("--output-path", "-o", help="Output directory")
@arg("--database", "-db", help="Timings database")
def main(input_config:Optional[Path]=None,
         output_path:Path=Path("."),
         database: Union[Path,str] = ":memory:",
         log_level:str="INFO"):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    config = read_config(input_config)
    with ProgressBar(total=1, desc="compiling") as progress:
        compute_mandelbrot(1, 1, 0, 1+1j, 1, progress)

    for section in config:
        logging.info(f"Rendering {section}")
        box = BoundingBox.from_config(config[section])
        max_iter = int(config[section]["max_iter"])
        conn = sqlite3.connect(database)

        with conn:
            create_table(conn)
            with timeit("numba", section, 0, conn), \
                 ProgressBar(desc=section, total=box.width*box.height) as progress:
                result = compute_mandelbrot(
                    box.width, box.height, box.center, box.scale,
                    max_iter=max_iter, progress=progress)

        fig, _ = plot_fractal(box, np.log(result + 1))
        output_image = output_path / (mangle_name(section) + ".png")
        logging.info(f"Saving to {output_image}")
        fig.savefig(output_image, bbox_inches="tight")

if __name__ == "__main__":
    dispatch_command(main)
```

## Domain splitting
We split the computation into a set of sub-domains. The `BoundingBox.split()` method is designed such that if we deep-map the resulting list-of-lists, we can recombine the results using `numpy.block()`.

``` {.python #bounding-box-methods-py}
def split(self, n):
    """Split the domain in nxn subdomains, and return a grid of BoundingBoxes."""
    w = self.width // n
    h = self.height // n
    e = self.scale * w + self.scale * h * 1j
    x0 = self.center - e * (n / 2 - 0.5)
    return [[BoundingBox(w, h, x0 + i * e.real + j * e.imag * 1j, e)
             for i in range(n)]
            for j in range(n)]
```

To perform the computation in parallel, I went ahead and chose the most difficult path: `asyncio`. There are other ways to do this, setting up a number of threads, or use Dask. However, `asyncio` is available to us in Python natively. In the end, the result is very similar to what I would do were I using `dask.delayed`.

``` {.python file=src/python/mandelbrot/domain_splitting.py}
import numba  # type:ignore
import numpy as np
from pathlib import Path
from typing import Optional, AsyncContextManager, Union
import asyncio
import logging
from psutil import cpu_count  # type:ignore
from contextlib import nullcontext
import sqlite3

from numba_progress import ProgressBar  # type:ignore
from argh import arg, dispatch_command  # type:ignore

from .config import read_config
from .bounding_box import BoundingBox
from .plotting import plot_fractal
from .timeit import timeit, create_table

@numba.njit(nogil=True)
def compute_mandelbrot_numba(result, width: int, height: int, center: complex, scale: complex, max_iter:int, progress):
    for j in range(height):
        for i in range(width):
            c = center + (i - width // 2 + (j - height // 2) * 1j) * scale
            z = 0.0+0.0j
            for k in range(max_iter):
                z = z**2 + c
                if (z*z.conjugate()).real >= 4.0:
                    break
            result[j, i] = k
        progress.update(width)
    return result

async def compute_mandelbrot(
        box: BoundingBox,
        max_iter: int,
        progress: ProgressBar,
        semaphore: Optional[asyncio.Semaphore]):
    result = np.zeros((box.height, box.width), np.int64)
    async with semaphore or nullcontext():
        await asyncio.to_thread(compute_mandelbrot_numba,
            result, box.width, box.height, box.center, box.scale,
            max_iter=max_iter, progress=progress)
    return result

def mangle_name(s: str) -> str:
    return s.replace(" ", "_").lower()


async def main(
        config,
        output_path: Path,
        throttle: Optional[int] = None,
        conn: Optional[sqlite3.Connection] = None):
    sem = asyncio.Semaphore(throttle) if throttle is not None else None
    n_cpus = cpu_count(logical=True)

    with ProgressBar(total=1, desc="compiling") as progress:
        await compute_mandelbrot(BoundingBox(1, 1, 0, 1+1j), 1, progress, None)

    for section in config:
        logging.info(f"Rendering {section}")
        box = BoundingBox.from_config(config[section])
        max_iter = int(config[section]["max_iter"])

        split = box.split(n_cpus)
        with timeit("domain_splitting", section, throttle or n_cpus, conn), \
             ProgressBar(total=box.width*box.height) as progress:
            split_result = await asyncio.gather(
                *(asyncio.gather(
                    *(compute_mandelbrot(b, max_iter, progress, sem)
                      for b in row))
                  for row in split))
            result = np.block(split_result)

        fig, _ = plot_fractal(box, np.log(result + 1))
        output_image = output_path / (mangle_name(section) + ".png")
        logging.info(f"Saving to {output_image}")
        fig.savefig(output_image, bbox_inches="tight")


@arg("--input_config", "-i", help="Input parameters ini file")
@arg("--output-path", "-o", help="Output directory")
@arg("--throttle", "-j", help="Throttle number of cpu's used")
@arg("--database", "-db", help="Timings database")
def command_line(
        input_config: Optional[Path] = None,
        output_path: Path = Path("."), 
        throttle: Optional[int] = None,
        database: Union[Path,str] = ":memory:",
        log_level: str = "INFO"):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    conn = sqlite3.connect(database)
    throttle = int(throttle) if throttle is not None else None
    with conn:
        create_table(conn)
        asyncio.run(main(read_config(input_config), output_path, throttle, conn))

if __name__ == "__main__":
    dispatch_command(command_line)
```

### Numba vectorize

``` {.python #bounding-box-methods-py}
def grid(self):
    """Return the complex values on the grid in a 2d array."""
    x0 = self.center - self.extent / 2
    x1 = self.center + self.extent / 2
    g = np.mgrid[x0.imag:x1.imag:self.height*1j,
                 x0.real:x1.real:self.width*1j]
    return g[1] + g[0]*1j
```

``` {.python file=src/python/mandelbrot/vectorized.py}
from numba import guvectorize, int64, complex128  # type:ignore
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging
import sqlite3

from numba_progress import ProgressBar  # type:ignore
from argh import arg, dispatch_command  # type:ignore

from .config import read_config
from .bounding_box import BoundingBox
from .plotting import plot_fractal
from .timeit import timeit, create_table

@guvectorize([(complex128[:,:], int64, int64[:,:])],
             "(n,m),()->(n,m)",
             nopython=True, parallel=True)
def compute_mandelbrot_numba(inp, max_iter:int, result):
    for j in range(inp.shape[0]):
        for i in range(inp.shape[1]):
            c = inp[j,i]
            z = 0.0+0.0j
            for k in range(max_iter):
                z = z**2 + c
                if (z*z.conjugate()).real >= 4.0:
                    break
            result[j,i] = k

def compute_mandelbrot(
        box: BoundingBox, max_iter: int,
        progress: ProgressBar):
    result = np.zeros((box.height, box.width), np.int64)
    c = box.grid()
    compute_mandelbrot_numba(c, max_iter, result)
    progress.update(c.size)
    return result

def mangle_name(s: str) -> str:
    return s.replace(" ", "_").lower()

def main(config, output_path: Path, conn: sqlite3.Connection):
    for section in config:
        logging.info(f"Rendering {section}")
        box = BoundingBox.from_config(config[section])
        max_iter = int(config[section]["max_iter"])

        with timeit("vectorized", section, 0, conn), \
             ProgressBar(total=box.width*box.height) as progress:
            result = compute_mandelbrot(box, max_iter, progress)

        fig, _ = plot_fractal(box, np.log(result + 1))
        output_image = output_path / (mangle_name(section) + ".png")
        logging.info(f"Saving to {output_image}")
        fig.savefig(output_image, bbox_inches="tight")

@arg("--input_config", "-i", help="Input parameters ini file")
@arg("--output-path", "-o", help="Output directory")
@arg("--database", "-db", help="Timings database")
def command_line(
        input_config: Optional[Path] = None,
        output_path: Path = Path("."), 
        database: Union[Path,str] = ":memory:",
        log_level: str = "INFO"):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    conn = sqlite3.connect(database)
    with conn:
        create_table(conn)
        main(read_config(input_config), output_path, conn)

if __name__ == "__main__":
    dispatch_command(command_line)
```

