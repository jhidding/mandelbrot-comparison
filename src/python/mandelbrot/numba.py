# ~\~ language=Python filename=src/python/mandelbrot/numba.py
# ~\~ begin <<docs/src/python.md|src/python/mandelbrot/numba.py>>[init]
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
# ~\~ end
