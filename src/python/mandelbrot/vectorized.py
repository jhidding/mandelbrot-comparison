# ~\~ language=Python filename=src/python/mandelbrot/vectorized.py
# ~\~ begin <<docs/src/python.md|src/python/mandelbrot/vectorized.py>>[init]
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
# ~\~ end
