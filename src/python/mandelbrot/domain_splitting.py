# ~\~ language=Python filename=src/python/mandelbrot/domain_splitting.py
# ~\~ begin <<docs/src/python.md|src/python/mandelbrot/domain_splitting.py>>[init]
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
# ~\~ end
