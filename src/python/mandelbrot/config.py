# ~\~ language=Python filename=src/python/mandelbrot/config.py
# ~\~ begin <<docs/src/python.md|src/python/mandelbrot/config.py>>[init]
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
# ~\~ end
