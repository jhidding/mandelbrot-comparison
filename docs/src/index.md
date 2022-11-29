# The scientific programming language comparison
What is the best language to start a new project? Criteria are:

- What do your colleagues use? This may be the most important question.
- Readability
- Efficiency
- Ease of development

I'll compare a few languages here: Python, C++, Rust, Julia. This comparison will be completely subjective. The implementations will have some common factors:

- Read input from a config file
- Define a bounding-box structure
- Compute in parallel
- Show a progress bar
- Plot result (either through plotting library, or Gnuplot)
- Store timings in Sqlite3 database

## Mandelbrot fractal
We have the following iteration:

```math
z_{i+1} = z_i^2 + c,
```

where both $z, c \in \mathbb{C}$. The question is, for some $c$ chosen on the complex plane, and $z_0 = 0$, does this sequence diverge or does it stay within a confined region? The way this is usually done, we check the smallest iteration number $i$ for which $|z_i| > 2$, since it can be shown that if any $|z_i| > 2$ for some $c$, the sequence will always diverge. We compute one complete image of the Mandelbrot fractal and one zoom, centered at $-1.1195+0.2718\mathi$ with an extent of $0.005 \times 0.005\mathi$.

``` {.toml file=data/mandelbrot.toml}
["Full image"]
max_iter = 256
width = 1024
height = 1024
center = [-0.8, 0.0]
extent = [3.0, 3.0]

[Zoom]
max_iter = 1024
width = 1024
height = 1024
center = [-1.1195, 0.2718]
extent = [0.005, 0.005]

["Deep zoom"]
max_iter = 5000000
width = 1920
height = 1024
center = [-0.743643887037151, 0.131825904205330]
extent = [3.0e-11, 3.0e-11]
```

``` {.sqlite3 file=src/create_timings.sql}
drop table if exists "timings"
create table "timings" (
    "program" text not null,
    "language" text not null,
    "case" text not null,
    "parallel" integer not null,
    "runtime" real not null
);
```
