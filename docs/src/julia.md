# Julia

``` {.julia file=src/Mandelbrot.jl}
module Mandelbrot

using TOML
using ProgressBars

struct BoundingBox
    width :: Int
    height :: Int
    center :: Complex{Float64}
    extent :: Complex{Float64}
end

function scale(box::BoundingBox)
    max(real(box.extent) / box.width, imag(box.extent) / box.height)
end

function BoundingBox(cfg :: Dict{String,Any})
    BoundingBox(cfg["width"], cfg["height"],
                Complex{Float64}(cfg["center"]...),
                Complex{Float64}(cfg["extent"]...))
end

function read_config(fname :: String)
    TOML.parsefile(fname)
end

function split(box :: BoundingBox, n :: Int)
    w = box.width ÷ n
    h = box.height ÷ n
    s = scale(box)
    e = s * w + s * h * 1im
    x0 = box.center - e * ((n - 1) / 2 + 1)
    [BoundingBox(w, h, x0 + i * real(e) + j * imag(e) * 1im, e)
     for i in 1:n, j in 1:n]
end

function iter_mb(c :: Complex{Float64}, max_iter :: Int)
    z = 0.0+0.0im
    for k in 1:max_iter
        z = z^2 + c
        real(z * conj(z)) >= 4.0 && return k 
    end
    max_iter
end

function compute_mandelbrot(box::BoundingBox, max_iter::Int)
    result = zeros(Int, box.height, box.width)
    for j in 1:box.height
        for i in 1:box.width
            c = box.center + (i - box.width ÷ 2 + (j - box.height ÷ 2) * 1im) * scale(box)
            result[j, i] = iter_mb(c, max_iter)
        end
    end
    result
end

function compute_threads(box::BoundingBox, max_iter::Int)
    result = zeros(Int, box.height, box.width)
    Threads.@threads for j in ProgressBar(1:box.height)
        for i in 1:box.width
            c = box.center + (i - box.width ÷ 2 + (j - box.height ÷ 2) * 1im) * scale(box)
            result[j, i] = iter_mb(c, max_iter)
        end
    end
    result
end

function print_matrix(io :: IO, m :: Matrix{T}) where T <: Number
    for row in eachrow(m)
        for item in row
            print(io, item, " ")
        end
        print(io, "\n")
    end
end

function compute_parallel(box::BoundingBox, max_iter::Int)
    n = Threads.nthreads()
    tasks = [Threads.@spawn compute_mandelbrot(b, max_iter)
             for b in split(box, n)]
    result = map(fetch, tasks)   # ProgressBar(tasks))
    hvcat(n, result...)
end
end
```
