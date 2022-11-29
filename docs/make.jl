push!(LOAD_PATH,"../src/")

using Documenter, Mandelbrot

function noweb_label_pass(src, target_path)
    mkpath(joinpath(target_path, dirname(src)))
    run(pipeline(src, `awk -f noweb_label_pass.awk`, joinpath(target_path, src)))
end

function is_markdown(path)
    splitext(path)[2] == ".md"
end

sources = filter(is_markdown, readdir("./src", join=true))
path = mktempdir(".")
noweb_label_pass.(sources, path)

makedocs(source = joinpath(path, "src"), sitename="Mandelbrot comparison")
deploydocs(
    repo = "github.com/jhidding/mandelbrot-comparison.git"
)
