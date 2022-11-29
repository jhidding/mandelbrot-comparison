.PHONY: docs timings

input := data/mandelbrot.ini
timings := output/timings.db
python := poetry run python

$(timings): src/create_timings.sql
	@mkdir -p $(@D)
	sqlite3 $@ < $<

timings: $(timings)
	$(python) -m mandelbrot.numba -i $(input) -db $(timings)
	$(python) -m mandelbrot.domain_splitting -i $(input) -db $(timings)

docs:
	cd docs; \
	julia --project=.. --compile=min -O0 make.jl

all: docs
