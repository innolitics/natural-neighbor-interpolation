# Natural Neighbor Interpolation

## Development Instructions

- `make module` compiles the module with `python setup.py build`.

- `make demo` copies the compiled library into the current working directory
  and calls a demo script

- `make clean` removes the build directory

## Dependencies

- Python dependencies in `requirements.txt`

## TODO:

- Change arguments so that you don't need to pass in a full grid (i.e. change
  from the `griddata` API); this is good because it reduces memory footprint
  and, unlike other griddata methods, I think we really need our interpolated
  points to lie on a grid.
- Add a bunch of tests
- Check that the input dimensions are correct
- Add option to avoid extrapolation
- Suppoart floats and doubles
- Float multiple dimensions
- Add documentation with discussion on limitations of discrete sibson's method
