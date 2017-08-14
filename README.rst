Discrete Sibson (Natural Neighbor) Interpolation
================================================

Natural neighbor interpolation is a method for interpolating scattered data
(i.e. you know the values of a function at scattered locations).  It is often superior to linear barycentric interpolation, which is a commonly used method of interpolation provided by Scipy's `griddata` function.

There are several implementations of 2D natural neighbor interpolation in Python.  We needed a fast 3D implementation that could run without a GPU, so we wrote an implementation of Discrete Sibson Interpolation (a version of natural neighbor interpolation that is fast but introduces slight errors as compared to "geometric" natural neighbor interpolation).

See https://doi.org/10.1109/TVCG.2006.27 for details.

Future Work
-----------

- Change arguments so that you don't need to pass in a full grid (i.e. change
  from the `griddata` API); this is good because it reduces memory footprint
  and, unlike other griddata methods, I think we really need our interpolated
  points to lie on a grid.
- Add a bunch of tests
- Check that the input dimensions are correct
- Add option to avoid extrapolation
- Support floats and doubles
- Support 2D
- Support higher dimensions (?)
- Add documentation with discussion on limitations of discrete sibson's method
