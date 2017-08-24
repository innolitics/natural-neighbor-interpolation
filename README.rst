.. image:: https://travis-ci.org/innolitics/natural-neighbor-interpolation.svg?branch=master
   :target: https://travis-ci.org/innolitics/natural-neighbor-interpolation

Discrete Sibson (Natural Neighbor) Interpolation
================================================

Natural neighbor interpolation is a method for interpolating scattered data
(i.e. you know the values of a function at scattered locations).  It is often superior to linear barycentric interpolation, which is a commonly used method of interpolation provided by Scipy's `griddata` function.

There are several implementations of 2D natural neighbor interpolation in Python.  We needed a fast 3D implementation that could run without a GPU, so we wrote an implementation of Discrete Sibson Interpolation (a version of natural neighbor interpolation that is fast but introduces slight errors as compared to "geometric" natural neighbor interpolation).

See https://doi.org/10.1109/TVCG.2006.27 for details.

Dependencies
------------

- Python 3.4+
- Numpy (has been tested with 1.13+)

Demonstration
-------------

Natural neighbor interpolation can be more accurate than linear barycentric interpolation (Scipy's default) for smoothly varying functions.

Also, the final result looks better.

.. image:: https://raw.githubusercontent.com/innolitics/natural-neighbor-interpolation/master/demo/linear_comparison.png
   :target: https://raw.githubusercontent.com/innolitics/natural-neighbor-interpolation/master/demo/linear_comparison.png


.. image:: https://raw.githubusercontent.com/innolitics/natural-neighbor-interpolation/master/demo/sin_sin_comparison.png
   :target: https://raw.githubusercontent.com/innolitics/natural-neighbor-interpolation/master/demo/sin_sin_comparison.png

Future Work
-----------

- Provide options for extrapolation handling
- Support floats and complex numbers (only support doubles at the moment)
- Support 2D (only support 3D)
- Add documentation with discussion on limitations of discrete sibson's method
- Uncomment cpplint from tox.ini and cleanup C++ code
