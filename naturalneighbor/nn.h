#include <vector>

#include "geometry.h"

typedef geometry::Point <double, 3> Point;

std::vector<double> *natural_neighbor(std::vector<Point>& known_coordinates,
                                      std::vector<double>& known_values,
                                      std::vector<Point>& interpolation_points,
                                      int coord_max);
