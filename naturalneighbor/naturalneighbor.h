#ifndef NATURALNEIGHBOR_H_
#define NATURALNEIGHBOR_H_

#include <vector>

#include "geometry.h"

namespace naturalneighbor {

typedef geometry::Point <double, 3> Point;

std::vector<double> *natural_neighbor(
        std::vector<Point>& known_coordinates,
        std::vector<double>& known_values,
        std::vector<Point>& interpolation_points,
        int coord_max);

} // namespace naturalneighbor

#endif /* NATURALNEIGHBOR_H_ */
