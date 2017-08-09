#include <vector>
#include <boost/geometry.hpp>

typedef boost::geometry::model::point <double, 3, boost::geometry::cs::cartesian> Point;
std::vector<double> *natural_neighbor(std::vector<Point>& known_coordinates,
                                      std::vector<double>& known_values,
                                      std::vector<Point>& interpolation_points,
                                      int coord_max);
