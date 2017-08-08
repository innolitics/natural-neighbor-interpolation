#include <vector>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include "kdtree.h"

void natural_neighbor(std::vector<double>& known_points);

typedef boost::geometry::model::point <double, 3, boost::geometry::cs::cartesian> Point;

using namespace spatial_index;
std::vector<double> *natural_neighbor(std::vector<Point>& known_coordinates,
                                      std::vector<double>& known_values,
                                      std::vector<Point>& interpolation_points) {

    printf("Building KD-Tree\n");
    // Build KD-Tree from known points
    kdtree<double> *tree = new kdtree<double>();
    for (int i = 0; i < known_coordinates.size(); i ++) {
        tree->add(&known_coordinates[i], &known_values[i]);
    }
    tree->build();

    std::vector<double> accumulator(interpolation_points.size());
    std::vector<double> contribution_counter(interpolation_points.size());

    for (int i = 0; i < interpolation_points.size(); i++) {
        accumulator[i] = 0;
        contribution_counter[i] = 0;
    }

    printf("Calculating scattered contributions\n");
    // Scatter method discrete Sibson
    for (int i = 0; i < interpolation_points.size(); i++) {
        const QueryResult *q = tree->nearest_iterative(interpolation_points[i]);
        double r = q->distance;
        double comparison_distance = r*r;
        for (int j = 0; j < interpolation_points.size(); j++) {
            // Bounding Box Elimination
            // TODO: this is the performance bottleneck, ways to speed up?
            double distance_x = interpolation_points[j].get<0>() - interpolation_points[i].get<0>();
            double distance_y = interpolation_points[j].get<1>() - interpolation_points[i].get<1>();
            double distance_z = interpolation_points[j].get<2>() - interpolation_points[i].get<2>();
            if (distance_x*distance_x + distance_y*distance_y
                    + distance_z*distance_z > comparison_distance){
                continue;
            }
            accumulator[i] += q->value;
            contribution_counter[i] += 1;
        }
    }

    std::vector<double> interpolation_values(interpolation_points.size());

    printf("Calculating final interpolation values (%d values)\n", interpolation_values.size());
    for (int i = 0; i < interpolation_values.size(); i++) {
        if (contribution_counter[i] != 0) {
            interpolation_values[i] = accumulator[i] / contribution_counter[i];
        } else {
            interpolation_values[i] = NULL; //TODO: this is just 0, better way to mark NAN?
        }
    }
    return new std::vector<double>(interpolation_values);
}

typedef boost::geometry::model::point <float, 3, boost::geometry::cs::cartesian> point;


int main(int argc, char** argv) {
    /* point a = point(1,2,3); */
    /* point b= point(0,0,0); */
    /* double distance = boost::geometry::distance(a, b); */
    /* printf("%f\n", distance); */

    int coord_max = 20;
    int num_known_points = 5000;
    std::vector<Point> known_points(num_known_points);
    std::vector<double> known_values(num_known_points);
    for (int i = 0; i < num_known_points; i++) {
        known_points[i] = Point(std::rand(), std::rand(), std::rand());
        known_values[i] = std::rand();
    }
    /* interpolation_points */
    std::vector<Point> interpolation_points(coord_max*coord_max*coord_max);
    int idx = 0;
    for (int i = 0; i < coord_max; i++) {
        for (int j = 0; j < coord_max; j++) {
            for (int k = 0; k < coord_max; k++) {
                interpolation_points[idx] = Point(i, j, k);
                idx += 1;
            }
        }
    }
    std::vector<double>* interpolation_values = natural_neighbor(known_points,
                                                                 known_values,
                                                                 interpolation_points);
    printf("Completed with %d interpolated values\n", interpolation_values->size());
}
