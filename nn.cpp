#include <vector>
#include <boost/geometry.hpp>

#include "kdtree.h"

using namespace spatial_index;

typedef boost::geometry::model::point <double, 3, boost::geometry::cs::cartesian> Point;
std::vector<double> *natural_neighbor(std::vector<Point>& known_coordinates,
                                      std::vector<double>& known_values,
                                      std::vector<Point>& interpolation_points,
                                      int coord_max) {

    printf("Building KD-Tree\n");

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
    int xscale = coord_max*coord_max;
    int yscale = coord_max;

    // Scatter method discrete Sibson
    for (int i = 0; i < interpolation_points.size(); i++) {
        if (i%10000 == 0){
            printf("\tPoint %d of %d\n", i, interpolation_points.size());
        }
        const QueryResult *q = tree->nearest_iterative(interpolation_points[i]);
        int r = (int) q->distance;
        double comparison_distance = r*r;
        int px = interpolation_points[i].get<0>();
        int py = interpolation_points[i].get<1>();
        int pz = interpolation_points[i].get<2>();
        // Search neighboring interpolation points within a bounding box
        // of r indices. From this subset of points, calculate their distance
        // and tally the ones that fall within the sphere of radius r surrounding
        // interpolation_points[i].
        // TODO: Bounds check above as well
        for (int x = px - r; x < px + r; x++) {
            if (x < 0) { continue; }
            for (int y = py - r; y < py + r; y++) {
                if (y < 0) { continue; }
                for (int z = pz - r; z < px + r; z++) {
                    if (z < 0) { continue; }
                    int idx = x*xscale + y*yscale + z;
                    double distance_x = interpolation_points[idx].get<0>() - px;
                    double distance_y = interpolation_points[idx].get<1>() - py;
                    double distance_z = interpolation_points[idx].get<2>() - pz;
                    if (distance_x*distance_x + distance_y*distance_y
                            + distance_z*distance_z > comparison_distance){
                        continue;
                    }
                    accumulator[i] += q->value;
                    contribution_counter[i] += 1;
                }
            }
        }
    }

    std::vector<double> interpolation_values(interpolation_points.size());
    printf("Calculating final interpolation values (%d values)\n", (int) interpolation_values.size());
    for (int i = 0; i < interpolation_values.size(); i++) {
        if (contribution_counter[i] != 0) {
            interpolation_values[i] = accumulator[i] / contribution_counter[i];
        } else {
            interpolation_values[i] = NULL; //TODO: this is just 0, better way to mark NAN?
        }
    }
    delete tree;
    return new std::vector<double>(interpolation_values);
}

typedef boost::geometry::model::point <double, 3, boost::geometry::cs::cartesian> point;

#define RAND_MAX 300

int main(int argc, char** argv) {
    int coord_max = RAND_MAX;
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
                                                                 interpolation_points,
                                                                 coord_max);

    printf("Completed with %d interpolated values\n", (int) interpolation_values->size());
    delete interpolation_values;
}
