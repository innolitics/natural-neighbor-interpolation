#include <cmath>

#include <vector>

#include "kdtree.h"
#include "geometry.h"
#include "naturalneighbor.h"

namespace naturalneighbor {

typedef geometry::Point <double, 3> Point;

std::vector<double> *natural_neighbor(std::vector<Point>& known_coordinates,
                                      std::vector<double>& known_values,
                                      std::vector<Point>& interpolation_points,
                                      int coord_max) {
    /*
     * Assumptions:
     *  - known_coordinates and known_values are in parallel
     *  - interpolation_points are regularly spaced and ordered
     *    such that X is the outermost loop, then Y, then Z.
     */
    kdtree::kdtree<double> *tree = new kdtree::kdtree<double>();
    for (std::size_t i = 0; i < known_coordinates.size(); i++) {
        tree->add(&known_coordinates[i], &known_values[i]);
    }
    tree->build();

    auto interpolation_values = new std::vector<double>(interpolation_points.size(), 0.0);
    auto contribution_counter = new std::vector<int>(interpolation_points.size(), 0);

    int xscale = coord_max*coord_max;
    int yscale = coord_max;

    // Scatter method discrete Sibson
    // For each interpolation point p, search neighboring interpolation points
    // within a sphere of radius r, where r = distance to nearest known point.
    for (std::size_t i = 0; i < interpolation_points.size(); i++) {
        const kdtree::QueryResult *q = tree->nearest_iterative(interpolation_points[i]);
        double comparison_distance = q->distance;
        int r = floor(comparison_distance);
        int px = interpolation_points[i][0];
        int py = interpolation_points[i][1];
        int pz = interpolation_points[i][2];
        // Search neighboring interpolation points within a bounding box
        // of r indices. From this subset of points, calculate their distance
        // and tally the ones that fall within the sphere of radius r surrounding
        // interpolation_points[i].
        for (auto x = px - r; x < px + r; x++) {
            if (x < 0 || x >= coord_max) { continue; }
            for (auto y = py - r; y < py + r; y++) {
                if (y < 0 || y >= coord_max) { continue; }
                for (auto z = pz - r; z < pz + r; z++) {
                    if (z < 0 || z >= coord_max) { continue; }
                    int idx = x*xscale + y*yscale + z;
                    double distance_x = interpolation_points[idx][0] - px;
                    double distance_y = interpolation_points[idx][1] - py;
                    double distance_z = interpolation_points[idx][2] - pz;
                    if (distance_x*distance_x + distance_y*distance_y
                            + distance_z*distance_z > comparison_distance){
                        continue;
                    }
                    (*interpolation_values)[idx] += q->value;
                    (*contribution_counter)[idx] += 1;
                }
            }
        }
    }

    for (std::size_t i = 0; i < interpolation_values->size(); i++) {
        if ((*contribution_counter)[i] != 0) {
            (*interpolation_values)[i] /= (*contribution_counter)[i];
        } else {
            (*interpolation_values)[i] = 0; //TODO: this is just 0, better way to mark NAN?
        }
    }

    delete tree;
    delete contribution_counter;

    return interpolation_values;
}

}  // namespace naturalneighbor
