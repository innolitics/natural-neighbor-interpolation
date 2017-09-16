#include "gtest/gtest.h"

#include "geometry.h"
#include "kdtree.h"

using Point = geometry::Point<double, 3>;
using QueryResult = kdtree::QueryResult;
using KDTree = kdtree::kdtree<double, Point>;

TEST(SimpleTest, Simple) {
    KDTree tree {};
    Point point {0, 0, 0};
    double value {7.0};
    tree.add(point, value);
    tree.build();

    QueryResult result = tree.nearest_iterative({0, 0, 10});
    EXPECT_EQ(100, result.distance);
    EXPECT_EQ(value, result.value);

    result = tree.nearest_iterative({-10, 0, 0});
    EXPECT_EQ(100, result.distance);
    EXPECT_EQ(value, result.value);

    result = tree.nearest_iterative({0, 0, 0});
    EXPECT_EQ(0, result.distance);
    EXPECT_EQ(value, result.value);
}
