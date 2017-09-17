#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <memory>
#include <limits>
#include <queue>

namespace geometry {

template <typename CoordinateType, std::size_t DimensionCount>
class Point {
public:
    Point() = default;

    Point(CoordinateType const& a, CoordinateType const& b) {
        values[0] = a;
        values[1] = b;
    }

    Point(CoordinateType const& a, CoordinateType const& b, CoordinateType const& c) {
        values[0] = a;
        values[1] = b;
        values[2] = c;
    }

    const CoordinateType comparable_distance(Point<CoordinateType, DimensionCount> const& p) const {
        CoordinateType result = 0;
        for (std::size_t i = 0; i < DimensionCount; i++) {
            result += (values[i] - p.values[i])*(values[i] - p.values[i]);
        }
        return result;
    }

    CoordinateType& operator[] (const int index) {
        return values[index];
    }

    CoordinateType operator[] (const int index) const {
        return this->values[index];
    }
private:
    CoordinateType values[DimensionCount];
};

}  // namespace geometry

#endif
