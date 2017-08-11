#ifndef KDTREE_H_
#define KDTREE_H_

#include <memory>
#include <limits>
#include <queue>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

namespace spatial_index {

typedef struct {
    double value;
    double distance;
} QueryResult;

namespace util {

// Some compile time recursion in order to get a dimension dynamically
template <typename Point, std::size_t Dimension, std::size_t Count>
struct dimension_extractor {
    static inline typename boost::geometry::default_distance_result<Point>::type subtract(const Point &p1, const Point &p2, std::size_t dim) {
        if (Dimension == dim) {
            return boost::geometry::get<Dimension>(p1) - boost::geometry::get<Dimension>(p2);
        }
        return dimension_extractor<Point, Dimension + 1, Count>::subtract(p1, p2, dim);
    }
};
// end recursion
template <typename Point, std::size_t Count>
struct dimension_extractor<Point, Count, Count> {
    static inline typename boost::geometry::default_distance_result<Point>::type subtract(const Point &p1, const Point &p2, std::size_t dim) {
    }
};

template <typename Point>
typename boost::geometry::default_distance_result<Point>::type subtract(const Point &p1, const Point &p2, std::size_t dim) {
    return dimension_extractor<Point, 0, boost::geometry::dimension<Point>::type::value>::subtract(p1, p2, dim);
}

} // namespace util

template <typename Data, typename Point = boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>>
class kdtree {
public:
    kdtree() {}
    virtual ~kdtree() {}
    void add(const Point *point, const Data *data) {
        typename kdnode::ptr node = std::make_shared<kdnode>(point, data);
        m_nodes.push_back(node);
    }
    void build() {
        if (m_nodes.empty()) {
            return;
        }
        m_root = build(m_nodes, 0);
    }
    void clear() {
        m_root.reset();
        m_nodes.clear();
    }
    const Data *nearest_recursive(const Point &query) const {
        if (!m_root) {
            return NULL;
        }
        best_match best(m_root, std::numeric_limits<double>::max());
        nearest(query, m_root, best);
        return best.node->data;
    }
    const QueryResult *nearest_iterative(const Point &query) const {
        if (!m_root) {
            return NULL;
        }
        MinPriorityQueue pq;
        best_match best(m_root, std::numeric_limits<double>::max());
        pq.push(DistanceTuple(0, m_root));
        while (!pq.empty()) {
            const auto current = pq.top();
            if (current.first >= best.distance) {
                QueryResult *result = new QueryResult();
                result->value = *(best.node->data);
                result->distance = best.distance;
                return result;
            }
            pq.pop();
            auto currentNode = current.second;
            double d = boost::geometry::comparable_distance(query, *currentNode->split); // no sqrt
            double dx = util::subtract(query, *currentNode->split, currentNode->axis);
            if (d < best.distance) {
                best.node = currentNode;
                best.distance = d;
            }
            node_ptr near = dx <= 0 ? currentNode->left : currentNode->right;
            node_ptr far = dx <= 0 ? currentNode->right : currentNode->left;
            if (far) pq.push(DistanceTuple(dx * dx, far));
            if (near) pq.push(DistanceTuple(0, near));
        }
        QueryResult *result = new QueryResult();
        result->value = *(best.node->data);
        result->distance = best.distance;
        return result;
    }
private:
    struct kdnode {
        typedef std::shared_ptr<kdnode> ptr;
        ptr left;
        ptr right;
        int axis;
        const Point *split;
        const Data *data;
        kdnode(const Point *g, const Data *d) : axis(0), split(g), data(d) {}
    };
    typedef typename kdnode::ptr node_ptr; // get rid of annoying typename
    typedef std::vector<node_ptr> Nodes;
    typedef std::pair<double, node_ptr> DistanceTuple;
    struct SmallestOnTop {
        bool operator()(const DistanceTuple &a, const DistanceTuple &b) const {
            return a.first > b.first;
        }
    };
    struct LargestOnTop {
        bool operator()(const DistanceTuple &a, const DistanceTuple &b) const {
            return a.first < b.first;
        }
    };
    typedef std::priority_queue<DistanceTuple, std::vector<DistanceTuple>, SmallestOnTop> MinPriorityQueue;
    typedef std::priority_queue<DistanceTuple, std::vector<DistanceTuple>, LargestOnTop> MaxPriorityQueue;
    Nodes m_nodes;
    node_ptr m_root;
    struct best_match {
        node_ptr node;
        double distance;
        best_match(const node_ptr &n, double d) : node(n), distance(d) {}
    };

    template<typename NODE_TYPE>
    struct Sort : std::binary_function<NODE_TYPE, NODE_TYPE, bool> {
        Sort(std::size_t dim) : m_dimension(dim) {}
        bool operator()(const NODE_TYPE &lhs, const NODE_TYPE &rhs) const {
            return util::subtract(*lhs->split, *rhs->split, m_dimension) < 0;
        }
        std::size_t m_dimension;
    };
    node_ptr build(Nodes &nodes, int depth) {
        if (nodes.empty()) {
            return node_ptr();
        }
        int axis = depth % boost::geometry::dimension<Point>();
        size_t median = nodes.size() / 2;
        std::nth_element(nodes.begin(), nodes.begin() + median, nodes.end(), Sort<node_ptr>(axis));
        node_ptr node = nodes.at(median);
        node->axis = axis;

        Nodes left(nodes.begin(), nodes.begin() + median);
        Nodes right(nodes.begin() + median + 1, nodes.end());
        node->left = build(left, depth + 1);
        node->right = build(right, depth + 1);

        return node;
    }

    static void nearest(const Point &query, const node_ptr &currentNode, best_match &best) {
        if (!currentNode) {
            return;
        }
        double d = boost::geometry::comparable_distance(query, *currentNode->split); // no sqrt
        double dx = util::subtract(query, *currentNode->split, currentNode->axis);
        if (d < best.distance) {
            best.node = currentNode;
            best.distance = d;
        }
        node_ptr near = dx <= 0 ? currentNode->left : currentNode->right;
        node_ptr far = dx <= 0 ? currentNode->right : currentNode->left;
        nearest(query, near, best);
        if ((dx * dx) >= best.distance) {
            return;
        }
        nearest(query, far, best);
    }
};

} // namespace spatial_index

#endif /* KDTREE_H_ */
