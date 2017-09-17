#ifndef KDTREE_H_
#define KDTREE_H_

#include <memory>
#include <limits>
#include <queue>
#include <algorithm>

#include "geometry.h"

namespace kdtree {

// TODO: make this a public member type of kdtree
struct QueryResult {
    double value;
    double distance;
};

template <typename Data, typename Point = geometry::Point<double, 3>>
class kdtree {
public:
    kdtree() {}
    virtual ~kdtree() {}

    void add(Point point, Data data) {
        node_ptr node = std::make_shared<kdnode>(point, data);
        m_nodes.push_back(node);
    }

    void build() {
        if (!m_nodes.empty()) {
            m_root = build(m_nodes, 0);
        }
    }

    QueryResult nearest_iterative(const Point &query) const {
        if (m_root == nullptr) {
            throw std::exception {};
        }
        QueryResult best {m_root->data, std::numeric_limits<double>::max()};
        double num_contributions = 1;

        MinPriorityQueue pq {};
        pq.push(DistanceTuple(0, m_root));
        while (!pq.empty()) {
            DistanceTuple current = pq.top();
            if (current.distance > best.distance) {
                break;
            }
            pq.pop();

            node_ptr currentNode = current.node;
            Point splitPoint = currentNode->split;
            double d = query.comparable_distance(splitPoint);
            double dx = query[currentNode->axis] - splitPoint[currentNode->axis];
            if (d < best.distance) {
                num_contributions = 1;
                best.value = currentNode->data;
                best.distance = d;
            } else if (d == best.distance) {
                num_contributions++;
                best.value = best.value*(num_contributions - 1)/num_contributions +
                        currentNode->data/num_contributions;
            }
            node_ptr near = dx <= 0 ? currentNode->left : currentNode->right;
            node_ptr far = dx <= 0 ? currentNode->right : currentNode->left;
            if (far) {
                pq.push(DistanceTuple(dx * dx, far));
            }
            if (near) {
                pq.push(DistanceTuple(0, near));
            }
        }
        return best;
    }

private:
    struct kdnode {
        std::shared_ptr<kdnode> left;
        std::shared_ptr<kdnode> right;
        int axis;
        const Point split;
        const Data data;

        kdnode(const Point g, const Data d) : axis{0}, split{g}, data{d} {}
    };
    typedef std::shared_ptr<kdnode> node_ptr;
    typedef std::vector<node_ptr> Nodes;

    struct DistanceTuple {
        double distance;
        node_ptr node;
        DistanceTuple(double d, const node_ptr &n) : distance{d}, node{n} {}
    };

    struct SmallestOnTop {
        bool operator()(const DistanceTuple &a, const DistanceTuple &b) const {
            return a.distance > b.distance;
        }
    };
    typedef std::priority_queue<DistanceTuple, std::vector<DistanceTuple>, SmallestOnTop> MinPriorityQueue;

    template<typename NODE_TYPE>
    struct Sort : std::binary_function<NODE_TYPE, NODE_TYPE, bool> {
        Sort(std::size_t dim) : m_dimension(dim) {}
        bool operator()(const NODE_TYPE &lhs, const NODE_TYPE &rhs) const {
            Point lhsPoint = lhs->split;
            Point rhsPoint = rhs->split;
            return lhsPoint[m_dimension] - rhsPoint[m_dimension] < 0;
        }
        std::size_t m_dimension;
    };

    Nodes m_nodes;
    node_ptr m_root;

    node_ptr build(Nodes &nodes, int depth) {
        if (nodes.empty()) {
            return nullptr;
        }
        //int axis = depth % geometry::dimension<Point>();
        int axis = depth % 3;  // TODO: generalize over dimensions
        size_t median = nodes.size() / 2;
        std::nth_element(nodes.begin(), nodes.begin() + median, nodes.end(), Sort<node_ptr>(axis));

        Nodes left {nodes.begin(), nodes.begin() + median};
        Nodes right {nodes.begin() + median + 1, nodes.end()};

        node_ptr node = nodes.at(median);
        node->axis = axis;
        node->left = build(left, depth + 1);
        node->right = build(right, depth + 1);

        return node;
    }
};

} // namespace kdtree

#endif
