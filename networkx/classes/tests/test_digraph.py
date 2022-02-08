import pytest

import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import TestBaseGraph, TestBaseAttrGraph
from .test_graph import TestGraph as _TestGraph
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraphUpdate
from networkx.utils import graphs_equal

class DiGraphSetup:
    
    def setup_method(self):

        # base class for tests inherited from test_graph
        self.Graph = nx.DiGraph

        # Complete directed graph on 3 nodes
        self.K3 = nx.complete_graph(3, nx.DiGraph)

        # Create a single-linked path graph on 3 nodes.
        self.P3 = nx.path_graph(3, nx.DiGraph)
        

        # Create a five node doubly-linked path graph /w attributes.
        self.G = nx.DiGraph(nx.path_graph(5))
        for i in range(5):
            self.G.nodes[i]["name"] = f"node{i}"
        self.G.edges[0, 1]["name"] = "edge01"
        self.G.edges[3, 4]["name"] = "edge34"
        self.G.graph["name"] = "graph"

        # Get the subgraph induced by the first and last edges.
        self.H = self.G.edge_subgraph([(0, 1), (3, 4)])

class TestBaseDiGraph(DiGraphSetup, TestBaseGraph):

    def test_has_successor(self):
        G = self.K3
        assert G.has_successor(0, 1)
        assert not G.has_successor(0, -1)

    def test_successors(self):
        G = self.K3
        assert sorted(G.successors(0)) == [1, 2]
        with pytest.raises(nx.NetworkXError):
            G.successors(-1)

    def test_has_predecessor(self):
        G = self.K3
        assert G.has_predecessor(0, 1)
        assert not G.has_predecessor(0, -1)

    def test_predecessors(self):
        G = self.K3
        assert sorted(G.predecessors(0)) == [1, 2]
        with pytest.raises(nx.NetworkXError):
            G.predecessors(-1)

    def test_edges(self):
        G = self.K3
        assert sorted(G.edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.edges(0)) == [(0, 1), (0, 2)]
        assert sorted(G.edges([0, 1])) == [(0, 1), (0, 2), (1, 0), (1, 2)]
        with pytest.raises(nx.NetworkXError):
            G.edges(-1)

    def test_out_edges(self):
        G = self.K3
        assert sorted(G.out_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.out_edges(0)) == [(0, 1), (0, 2)]
        with pytest.raises(nx.NetworkXError):
            G.out_edges(-1)

    def test_out_edges_dir(self):
        G = self.P3
        assert sorted(G.out_edges()) == [(0, 1), (1, 2)]
        assert sorted(G.out_edges(0)) == [(0, 1)]
        assert sorted(G.out_edges(2)) == []

    def test_out_edges_data(self):
        G = nx.DiGraph([(0, 1, {"data": 0}), (1, 0, {})])
        assert sorted(G.out_edges(data=True)) == [(0, 1, {"data": 0}), (1, 0, {})]
        assert sorted(G.out_edges(0, data=True)) == [(0, 1, {"data": 0})]
        assert sorted(G.out_edges(data="data")) == [(0, 1, 0), (1, 0, None)]
        assert sorted(G.out_edges(0, data="data")) == [(0, 1, 0)]

    def test_in_edges_dir(self):
        G = self.P3
        assert sorted(G.in_edges()) == [(0, 1), (1, 2)]
        assert sorted(G.in_edges(0)) == []
        assert sorted(G.in_edges(2)) == [(1, 2)]

    def test_in_edges_data(self):
        G = nx.DiGraph([(0, 1, {"data": 0}), (1, 0, {})])
        assert sorted(G.in_edges(data=True)) == [(0, 1, {"data": 0}), (1, 0, {})]
        assert sorted(G.in_edges(1, data=True)) == [(0, 1, {"data": 0})]
        assert sorted(G.in_edges(data="data")) == [(0, 1, 0), (1, 0, None)]
        assert sorted(G.in_edges(1, data="data")) == [(0, 1, 0)]

    def test_degree(self):
        G = self.K3
        assert sorted(G.degree()) == [(0, 4), (1, 4), (2, 4)]
        assert dict(G.degree()) == {0: 4, 1: 4, 2: 4}
        assert G.degree(0) == 4
        assert list(G.degree(iter([0]))) == [(0, 4)]  # run through iterator

    def test_in_degree(self):
        G = self.K3
        assert sorted(G.in_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.in_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.in_degree(0) == 2
        assert list(G.in_degree(iter([0]))) == [(0, 2)]  # run through iterator

    def test_out_degree(self):
        G = self.K3
        assert sorted(G.out_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.out_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.out_degree(0) == 2
        assert list(G.out_degree(iter([0]))) == [(0, 2)]

    def test_size(self):
        G = self.K3
        assert G.size() == 6
        assert G.number_of_edges() == 6

    def test_to_undirected_reciprocal(self):
        G = nx.DiGraph()
        G.add_edge(1, 2)
        assert G.to_undirected().has_edge(1, 2)
        assert not G.to_undirected(reciprocal=True).has_edge(1, 2)
        G.add_edge(2, 1)
        assert G.to_undirected(reciprocal=True).has_edge(1, 2)

    def test_reverse_copy(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        R = G.reverse()
        assert sorted(R.edges()) == [(1, 0), (2, 1)]
        R.remove_edge(1, 0)
        assert sorted(R.edges()) == [(2, 1)]
        assert sorted(G.edges()) == [(0, 1), (1, 2)]

    def test_reverse_nocopy(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        R = G.reverse(copy=False)
        assert sorted(R.edges()) == [(1, 0), (2, 1)]
        with pytest.raises(nx.NetworkXError):
            R.remove_edge(1, 0)

    def test_reverse_hashable(self):
        class Foo:
            pass

        x = Foo()
        y = Foo()
        G = nx.DiGraph()
        G.add_edge(x, y)
        assert nodes_equal(G.nodes(), G.reverse().nodes())
        assert [(y, x)] == list(G.reverse().edges())


class TestBaseAttrDiGraph(DiGraphSetup, TestBaseAttrGraph):

    def test_in_degree_weighted(self):
        G = self.K3.copy()
        G.add_edge(0, 1, weight=0.3, other=1.2)
        assert sorted(G.in_degree(weight="weight")) == [(0, 2), (1, 1.3), (2, 2)]
        assert dict(G.in_degree(weight="weight")) == {0: 2, 1: 1.3, 2: 2}
        assert G.in_degree(1, weight="weight") == 1.3
        assert sorted(G.in_degree(weight="other")) == [(0, 2), (1, 2.2), (2, 2)]
        assert dict(G.in_degree(weight="other")) == {0: 2, 1: 2.2, 2: 2}
        assert G.in_degree(1, weight="other") == 2.2
        assert list(G.in_degree(iter([1]), weight="other")) == [(1, 2.2)]

    def test_out_degree_weighted(self):
        G = self.K3.copy()
        G.add_edge(0, 1, weight=0.3, other=1.2)
        assert sorted(G.out_degree(weight="weight")) == [(0, 1.3), (1, 2), (2, 2)]
        assert dict(G.out_degree(weight="weight")) == {0: 1.3, 1: 2, 2: 2}
        assert G.out_degree(0, weight="weight") == 1.3
        assert sorted(G.out_degree(weight="other")) == [(0, 2.2), (1, 2), (2, 2)]
        assert dict(G.out_degree(weight="other")) == {0: 2.2, 1: 2, 2: 2}
        assert G.out_degree(0, weight="other") == 2.2
        assert list(G.out_degree(iter([0]), weight="other")) == [(0, 2.2)]


class TestDiGraph(DiGraphSetup, _TestGraph):
    """Tests specific to dict-of-dict-of-dict digraph data structure"""

    def test_data_input(self):
        G = nx.DiGraph({1: [2], 2: [1]}, name="test")
        assert G.name == "test"
        assert sorted(G.adj.items()) == [(1, {2: {}}), (2, {1: {}})]
        assert sorted(G.succ.items()) == [(1, {2: {}}), (2, {1: {}})]
        assert sorted(G.pred.items()) == [(1, {2: {}}), (2, {1: {}})]

    def test_add_edge(self):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        assert G.adj == {0: {1: {}}, 1: {}}
        assert G.succ == {0: {1: {}}, 1: {}}
        assert G.pred == {0: {}, 1: {0: {}}}
        G = nx.DiGraph()
        G.add_edge(*(0, 1))
        assert G.adj == {0: {1: {}}, 1: {}}
        assert G.succ == {0: {1: {}}, 1: {}}
        assert G.pred == {0: {}, 1: {0: {}}}

    def test_add_edges_from(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (0, 2, {"data": 3})], data=2)
        assert G.adj == {0: {1: {"data": 2}, 2: {"data": 3}}, 1: {}, 2: {}}
        assert G.succ == {0: {1: {"data": 2}, 2: {"data": 3}}, 1: {}, 2: {}}
        assert G.pred == {0: {}, 1: {0: {"data": 2}}, 2: {0: {"data": 3}}}

        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0,)])  # too few in tuple
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0, 1, 2, 3)])  # too many in tuple
        with pytest.raises(TypeError):
            G.add_edges_from([0])  # not a tuple

    def test_remove_edge(self):
        G = self.K3.copy()
        G.remove_edge(0, 1)
        assert G.succ == {0: {2: {}}, 1: {0: {}, 2: {}}, 2: {0: {}, 1: {}}}
        assert G.pred == {0: {1: {}, 2: {}}, 1: {2: {}}, 2: {0: {}, 1: {}}}
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)

    def test_remove_edges_from(self):
        G = self.K3.copy()
        G.remove_edges_from([(0, 1)])
        assert G.succ == {0: {2: {}}, 1: {0: {}, 2: {}}, 2: {0: {}, 1: {}}}
        assert G.pred == {0: {1: {}, 2: {}}, 1: {2: {}}, 2: {0: {}, 1: {}}}
        G.remove_edges_from([(0, 0)])  # silent fail

    def test_clear(self):
        G = self.K3
        G.graph["name"] = "K3"
        G.clear()
        assert list(G.nodes) == []
        assert G.succ == {}
        assert G.pred == {}
        assert G.graph == {}

    def test_clear_edges(self):
        G = self.K3
        G.graph["name"] = "K3"
        nodes = list(G.nodes)
        G.clear_edges()
        assert list(G.nodes) == nodes
        expected = {0: {}, 1: {}, 2: {}}
        assert G.succ == expected
        assert G.pred == expected
        assert list(G.edges) == []
        assert G.graph["name"] == "K3"

    def test_edges_data(self):
        G = self.K3
        # One node
        assert sorted(G.edges(0, data=True)) == [
            (0, 1, {}), (0, 2, {}),
        ]
        # Multiple nodes
        assert sorted(G.edges([0, 1], data=True)) == [
            (0, 1, {}), (0, 2, {}),
            (1, 0, {}), (1, 2, {}),
        ]
        # All edges
        assert sorted(G.edges(data=True)) == [
            (0, 1, {}), (0, 2, {}),
            (1, 0, {}), (1, 2, {}),
            (2, 0, {}), (2, 1, {}),
        ]
        # Nonexistent node
        with pytest.raises(nx.NetworkXError):
            G.edges(-1, True)


class TestEdgeSubgraph(DiGraphSetup, _TestGraphEdgeSubgraph):
    """Unit tests for the :meth:`DiGraph.edge_subgraph` method."""

    def test_pred_succ(self):
        """Test that nodes are added to predecessors and successors.

        For more information, see GitHub issue #2370.

        """
        G = nx.DiGraph()
        G.add_edge(0, 1)
        H = G.edge_subgraph([(0, 1)])
        assert list(H.predecessors(0)) == []
        assert list(H.successors(0)) == [1]
        assert list(H.predecessors(1)) == [0]
        assert list(H.successors(1)) == []


class TestDiGraphUpdate(DiGraphSetup, TestGraphUpdate):
    
    nlist = [
        (0, {}),
        (1, {}),
        (2, {}),
        (3, {}),
        (4, {"size": 2}),
        (5, {}),
        (6, {}),
        (7, {}),
    ]

    elist = [
        (0, 1, {}),
        (0, 2, {}),
        (1, 0, {}),
        (1, 2, {}),
        (2, 0, {}),
        (2, 1, {}),
        (4, 5, {}),
        (6, 7, {"weight": 2}),
    ]