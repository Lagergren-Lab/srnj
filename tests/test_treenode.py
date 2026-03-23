import unittest
from src.sparsernj.treenode import Tree, Node


class TestTreeNodeCase(unittest.TestCase):

    def test_init(self):
        root = Node(0, is_tip=True)
        tree = Tree(root)
        self.assertEqual(tree.root.id, 0)
        self.assertEqual(len(tree.nodes), 1)

    def test_add_node_and_edge(self):
        """
        0 -- 1 (edge between node 0 and node 1)
        """
        root = Node(0, is_tip=True)
        tree = Tree(root)
        node1 = Node(1, is_tip=True)
        tree.add_node(node1)
        tree.add_edge(root, node1)
        self.assertIn(1, tree.nodes)
        self.assertIn(node1, root.neighbors)
        self.assertIn(root, node1.neighbors)

    def test_insert_taxon_between(self):
        """
                            leaf3
                              |
        root -- __id2 -- new_internal -- leaf1
                   |
                 leaf2
        """
        root = Node(0, is_tip=True, label="root")
        leaf1 = Node(1, is_tip=True, label="leaf1")
        node2 = Node(2, is_tip=False)
        leaf2 = Node(3, is_tip=True, label="leaf2")
        tree = Tree(root)
        tree.add_node(node2)
        tree.add_node(leaf1)
        tree.add_node(leaf2)
        tree.add_edge(root, node2)
        tree.add_edge(node2, leaf1)
        tree.add_edge(node2, leaf2)

        centroid_node = tree.get_centroid()
        new_internal_node = tree.insert_taxon_between("leaf3", centroid_node, leaf1)
        taxon_node = tree.get_node_by_taxon("leaf3")  # to ensure taxon node is added
        self.assertIn(taxon_node.id, tree.nodes)
        self.assertEqual(len(tree.nodes), 6)  # root, node1, new internal node, taxon node
        self.assertIn(new_internal_node, centroid_node.neighbors)
        self.assertIn(new_internal_node, leaf1.neighbors)
        self.assertIn(new_internal_node, taxon_node.neighbors)

    def test_get_centroid(self):
        root = Node(0, is_tip=True)
        node1 = Node(1, is_tip=False)
        node2 = Node(2, is_tip=True)
        node3 = Node(3, is_tip=True)
        tree = Tree(root)
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        tree.add_edge(root, node1)
        tree.add_edge(node1, node2)
        tree.add_edge(node1, node3)

        centroid = tree.get_centroid()
        self.assertEqual(tree.root.get_tips_count(), 1)
        self.assertIsNotNone(centroid)
        self.assertIn(centroid.id, tree.nodes)
        self.assertEqual(centroid.id, 1)  # centroid should be node 1

    def test_barrier(self):
        root = Node(0, is_tip=True)
        node1 = Node(1, is_tip=False)
        node2 = Node(2, is_tip=False)
        node3 = Node(3, is_tip=True)
        node4 = Node(4, is_tip=True)
        node5 = Node(5, is_tip=True)
        tree = Tree(root)
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        tree.add_node(node4)
        tree.add_edge(root, node1)
        tree.add_edge(node1, node2)
        tree.add_edge(node1, node3)
        tree.add_edge(node2, node4)
        tree.add_edge(node2, node5)
        # get centroid
        centroid = tree.get_centroid()
        self.assertEqual(centroid.id, 1)
        self.assertEqual(tree._total_tips, 4)
        # check subtree leaves NOT FOR CENTROID: not relevant
        self.assertEqual(1, tree.root.get_tips_count())
        self.assertEqual({0}, tree.root.get_subtree_leaves())
        self.assertEqual(2, tree.nodes[2].get_tips_count())
        self.assertSetEqual({4, 5}, tree.nodes[2].get_subtree_leaves())
        # mark barrier
        tree.mark_centroid_barrier_from(tree.nodes[2])
        self.assertTrue(centroid.barrier)
        # check that tips, leaves and centroid are updated
        self.assertEqual(tree._total_tips, 3)  # two tips replaced by barrier
        self.assertSetEqual(tree._total_leaves, {0, 3, 4, 5})  # leaves unchanged
        self.assertEqual(centroid.get_tips_count(), 1)  # old centroid now has 1 tip (itself)
        self.assertSetEqual(centroid.get_subtree_leaves(), {0, 3})
        self.assertFalse(tree._dirty)
        new_centroid = tree.get_centroid()
        self.assertEqual(new_centroid.id, 2)
        print(new_centroid.get_subtree_leaves())

    def test_insert_node_updates_centroid(self):
        root = Node(0, is_tip=True)
        node1 = Node(1, is_tip=False)
        node2 = Node(2, is_tip=True)
        node3 = Node(3, is_tip=True)

        tree = Tree(root)
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        tree.add_edge(root, node1)
        tree.add_edge(node1, node2)
        tree.add_edge(node1, node3)

        self.assertTrue(tree._dirty)
        centroid = tree.get_centroid()
        self.assertFalse(tree._dirty)
        self.assertEqual(centroid.id, 1)

        # insert new taxon between root and node1
        tree.insert_taxon_between(5, root, node1)
        self.assertTrue(tree._dirty)
        new_expected_centroid = tree.nodes[5].neighbors[0]
        self.assertEqual(new_expected_centroid.id, 4)  # new internal node has id = len(nodes) before insertion
        new_centroid = tree.get_centroid()
        self.assertEqual(new_expected_centroid.id, new_centroid.id)

    def test_pick_ort_leaves(self):
        root = Node(0, is_tip=True)
        node1 = Node(1, is_tip=False)
        node2 = Node(2, is_tip=False)
        node3 = Node(3, is_tip=True)
        node4 = Node(4, is_tip=True)
        node5 = Node(5, is_tip=True)
        tree = Tree(root)
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        tree.add_node(node4)
        tree.add_edge(root, node1)
        tree.add_edge(node1, node2)
        tree.add_edge(node1, node3)
        tree.add_edge(node2, node4)
        tree.add_edge(node2, node5)

        centroid = tree.get_centroid()
        self.assertEqual(centroid.id, 1)
        ort_leaves, direction_nodes = tree.pick_AB_leaves()
        self.assertEqual(len(ort_leaves), 2)
        self.assertEqual(1, len(ort_leaves[0]))
        self.assertEqual(1, len(ort_leaves[1]))  # each ort leaves set will have one leaf (default k=1 leaf per side)
        self.assertIn(ort_leaves[0][0], [3, 4, 5])
        self.assertIn(ort_leaves[1][0], [3, 4, 5])
        self.assertNotEqual(ort_leaves[0], ort_leaves[1])
        self.assertEqual(set([n.id for n in direction_nodes]), set([0, 2, 3]))

    def test_tree_to_networkx(self):
        root = Node(0, is_tip=True, label="root")
        node1 = Node(1, is_tip=False)
        node2 = Node(2, is_tip=True, label="leaf2")
        node3 = Node(3, is_tip=True, label="leaf3")
        tree = Tree(root)
        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)
        tree.add_edge(root, node1)
        tree.add_edge(node1, node2)
        tree.add_edge(node1, node3)

        nx_tree = tree.to_networkx()
        self.assertEqual(len(nx_tree.nodes), 4)
        self.assertEqual(len(nx_tree.edges), 3)
        self.assertIn(('root', '__i1'), nx_tree.edges)
        self.assertIn("leaf2", nx_tree.nodes)
        self.assertIn("leaf3", nx_tree.nodes)
        # root should be labeled as "root", verify its indegree
        self.assertEqual(0, nx_tree.in_degree("root"))

    def test_tree_from_networkx(self):
        import networkx as nx
        nx_tree = nx.DiGraph()
        nx_tree.add_edge("root", "__i1")
        nx_tree.add_edge("__i1", "leaf2")
        nx_tree.add_edge("__i1", "leaf3")

        tree = Tree.from_networkx(nx_tree)
        self.assertEqual(len(tree.nodes), 4)
        self.assertIn("root", [n.label for n in tree.nodes.values()])
        self.assertIn("leaf2", [n.label for n in tree.nodes.values()])
        self.assertIn("leaf3", [n.label for n in tree.nodes.values()])
        internal_nodes = [n for n in tree.nodes.values() if not n.is_tip]
        self.assertEqual(1, len(internal_nodes))
        internal_node = internal_nodes[0]
        self.assertEqual(3, len(internal_node.neighbors))


if __name__ == '__main__':
    unittest.main()
