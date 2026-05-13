import random

import networkx as nx
from collections import deque


class Node:

    def __init__(self, node_id, is_tip=False, label=None):
        self.id = node_id
        if is_tip and label is None:
            self.label = node_id
        else:
            self.label = label        # taxon label for tips
        self.neighbors = []          # undirected edges
        self.is_tip = is_tip          # leaf or collapsed subtree
        self.barrier = False          # temporary stop node (max 3 at a time)
        self._subtree_leaves = set()      # cache for dfs_leaves
        self._tips_count = None    # number of tips in subtree depends on tree state

    def set_tips_count(self, count: int):
        self._tips_count = count

    def get_tips_count(self) -> int:
        return self._tips_count

    def get_subtree_leaves(self):
        return self._subtree_leaves.copy()

    def set_subtree_leaves(self, subtree_leaves: set):
        self._subtree_leaves = subtree_leaves

    def __repr__(self):
        return (f"Node(id={self.id}, is_tip={self.is_tip}, label={self.label}, barrier={self.barrier}, tips_count={self._tips_count}, "
                f"leaves={self._subtree_leaves})")

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

class Tree:
    def __init__(self, root: Node):
        assert root.is_tip
        if root.label is None:
            root.label = root.id
        self.root = root              # special reference, edges still undirected
        self.nodes = {root.id: root}
        self.taxa_map = {}         # taxon label to node id
        self._centroid = None
        self._dirty = True            # mark tree changed
        # total tips and leaves computed during centroid finding
        self._total_tips = 1
        self._total_leaves = {root.label}

    # TODO: these methods are equal to the UTree class, therefore can be inherited
    # === START REPEATED METHODS ===
    def add_node(self, node: Node):
        # make an id for the node
        self.nodes[node.id] = node
        if node.is_tip and node.label is not None:
            self.taxa_map[node.label] = node.id
        self._dirty = True

    def add_edge(self, u: Node, v: Node):
        u.neighbors.append(v)
        v.neighbors.append(u)
        assert len(u.neighbors) <= 3, "Node cannot have more than 3 neighbors"
        assert len(v.neighbors) <= 3, "Node cannot have more than 3 neighbors"
        self._dirty = True

    def remove_edge(self, u: Node, v: Node):
        if v in u.neighbors:
            u.neighbors.remove(v)
        if u in v.neighbors:
            v.neighbors.remove(u)
        self._dirty = True

    # helper to get a fresh integer id
    def _next_id(self):
        return max(self.nodes.keys()) + 1

    def insert_taxon_between(self, taxon_label, u: Node, v: Node) -> Node:
        """Insert a new internal node between a centroid node `u` and another node `v`,
        attach a new tip with `taxon_label` to that internal node.
        """
        if v not in u.neighbors or u not in v.neighbors:
            raise ValueError("Nodes u and v must be neighbors")

        # remove existing edge u-v
        self.remove_edge(u, v)

        # create and register the new internal node
        new_internal_id = self._next_id()
        new_internal = Node(new_internal_id, is_tip=False)
        self.add_node(new_internal)

        # reconnect u <-> new_internal <-> v
        self.add_edge(u, new_internal)
        self.add_edge(new_internal, v)

        # create and attach the new taxon tip
        taxon_id = self._next_id()
        taxon_node = Node(taxon_id, is_tip=True, label=taxon_label)
        self.add_node(taxon_node)
        self.add_edge(new_internal, taxon_node)

        # update tree summary info
        self._total_tips += 1
        self._total_leaves.add(taxon_label)

        return new_internal

    def get_node_by_taxon(self, taxon_label) -> Node:
        return self.nodes[self.taxa_map[taxon_label]]

    # --- centroid logic ---

    def get_centroid(self):
        if self._dirty or self._centroid is None:
            self._centroid = self._compute_centroid()
            self._dirty = False
        return self._centroid

    def expand(self):
        """Release all cluster tips (barriers)."""
        for n in self.nodes.values():
            n.barrier = False
        self._dirty = True

    def bfs_edges(self):
        visited = set()
        queue = deque([self.root])
        visited.add(self.root.id)

        while queue:
            u = queue.popleft()
            for v in u.neighbors:
                if v.id not in visited:
                    visited.add(v.id)
                    queue.append(v)
                    yield u, v

    def _compute_tips_dfs(self, node: Node, parent: Node = None) -> tuple[int, set]:
        """Compute number of tips in subtree rooted at node. Performs post-order DFS from source=parent. Also caches the leaf labels."""
        # first call from root
        if parent is None:
            assert len(node.neighbors) == 1, "Root node must have exactly one neighbor"
            # move to the only neighbor
            ntips, leaves = self._compute_tips_dfs(node.neighbors[0], node)
            leaves.add(node.label)
            node.set_tips_count(ntips)
            node.set_subtree_leaves(leaves)
            return ntips + 1, leaves  # include root

        # end of recursion
        if node.is_tip:
            node.set_tips_count(1)
            node.set_subtree_leaves({node.label})
            # barriers do not exist when counting tips
            return node.get_tips_count(), node.get_subtree_leaves()

        # internal node
        count = 0
        leaves = set()
        for nbr in node.neighbors:
            if nbr is not parent:
                ntips, nbr_leaves = self._compute_tips_dfs(nbr, node)
                count += ntips
                leaves = leaves.union(nbr_leaves)

        node.set_tips_count(count)
        node.set_subtree_leaves(leaves)
        return count, node.get_subtree_leaves()

    def _walk_to_centroid(self, node: Node, parent: Node):
        """Walk down the tree to find centroid node. Assumes tips counts are computed."""
        # updates the tips count along the way so that next walks are faster
        # also updates the leaves cache
        assert not (node.is_tip or node.barrier), "A tip node cannot be a centroid"
        # parent node inherits tips count and leaves from grandparents
        parent.set_tips_count(self._total_tips - node.get_tips_count())
        parent.set_subtree_leaves(self._total_leaves.difference(node.get_subtree_leaves()))
        # check other two neighbors
        for nbr in node.neighbors:
            if nbr is not parent:
                nbr_tips = nbr.get_tips_count()
                if nbr_tips > self._total_tips // 2:
                    return self._walk_to_centroid(nbr, node)
        return node

    # --- cluster / barrier handling ---

    def mark_centroid_barrier_from(self, open_node: Node):
        # NOTE: changes the number of tips in the tree
        # this is called on a centroid node when a direction is chosen
        # open_node is the neighbor on the other side of the barrier
        curr_centroid = self._centroid
        curr_centroid.barrier = True
        for nbr in curr_centroid.neighbors:
            if nbr is not open_node:
                self._total_tips -= nbr.get_tips_count()
        self._total_tips += 1
        curr_centroid.set_tips_count(1)  # barrier counts as one tip
        curr_centroid.set_subtree_leaves(self._total_leaves.difference(open_node.get_subtree_leaves()))
        # the new centroid must be on the open_node side and the tree is kept clean
        self._centroid = self._walk_to_centroid(open_node, curr_centroid)

    # == END REPEATED METHODS ===

    def pick_AB_leaves(self, k: int = 1) -> tuple[tuple, tuple]:
        """
        Pick k leaves per direction (x2) going out of the centroid (directed away from the root).
        Also returns the neighbors of the centroid, with the first two matching the picked leaf and the third/last
         neighbor being the one toward the root.
        """
        # FIXME: mostly equal to UTree implementation, may be partially inherited
        centroid = self.get_centroid()
        leaves = []
        nbrs = []
        up_nbr = None
        for nbr in centroid.neighbors:
            # pick any leaf from the subtree
            subtree_leaves = nbr.get_subtree_leaves()
            if self.root.label in subtree_leaves:  # difference from UTree
                up_nbr = nbr
                continue  # skip the root side
            assert len(subtree_leaves) > 0, "Subtree must have at least one leaf"
            # sample k leaves at random without replacement
            leaves_labels = random.sample(list(subtree_leaves), k=min(k, len(subtree_leaves)))
            leaves.append(leaves_labels)
            nbrs.append(nbr)
        nbrs.append(up_nbr)
        assert len(leaves[0]) + len(leaves[1]) >= 2, "Centroid must have two children, each with at least one leaf"
        assert len(nbrs) == 3 and up_nbr is not None, "Centroid must have three neighbors"
        return tuple(leaves), tuple(nbrs)

    def _compute_centroid(self):
        """Compute centroid node of the tree."""
        self._total_tips, self._total_leaves = self._compute_tips_dfs(self.root) # include root
        # print(f"Total tips: {self._total_tips} leaves: {self._total_leaves}")
        return self._walk_to_centroid(self.root.neighbors[0], self.root)


    # --- export ---

    def to_networkx(self):
        """
        Convert the tree to a NetworkX graph.
        Nodes are labeled by their label if available, otherwise by their internal id.
        """
        # FIXME: mostly equal to UTree implementation, may be partially inherited
        import networkx as nx
        G = nx.DiGraph()

        root = self.root
        root_id = root.label if root.label is not None else f"__i{root.id}"
        G.add_node(root_id)
        for u, v in self.bfs_edges():
            u_id = u.label if u.label is not None else f"__i{u.id}"
            v_id = v.label if v.label is not None else f"__i{v.id}"
            G.add_node(v_id)
            G.add_edge(u_id, v_id)
        return G

    @classmethod
    def from_networkx(self, G):
        node_id = 0
        # check that G is arborescence and binary
        assert nx.is_arborescence(G)
        assert all(G.out_degree(n) <= 2 for n in G.nodes())
        root_label = [n for n in G.nodes() if G.in_degree(n) == 0][0]
        root_node = Node(node_id, is_tip=True, label=root_label)
        internal_nodes_map = {root_label: node_id}
        tree = Tree(root_node)
        for u, v in nx.bfs_edges(G, source=root_label):
            node_id += 1
            if G.degree(v) == 1:
                node = Node(node_id, is_tip=True, label=v)
            else:
                node = Node(node_id, is_tip=False)
            internal_nodes_map[v] = node_id  # temporary mapping to get parent id from graph edges
            parent_node = tree.nodes[internal_nodes_map[u]]
            tree.add_node(node)
            tree.add_edge(parent_node, node)
        return tree

    def __repr__(self):
        return f"Tree(root={self.root}, nodes={self.nodes})"


class UTree:
    """Represents an unrooted tree with lazy distance computation."""

    def __init__(self):
        """
        Initialize an empty tree (no nodes).
        """
        self.nodes = {}
        self.taxa_map = {}
        self._centroid = None
        self._dirty = True
        # total tips and leaves computed during centroid finding
        self._total_tips = 1
        self._total_leaves = set()

    def add_node(self, node: Node):
        """Register a new node in the tree."""
        self.nodes[node.id] = node
        if node.is_tip and node.label is not None:
            self.taxa_map[node.label] = node.id
        self._dirty = True

    def add_edge(self, u: Node, v: Node):
        """Add an undirected edge between u and v."""
        u.neighbors.append(v)
        v.neighbors.append(u)
        assert len(u.neighbors) <= 3, "Node cannot have more than 3 neighbors"
        assert len(v.neighbors) <= 3, "Node cannot have more than 3 neighbors"
        self._dirty = True

    def remove_edge(self, u: Node, v: Node):
        """Remove an edge between u and v."""
        if v in u.neighbors:
            u.neighbors.remove(v)
        if u in v.neighbors:
            v.neighbors.remove(u)
        self._dirty = True

    def _next_id(self):
        """Get a fresh node ID."""
        return max(self.nodes.keys()) + 1

    def get_node_by_taxon(self, taxon_label) -> Node:
        """Get a node by its taxon label."""
        return self.nodes[self.taxa_map[taxon_label]]

    # --- centroid logic ---

    def get_centroid(self):
        if self._dirty or self._centroid is None:
            self._centroid = self._compute_centroid()
            self._dirty = False
        # print(f"Found centroid: {self._centroid}")
        return self._centroid

    def expand(self):
        """Release all cluster tips (barriers)."""
        for n in self.nodes.values():
            n.barrier = False
        self._dirty = True

    def insert_taxon_between(self, taxon_label, u: Node, v: Node) -> Node:
        """
        Insert a new taxon between nodes u and v.

        Creates a new internal node between u and v, and attaches the new taxon to it.

        Parameters:
            taxon_label: Label of the new taxon
            u: First neighbor node
            v: Second neighbor node

        Returns:
            The new internal node
        """
        if v not in u.neighbors or u not in v.neighbors:
            raise ValueError("Nodes u and v must be neighbors")

        # Remove edge u-v
        self.remove_edge(u, v)

        # Create new internal node
        new_internal_id = self._next_id()
        new_internal = Node(new_internal_id, is_tip=False)
        self.add_node(new_internal)

        # Reconnect u <-> new_internal <-> v
        self.add_edge(u, new_internal)
        self.add_edge(new_internal, v)

        # Create and attach new taxon node
        taxon_id = self._next_id()
        taxon_node = Node(taxon_id, is_tip=True, label=taxon_label)
        self.add_node(taxon_node)
        self.add_edge(new_internal, taxon_node)

        # update tree summary info
        self._total_tips += 1
        self._total_leaves.add(taxon_label)

        return new_internal

    def pick_ABC_leaves(self, k: int) -> tuple[tuple, tuple]:
        """
        Pick 3 sets of k leaves going out of the centroid.
        Also returns the neighbors of the centroid matching these directions.
        """
        centroid = self.get_centroid()
        leaves = []  # list of 3 lists of leaf labels for each direction
        nbrs = []  # list of 3 neighbor nodes for each direction
        for nbr in centroid.neighbors:
            # pick any leaf from the subtree
            subtree_leaves = nbr.get_subtree_leaves()
            assert len(subtree_leaves) > 0, "Subtree must have at least one leaf"
            # sample k leaves at random without replacement
            leaves_labels = random.sample(list(subtree_leaves), k=min(k, len(subtree_leaves)))
            leaves.append(leaves_labels)
            nbrs.append(nbr)
        # print the tree edges
        # print("TREE EDGES:")
        # print([(u.id,v.id) for u,v in self.bfs_edges(centroid)])
        # print(f"centroid: {centroid} neighbors: {nbrs}")
        assert len(nbrs) == 3, f"Centroid must have three neighbors: nbrs={nbrs}"
        assert sum(len(l) for l in leaves) >= 3, "Centroid must have three neighbors, each with at least one leaf"
        return tuple(leaves), tuple(nbrs)


    def _compute_tips_dfs(self, node: Node, parent: Node = None) -> tuple[int, set]:
        """Compute number of tips in subtree starting from node.
         Performs post-order DFS from source=parent. Also caches the leaf labels."""
        # first call from start
        if parent is None:
            assert len(node.neighbors) == 1, "Root node must have exactly one neighbor"
            # move to the only neighbor
            ntips, leaves = self._compute_tips_dfs(node.neighbors[0], node)
            leaves.add(node.label)
            node.set_tips_count(ntips)
            node.set_subtree_leaves(leaves)
            return ntips + 1, leaves  # include root

        # end of recursion
        if node.is_tip:
            node.set_tips_count(1)
            node.set_subtree_leaves({node.label})
            # barriers do not exist when counting tips
            return node.get_tips_count(), node.get_subtree_leaves()

        # internal node
        count = 0
        leaves = set()
        for nbr in node.neighbors:
            if nbr is not parent:
                ntips, nbr_leaves = self._compute_tips_dfs(nbr, node)
                count += ntips
                leaves = leaves.union(nbr_leaves)

        node.set_tips_count(count)
        node.set_subtree_leaves(leaves)
        return count, node.get_subtree_leaves()

    def _walk_to_centroid(self, node: Node, parent: Node):
        """Walk down the tree to find centroid node. Assumes tips counts are computed."""
        # updates the tips count along the way so that next walks are faster
        # also updates the leaves cache
        assert not (node.is_tip or node.barrier), "A tip node cannot be a centroid"
        # parent node inherits tips count and leaves from grandparents
        parent.set_tips_count(self._total_tips - node.get_tips_count())
        parent.set_subtree_leaves(self._total_leaves.difference(node.get_subtree_leaves()))
        # check other two neighbors
        for nbr in node.neighbors:
            if nbr is not parent:
                nbr_tips = nbr.get_tips_count()
                if nbr_tips > self._total_tips // 2:
                    return self._walk_to_centroid(nbr, node)
        return node

    def _compute_centroid(self):
        """Compute centroid node of the tree."""
        # start from any tip
        start_node = next((n for n in self.nodes.values() if n.is_tip))
        assert len(start_node.neighbors) == 1, "Start node must have exactly one neighbor"
        # NOTE: can be implemented also by starting from any node
        self._total_tips, self._total_leaves = self._compute_tips_dfs(start_node)
        return self._walk_to_centroid(start_node.neighbors[0], start_node)

    # --- cluster / barrier handling ---

    def mark_centroid_barrier_from(self, open_node: Node):
        # NOTE: changes the number of tips in the tree
        # this is called on a centroid node when a direction is chosen
        # open_node is the neighbor on the other side of the barrier
        curr_centroid = self._centroid
        curr_centroid.barrier = True
        for nbr in curr_centroid.neighbors:
            if nbr is not open_node:
                self._total_tips -= nbr.get_tips_count()
        self._total_tips += 1
        curr_centroid.set_tips_count(1)  # barrier counts as one tip
        curr_centroid.set_subtree_leaves(self._total_leaves.difference(open_node.get_subtree_leaves()))
        # the new centroid must be on the open_node side and the tree is kept clean
        self._centroid = self._walk_to_centroid(open_node, curr_centroid)

    def to_networkx(self, root_node=None) -> nx.Graph | nx.DiGraph:
        """
        Convert the tree to a NetworkX graph

        Parameters:
            root_label: Optional label for the root taxon (outgroup rooting)

        Returns:
            NetworkX Graph object
        """
        G = nx.Graph() if root_node is None else nx.DiGraph()
        start_node = root_node if root_node is not None else next(iter(self.nodes.values()))
        start_id = start_node.label if start_node.label is not None else f"__i{start_node.id}"
        G.add_node(start_id)
        for u, v in self.bfs_edges(start_node):
            u_id = u.label if u.label is not None else f"__i{u.id}"
            v_id = v.label if v.label is not None else f"__i{v.id}"
            G.add_node(v_id)
            G.add_edge(u_id, v_id)
        return G

    @classmethod
    def from_networkx(self, G):
        node_id = 0
        # check that G is arborescence and binary
        assert nx.is_tree(G)
        assert all(G.degree(n) <= 3 for n in G.nodes())
        tree = UTree()
        # pick one random node as source
        source_label = next(iter(G.nodes()))
        tip = G.degree(source_label) == 1
        tree.add_node(Node(node_id, is_tip=tip, label=source_label if tip else None))
        internal_nodes_map = {source_label: node_id}  # mapping from graph node to tree node id
        for u, v in nx.bfs_edges(G, source=source_label):
            node_id += 1
            if G.degree(v) == 1:
                node = Node(node_id, is_tip=True, label=v)
            else:
                node = Node(node_id, is_tip=False)
            internal_nodes_map[v] = node_id  # temporary mapping to get parent id from graph edges
            parent_node = tree.nodes[internal_nodes_map[u]]
            tree.add_node(node)
            tree.add_edge(parent_node, node)
        return tree

    def bfs_edges(self, start_node):
        """Generate edges in BFS order starting from the given label."""
        visited = set()
        queue = deque([start_node])
        visited.add(start_node.id)
        while queue:
            u = queue.popleft()
            for v in u.neighbors:
                if v.id not in visited:
                    visited.add(v.id)
                    queue.append(v)
                    yield u, v

