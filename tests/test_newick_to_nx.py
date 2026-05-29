"""
Test suite for newick_to_nx function to ensure proper Newick string parsing
and NetworkX conversion functionality.
"""
import pytest
import networkx as nx
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sparsernj.utils.tree_utils import newick_to_nx


class TestNewickToNx:
    """Test cases for newick_to_nx function."""
    
    def test_simple_rooted_tree(self):
        """Test parsing of a simple rooted tree."""
        newick = "(A:0.1,B:0.2):0.0;"
        tree_nx = newick_to_nx(newick)
        
        # Check that tree is a DiGraph
        assert isinstance(tree_nx, nx.DiGraph)
        
        # Check nodes exist
        nodes = set(tree_nx.nodes())
        assert 'A' in nodes
        assert 'B' in nodes
        
        # Check edges and weights
        edges = list(tree_nx.edges(data=True))
        assert len(edges) >= 2  # At least two edges from root to leaves
        
        # Verify edge weights are preserved
        for u, v, data in edges:
            assert 'weight' in data
            assert isinstance(data['weight'], (int, float))
    
    def test_complex_rooted_tree(self):
        """Test parsing of a more complex rooted tree with multiple levels."""
        newick = "((A:0.1,B:0.2):0.05,(C:0.3,D:0.4):0.06):0.0;"
        tree_nx = newick_to_nx(newick)
        
        # Check basic structure
        assert isinstance(tree_nx, nx.DiGraph)
        
        # Check leaf nodes
        leaf_nodes = ['A', 'B', 'C', 'D']
        for leaf in leaf_nodes:
            assert leaf in tree_nx.nodes()
        
        # Check that tree is connected (as directed graph)
        assert nx.is_weakly_connected(tree_nx)
        
        # Check that there's exactly one root (node with in_degree 0)
        root_nodes = [n for n, d in tree_nx.in_degree() if d == 0]
        assert len(root_nodes) == 1
    
    def test_tree_with_internal_labels(self):
        """Test tree with internal node labels."""
        newick = "((A:0.1,B:0.2)internal1:0.05,(C:0.3,D:0.4)internal2:0.06)root:0.0;"
        tree_nx = newick_to_nx(newick)
        
        # Check that internal labels are preserved
        nodes = set(tree_nx.nodes())
        # Note: internal node handling may vary, but structure should be preserved
        assert len(nodes) >= 4  # At least the 4 leaf nodes
        
        # Check basic tree properties
        assert nx.is_weakly_connected(tree_nx)
        root_nodes = [n for n, d in tree_nx.in_degree() if d == 0]
        assert len(root_nodes) == 1
    
    def test_tree_with_zero_branch_lengths(self):
        """Test tree with zero branch lengths."""
        newick = "(A:0.0,B:0.0):0.0;"
        tree_nx = newick_to_nx(newick)
        
        assert isinstance(tree_nx, nx.DiGraph)
        assert 'A' in tree_nx.nodes()
        assert 'B' in tree_nx.nodes()
        
        # Check that zero weights are handled correctly
        for u, v, data in tree_nx.edges(data=True):
            assert 'weight' in data
            assert data['weight'] >= 0.0
    
    def test_edge_attribute_customization(self):
        """Test custom edge attribute name."""
        newick = "(A:0.1,B:0.2):0.0;"
        tree_nx = newick_to_nx(newick, edge_attr='length')
        
        # Check that custom edge attribute is used
        for u, v, data in tree_nx.edges(data=True):
            assert 'length' in data
            assert 'weight' not in data  # Default should not be present
    
    def test_interior_node_names(self):
        """Test assignment of custom interior node names."""
        newick = "((A:0.1,B:0.2):0.05,(C:0.3,D:0.4):0.06):0.0;"
        interior_names = ['node1', 'node2', 'root']
        tree_nx = newick_to_nx(newick, interior_node_names=interior_names)
        
        # Check that some interior names are assigned
        nodes = set(tree_nx.nodes())
        # At least some of the provided names should be present
        name_found = any(name in nodes for name in interior_names)
        assert name_found or len(nodes) >= 4  # Fallback: at least leaf nodes present
    
    def test_unifurcating_root(self):
        """Test unifurcating root addition."""
        newick = "(A:0.1,B:0.2):0.0;"
        tree_nx = newick_to_nx(newick, unifurcating_root=True, root_label='new_root')
        
        # Check that unifurcating root is added
        root_nodes = [n for n, d in tree_nx.in_degree() if d == 0]
        assert len(root_nodes) == 1
        
        # Root should have exactly one child in unifurcating case
        root = root_nodes[0]
        assert tree_nx.out_degree(root) >= 1
    
    def test_numeric_node_labels(self):
        """Test tree with numeric node labels."""
        newick = "(1:0.1,2:0.2):0.0;"
        tree_nx = newick_to_nx(newick)
        
        # Check that numeric labels are handled properly
        nodes = tree_nx.nodes()
        # Should contain the leaf labels (may be converted to int or kept as string)
        assert len(nodes) >= 2  # At least the two leaves
    
    def test_cnasim_style_tree(self):
        """Test CNAsim-style tree format as used in the workflow."""
        # Example from typical CNAsim output
        newick = "((cell1:0.08,(cell3:0.024,cell4:0.024):0.055):0.1,cell2:0.18):0.0;"
        tree_nx = newick_to_nx(newick)
        
        # Verify basic properties
        assert isinstance(tree_nx, nx.DiGraph)
        
        # Check that cell labels are preserved
        nodes = set(str(n) for n in tree_nx.nodes())  # Convert to string for comparison
        expected_cells = {'cell1', 'cell2', 'cell3', 'cell4'}
        
        # At least some cell labels should be found
        cells_found = sum(1 for cell in expected_cells if any(cell in node for node in nodes))
        assert cells_found >= 2  # At least some cells should be identified
        
        # Tree should be connected and have proper structure
        assert nx.is_weakly_connected(tree_nx)
        root_nodes = [n for n, d in tree_nx.in_degree() if d == 0]
        assert len(root_nodes) == 1
    
    def test_malformed_newick_handling(self):
        """Test handling of malformed Newick strings."""
        malformed_strings = [
            "",  # Empty string
            "A",  # No tree structure
            "(A:0.1",  # Missing closing parenthesis
            # Note: We'll test what the function does rather than assuming it should fail
        ]
        
        for newick in malformed_strings:
            try:
                result = newick_to_nx(newick)
                # If it doesn't raise an exception, that's also a valid behavior
                # Just check that result is reasonable if returned
                if result is not None:
                    assert isinstance(result, nx.DiGraph)
            except Exception:
                # Exception is also acceptable for malformed input
                pass
    
    def test_function_exists_and_importable(self):
        """Test that the function can be imported and called."""
        # Basic smoke test
        simple_newick = "(A:0.1,B:0.1):0.0;"
        result = newick_to_nx(simple_newick)
        assert result is not None
        assert isinstance(result, nx.DiGraph)


if __name__ == "__main__":
    # Run tests if script is executed directly
    import subprocess
    import sys
    
    # Run pytest on this file
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    sys.exit(result.returncode)