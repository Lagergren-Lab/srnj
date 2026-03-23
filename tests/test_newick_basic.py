#!/usr/bin/env python3
"""
Basic test script to verify newick_to_nx functionality works correctly.
This is a simple verification script - for comprehensive tests see test_newick_to_nx.py
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_newick_parsing():
    """Test current newick parsing behavior"""
    print("Testing newick parsing functionality...")
    
    # Test newick strings from the test suite
    test_newicks = [
        "((A:0.1,B:0.1):0.2,(C:0.1,(D:0.1,E:0.1):0.1):0.2);",
        "(((A,B),C),(D,E));", 
        "((3,4)2,(5,6)1);",
        "(A:0.1,B:0.2,C:0.3);"
    ]
    
    try:
        from utils.tree_utils import newick_to_nx
        import networkx as nx
        
        for i, nwk in enumerate(test_newicks):
            print(f"\nTest {i+1}: {nwk}")
            try:
                tree_nx = newick_to_nx(nwk)
                print(f"  ✅ Parsed successfully")
                print(f"  Nodes: {list(tree_nx.nodes())}")
                print(f"  Edges: {list(tree_nx.edges())}")
                print(f"  Is tree: {nx.is_tree(tree_nx)}")
                print(f"  Is directed: {nx.is_directed(tree_nx)}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                assert False, f"Newick parsing failed for: {nwk}"
                
        print(f"\n🎉 All {len(test_newicks)} newick parsing tests passed!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False, f"Failed to import required modules: {e}"

if __name__ == "__main__":
    success = test_newick_parsing()
    sys.exit(0 if success else 1)