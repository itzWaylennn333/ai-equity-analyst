"""Unit checks for the RAPTOR tree index (P6). Pure logic -- no Ollama needed.

Run:  python tests/test_raptor.py    (also pytest-compatible)
The full embed-cluster-summarize build is an integration step verified manually (slow:
~29 LLM summaries on the PYPL 10-K); here we test the deterministic clustering + tree plumbing.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import raptor  # noqa: E402


def test_cluster_shapes():
    vecs = [[1.0, 0.0, 0.0]] * 6 + [[0.0, 1.0, 0.0]] * 6   # two separated groups, n=12 > cluster_size
    labels = raptor._cluster(vecs, 6)
    assert len(labels) == 12 and len(set(labels)) >= 2
    assert raptor._cluster([[1.0, 0.0, 0.0]] * 4, 6) == [0, 0, 0, 0]   # n <= cluster_size -> one cluster


def test_leaf_chunk_ids_recurses():
    nodes = {
        "l1": {"node_id": "l1", "is_summary": False, "child_ids": []},
        "l2": {"node_id": "l2", "is_summary": False, "child_ids": []},
        "l3": {"node_id": "l3", "is_summary": False, "child_ids": []},
        "s1": {"node_id": "s1", "is_summary": True, "child_ids": ["l1", "l2"]},
        "s2": {"node_id": "s2", "is_summary": True, "child_ids": ["s1", "l3"]},   # summary of a summary
    }
    assert sorted(raptor.leaf_chunk_ids(nodes, "s1")) == ["l1", "l2"]
    assert sorted(raptor.leaf_chunk_ids(nodes, "s2")) == ["l1", "l2", "l3"]   # recurse through s1
    assert raptor.leaf_chunk_ids(nodes, "l1") == ["l1"]                       # leaf -> itself


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}: {repr(e)[:300]}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
