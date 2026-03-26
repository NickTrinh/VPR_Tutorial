"""
Wrapper around Vysotska's sequence matching algorithm.

Adapted from https://github.com/ovysotska/image_sequence_matcher
Uses their graph-based shortest path through the similarity matrix.

Usage:
    from experiments.vysotska_sequence_matcher import sequence_match
    matches_real, matches_hidden = sequence_match(sim_matrix, threshold, fanout=3)
"""

import sys
import os
import math
import numpy as np
from collections import deque


def _pair2index(i, j, cols):
    return i * cols + j


def _index2pair(index, cols):
    i = index // cols
    j = index - i * cols
    return (i, j)


def build_graph(cost_matrix, fanout):
    """Build DAG from cost matrix with given fanout.

    Each cell (i,j) is a node. Edges go from row i to row i+1
    within a fanout window. Source node (-1) connects to all of row 0.
    Target node connects from all of last row.
    """
    rows, cols = cost_matrix.shape
    graph = {}

    # Source node -> first row
    source_id = -1
    graph[source_id] = []
    for j in range(cols):
        idx = _pair2index(0, j, cols)
        graph[source_id].append((idx, cost_matrix[0, j]))

    # Interior edges: row i -> row i+1
    for i in range(rows - 1):
        for j in range(cols):
            parent_id = _pair2index(i, j, cols)
            child_row = i + 1
            f_begin = max(0, j - fanout)
            f_end = min(cols, j + fanout)
            for k in range(f_begin, f_end):
                child_id = _pair2index(child_row, k, cols)
                if parent_id not in graph:
                    graph[parent_id] = []
                graph[parent_id].append((child_id, cost_matrix[child_row, k]))

    # Last row -> target node
    target_id = _pair2index(rows, 0, cols)
    for j in range(cols):
        parent_id = _pair2index(rows - 1, j, cols)
        if parent_id not in graph:
            graph[parent_id] = []
        graph[parent_id].append((target_id, 0))
    graph[target_id] = []

    # Topological order: source, then row-by-row left-to-right, then target
    nodes_sort = deque(range(-1, rows * cols + 1))

    return graph, nodes_sort, cols, target_id


def shortest_path(graph, nodes_sort):
    """Compute shortest path through topologically sorted DAG."""
    dist = {}
    parent = {}
    for node_id in graph:
        dist[node_id] = math.inf
        parent[node_id] = None

    dist[nodes_sort[0]] = 0

    for u in nodes_sort:
        if u not in graph:
            continue
        for child_id, weight in graph[u]:
            if dist[child_id] > dist[u] + weight:
                dist[child_id] = dist[u] + weight
                parent[child_id] = u

    # Retrieve path from target back to source
    path = []
    node = nodes_sort[-1]
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()
    return path


def sequence_match(sim_matrix, non_matching_cost, fanout=3):
    """
    Run Vysotska's sequence matching on a similarity matrix.

    Args:
        sim_matrix: (n_query, n_ref) similarity matrix (higher = better match)
        non_matching_cost: threshold for separating real from hidden matches.
                          Applied to the COST matrix (1 - similarity).
                          So non_matching_cost=0.5 means sim < 0.5 is hidden.
        fanout: number of reference frames to consider per step (default 3)

    Returns:
        matches: dict mapping query_idx -> ref_idx for REAL matches only
        all_path: list of (query_idx, ref_idx) for the full path
        path_real: list of (query_idx, ref_idx) for real matches
        path_hidden: list of (query_idx, ref_idx) for hidden matches
    """
    # Convert similarity to cost (lower = better)
    cost_matrix = 1.0 - sim_matrix

    rows, cols = cost_matrix.shape
    graph, nodes_sort, n_cols, target_id = build_graph(cost_matrix, fanout)
    path = shortest_path(graph, nodes_sort)

    # Convert path node IDs to (row, col) coordinates
    all_path = []
    for node_id in path:
        if node_id < 0:  # source
            continue
        if node_id >= rows * cols:  # target
            continue
        r, c = _index2pair(node_id, n_cols)
        if r < rows and c < cols:
            all_path.append((r, c))

    # Split into real and hidden based on threshold
    path_real = []
    path_hidden = []
    for (r, c) in all_path:
        if cost_matrix[r, c] < non_matching_cost:
            path_real.append((r, c))
        else:
            path_hidden.append((r, c))

    # Build match dict (query -> ref) for real matches only
    matches = {}
    for (q, r) in path_real:
        matches[q] = r

    return matches, all_path, path_real, path_hidden


def evaluate_sequence_match(sim_matrix, matches, tolerance=0):
    """
    Evaluate sequence matching results against 1:1 ground truth.

    Args:
        sim_matrix: (N, N) similarity matrix
        matches: dict query_idx -> ref_idx (real matches only)
        tolerance: allow ±tolerance frame offset

    Returns:
        dict with precision, recall, F1
    """
    n = sim_matrix.shape[0]
    TP = 0
    FP = 0
    FN = 0

    for q in range(n):
        gt = q  # 1:1 correspondence
        if q in matches:
            pred = matches[q]
            if abs(pred - gt) <= tolerance:
                TP += 1
            else:
                FP += 1
        else:
            # Query not matched (hidden) — false negative
            FN += 1

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP, "FP": FP, "FN": FN,
        "n_matched": len(matches),
        "n_total": n,
    }
