"""
Graph Builder — adapted from colleague's approach.
Uses Counter-based co-occurrence counting + community accumulation.
"""
from __future__ import annotations

import itertools
import collections
from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# Try to import cdlib for Leiden
try:
    from cdlib import algorithms
    HAS_CDLIB = True
except ImportError:
    HAS_CDLIB = False


def build_rule_graph(
    df: pd.DataFrame,
    node_col: str,
    edge_col: str,
    analysis_col: Optional[str] = None,
    analysis_value: Any = None,
) -> nx.Graph:
    """
    Build a weighted graph for a single rule + (analysis, period) combo.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe (or full period).
    node_col : str, default 'creator_user_id'
        Column to use as nodes.
    edge_col : str
        Column to group by for edge creation.
    analysis_col : str, optional
        The analysis period column (for filtering if needed, not used here).
    analysis_value : any, optional
        The period value.

    Returns
    -------
    nx.Graph
        Weighted graph where weight = count of co-occurrences in groups.
    """
    G = nx.Graph()
    unique_nodes = df[node_col].unique()
    G.add_nodes_from(unique_nodes)

    if edge_col not in df.columns or node_col not in df.columns:
        return G

    grupos = df.groupby(edge_col, sort=False)

    # Collect all pair combinations (brute force, Counter for weights)
    raw_edges: List[Tuple] = []

    for nome_grupo, dados_grupo in grupos:
        nodes_in_group = dados_grupo[node_col].unique()
        if len(nodes_in_group) > 1:
            new_edges = [
                tuple(sorted(pair))
                for pair in itertools.combinations(nodes_in_group, 2)
            ]
            raw_edges.extend(new_edges)

    weight_counts = collections.Counter(raw_edges)

    weighted_edges = [(u, v, w) for (u, v), w in weight_counts.items()]
    G.add_weighted_edges_from(weighted_edges)

    return G


def detect_communities_leiden(
    G: nx.Graph,
) -> Tuple[List[List[Any]], Optional[Any]]:
    """Run Leiden on a graph, return communities and cdlib object."""
    if len(G.nodes) <= 1 or len(G.edges) == 0:
        return [[node] for node in G.nodes], None

    if not HAS_CDLIB:
        # Fallback: use networkx community detection
        try:
            from networkx.algorithms.community import louvain_communities
            communities_raw = louvain_communities(G, weight='weight')
            communities = [list(c) for c in communities_raw]
            return communities, None
        except Exception:
            return [[node] for node in G.nodes], None

    G_int = nx.convert_node_labels_to_integers(G, label_attribute='nome_original')
    try:
        comunidades_cdlib = algorithms.leiden(G_int, weights='weight')
    except Exception:
        try:
            from networkx.algorithms.community import louvain_communities
            communities_raw = louvain_communities(G, weight='weight')
            communities = [list(c) for c in communities_raw]
            # Need to map back from integer labels
            nome_map = nx.get_node_attributes(G_int, 'nome_original')
            communities_str = []
            for c in communities:
                mapped = [nome_map.get(n, n) for n in c]
                communities_str.append(mapped)
            return communities_str, None
        except Exception:
            return [[node] for node in G.nodes], None

    nome_map = nx.get_node_attributes(G_int, 'nome_original')
    communities = []
    for comunidade in comunidades_cdlib.communities:
        comunidade_str = [nome_map[node_id] for node_id in comunidade]
        communities.append(comunidade_str)

    return communities, comunidades_cdlib


def accumulate_communities_to_master(
    master_graph: nx.Graph,
    communities: List[List[Any]],
) -> nx.Graph:
    """
    For each community, increment weight in master_graph for all pairs.
    """
    for members in communities:
        pairs = list(itertools.combinations(members, 2))
        for u, v in pairs:
            if master_graph.has_edge(u, v):
                master_graph[u][v]['weight'] += 1
            else:
                master_graph.add_edge(u, v, weight=1)
    return master_graph


def compute_layout_weights(G: nx.Graph, partition: Dict) -> nx.Graph:
    """Adjust layout weights: same community = attract, different = repel."""
    layout_weights = {}
    for u, v, data in G.edges(data=True):
        peso_original = data.get('weight', 1)
        pu = partition.get(u)
        pv = partition.get(v)
        if pu is not None and pv is not None and pu == pv:
            layout_weights[(u, v)] = peso_original * 1.1
        else:
            layout_weights[(u, v)] = peso_original * 0.9
    nx.set_edge_attributes(G, layout_weights, 'layout_weight')
    return G


def build_rules(
    nodes: List[str] = None,
    edges: List[str] = None,
    analysis: List[str] = None,
) -> List[Dict[str, Any]]:
    """Generate all rule combinations."""
    if nodes is None:
        nodes = ['creator_user_id']
    if edges is None:
        edges = [
            'erp_book_account_name', 'profit_center_name', 'cost_center',
            'journal_entry_origin', 'exchange_id', 'movement_description',
            'faixa_movement', 'id', 'supplier_number', 'asset_number',
            'auxiliary_id', 'journal_entry_id', 'status_origin', 'partner_name',
            'erp_book_account_type', 'movement_type', 'entry_credit_or_debit',
            'erp_transaction_code', 'sap_journal_entry_id',
            'creation_day_of_month', 'creation_day_shift', 'creator_user_id',
        ]
    if analysis is None:
        analysis = [
            'full_period', 'creation_month', 'entry_month_number',
            'creation_day_of_month', 'entry_day_of_month',
            'entry_week', 'creation_week', 'creation_hour', 'creation_day_shift',
        ]

    rules_combinations = list(itertools.product(nodes, edges))
    rules = [
        {'node': node, 'connect_with': edge, 'analysis': analysis}
        for node, edge in rules_combinations
    ]
    return rules


def process_all_rules(
    df: pd.DataFrame,
    rules: List[Dict[str, Any]],
    master_graph: Optional[nx.Graph] = None,
    verbose: bool = False,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Process all rules: for each rule, for each analysis/period,
    build graph -> detect communities -> accumulate into master.

    Returns
    -------
    master_graph : nx.Graph
    metadata : Dict with rule results per period
    """
    if master_graph is None:
        master_graph = nx.Graph()

    metadata = {}

    for idx, rule in enumerate(rules):
        node_col = rule['node']
        edge_col = rule['connect_with']
        analysis_list = rule['analysis']

        if verbose:
            print(f"Rule {idx+1}/{len(rules)}: node={node_col}, edge={edge_col}")

        rule_meta = {'edge_col': edge_col, 'periods': {}}

        for analysis in analysis_list:
            if edge_col == analysis or node_col == edge_col:
                continue

            if analysis == 'full_period':
                G_rule = build_rule_graph(df, node_col, edge_col, analysis)
                communities, _ = detect_communities_leiden(G_rule)
                n_comms = len(communities)
                accumulate_communities_to_master(master_graph, communities)
                rule_meta['periods'][f'{analysis}__all'] = {
                    'n_communities': n_comms,
                    'n_edges': G_rule.number_of_edges(),
                    'n_nodes': G_rule.number_of_nodes(),
                }
            else:
                if analysis not in df.columns:
                    continue
                grouped = df.groupby(analysis, sort=False)
                for period, period_df in grouped:
                    G_rule = build_rule_graph(period_df, node_col, edge_col, analysis, period)
                    if G_rule.number_of_nodes() <= 1:
                        continue
                    communities, _ = detect_communities_leiden(G_rule)
                    n_comms = len(communities)
                    accumulate_communities_to_master(master_graph, communities)
                    period_key = f"{analysis}__{str(period)}"
                    rule_meta['periods'][period_key] = {
                        'n_communities': n_comms,
                        'n_edges': G_rule.number_of_edges(),
                        'n_nodes': G_rule.number_of_nodes(),
                    }

        metadata[f"rule_{idx}"] = rule_meta

    return master_graph, metadata


def convert_graph_to_json(G: nx.Graph, communities: List[List], pos: Dict) -> Dict:
    """Convert NetworkX graph to JSON format for frontend."""
    graph_data = {"nodes": [], "edges": []}

    # Build community lookup
    node_community = {}
    for idx, members in enumerate(communities):
        for m in members:
            node_community[m] = idx

    for node in G.nodes():
        x, y = pos.get(node, (0.0, 0.0))
        graph_data["nodes"].append({
            "id": str(node),
            "x": float(x),
            "y": float(y),
            "community": int(node_community.get(node, -1)),
        })

    for u, v, data in G.edges(data=True):
        graph_data["edges"].append({
            "source": str(u),
            "target": str(v),
            "weight": float(data.get('weight', 1.0)),
        })

    return graph_data