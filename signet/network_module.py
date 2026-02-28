from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

NodeId = Hashable
PartitionLike = Union[Mapping[NodeId, int], Sequence[int]]

# =============================================================================
# Backend detection
# =============================================================================

def is_nx(graph: Any) -> bool:
    return isinstance(graph, nx.Graph)


def is_ig(graph: Any) -> bool:
    return isinstance(graph, ig.Graph)


def _raise_unsupported_graph_type(graph: Any) -> None:
    raise TypeError(f"Unsupported graph type: {type(graph)}")

# =============================================================================
# Generic graph access helpers
# =============================================================================

def iter_nodes(graph: Union[nx.Graph, ig.Graph]) -> Iterator[Tuple[NodeId, Dict[str, Any]]]:
    """Yield (node_id, attribute_dict) pairs for both NetworkX and igraph."""
    if is_nx(graph):
        for node, data in graph.nodes(data=True):
            yield node, dict(data)
        return

    if is_ig(graph):
        for vertex in graph.vs:
            yield vertex.index, dict(vertex.attributes())
        return

    _raise_unsupported_graph_type(graph)


def iter_edges(graph: Union[nx.Graph, ig.Graph]) -> Iterator[Tuple[NodeId, NodeId, Dict[str, Any]]]:
    """Yield (u, v, attribute_dict) triples for both NetworkX and igraph."""
    if is_nx(graph):
        for u, v, data in graph.edges(data=True):
            yield u, v, dict(data)
        return

    if is_ig(graph):
        for edge in graph.es:
            yield edge.source, edge.target, dict(edge.attributes())
        return

    _raise_unsupported_graph_type(graph)


def neighbors_with_attrs(graph: Union[nx.Graph, ig.Graph], node: NodeId) -> Iterator[Tuple[NodeId, Dict[str, Any]]]:
    """Yield (neighbor, edge_attr_dict) pairs for a node."""
    if is_nx(graph):
        for neighbor, data in graph[node].items():
            yield neighbor, dict(data)
        return

    if is_ig(graph):
        for edge_id in graph.incident(node):
            edge = graph.es[edge_id]
            neighbor = edge.target if edge.source == node else edge.source
            yield neighbor, dict(edge.attributes())
        return

    _raise_unsupported_graph_type(graph)


def iter_neighbors(graph: Union[nx.Graph, ig.Graph], node: NodeId) -> Iterator[NodeId]:
    """Yield neighbor node IDs."""
    for neighbor, _ in neighbors_with_attrs(graph, node):
        yield neighbor


def subgraph_by_nodes(graph: Union[nx.Graph, ig.Graph], nodes: Iterable[NodeId]) -> Union[nx.Graph, ig.Graph]:
    """Return an induced subgraph containing the given nodes."""
    node_list = list(nodes)

    if is_nx(graph):
        return graph.subgraph(node_list)

    if is_ig(graph):
        return graph.induced_subgraph(node_list)

    _raise_unsupported_graph_type(graph)


def get_node_attr(
    graph: Union[nx.Graph, ig.Graph],
    node: NodeId,
    key: str,
    default: Any = None,
) -> Any:
    """Get a node attribute from either backend."""
    if is_nx(graph):
        return graph.nodes[node].get(key, default)

    if is_ig(graph):
        return graph.vs[node][key] if key in graph.vs.attributes() else default

    _raise_unsupported_graph_type(graph)


def set_node_attr(graph: Union[nx.Graph, ig.Graph], node: NodeId, key: str, value: Any) -> None:
    """Set a node attribute for either backend."""
    if is_nx(graph):
        graph.nodes[node][key] = value
        return

    if is_ig(graph):
        graph.vs[node][key] = value
        return

    _raise_unsupported_graph_type(graph)


def number_of_edges(graph: Union[nx.Graph, ig.Graph]) -> int:
    if is_nx(graph):
        return graph.number_of_edges()

    if is_ig(graph):
        return graph.ecount()

    _raise_unsupported_graph_type(graph)


def find_node_by_attr(
    graph: Union[nx.Graph, ig.Graph],
    key: str,
    value: Any,
) -> Optional[NodeId]:
    """Return the first node whose attribute matches value."""
    for node, data in iter_nodes(graph):
        if data.get(key) == value:
            return node
    return None


def build_attr_to_node_map(
    graph: Union[nx.Graph, ig.Graph],
    key: str,
) -> Dict[Any, NodeId]:
    """Build a reverse map from node attribute value to node ID."""
    mapping: Dict[Any, NodeId] = {}
    for node, data in iter_nodes(graph):
        if key in data:
            mapping[data[key]] = node
    return mapping


def normalize_partition(partition):
    if isinstance(partition, dict):
        return partition

    if hasattr(partition, "membership"):
        return {i: comm for i, comm in enumerate(partition.membership)}

    try:
        return {i: comm for i, comm in enumerate(partition)}
    except TypeError as exc:
        raise TypeError(f"Unsupported partition type: {type(partition)}") from exc


def to_networkx_graph(graph: Union[nx.Graph, ig.Graph]) -> nx.Graph:
    """
    Convert a graph to a NetworkX graph for unified drawing.

    If input is already NetworkX, return a copy.
    """
    if is_nx(graph):
        return graph.copy()

    if is_ig(graph):
        nx_graph = nx.Graph()

        for vertex in graph.vs:
            nx_graph.add_node(vertex.index, **dict(vertex.attributes()))

        for edge in graph.es:
            nx_graph.add_edge(edge.source, edge.target, **dict(edge.attributes()))

        return nx_graph

    _raise_unsupported_graph_type(graph)

# =============================================================================
# File loaders
# =============================================================================

def load_graph_from_TSV(filename: str, threshold: float) -> nx.Graph:
    """
    Load a weighted undirected graph from a TSV file into NetworkX.

    Expected columns:
    gene_id1, gene_id2, gene_name1, gene_name2, weight
    """
    graph = nx.Graph()

    with open(filename, "r") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            gene_id1, gene_id2, gene_name1, gene_name2, weight_str = parts[:5]
            weight = float(weight_str)

            if weight < threshold:
                continue

            graph.add_node(gene_id1, name=gene_name1)
            graph.add_node(gene_id2, name=gene_name2)
            graph.add_edge(gene_id1, gene_id2, weight=weight)

    return graph

def load_graph_from_TSV_igraph(filename: str, threshold: float) -> ig.Graph:
    """
    Load a weighted undirected graph from a TSV file into igraph.

    Vertex attributes:
    - name
    - gene_id

    Edge attributes:
    - weight
    """
    edges: List[Tuple[str, str]] = []
    weights: List[float] = []
    gene_ids: set[str] = set()
    gene_name_map: Dict[str, str] = {}

    with open(filename, "r") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            gene_id1, gene_id2, gene_name1, gene_name2, weight_str = parts[:5]
            weight = float(weight_str)

            if weight < threshold:
                continue

            edges.append((gene_id1, gene_id2))
            weights.append(weight)

            gene_ids.add(gene_id1)
            gene_ids.add(gene_id2)
            gene_name_map[gene_id1] = gene_name1
            gene_name_map[gene_id2] = gene_name2

    sorted_gene_ids = sorted(gene_ids)
    id_to_index = {gene_id: idx for idx, gene_id in enumerate(sorted_gene_ids)}
    edge_indices = [(id_to_index[u], id_to_index[v]) for u, v in edges]

    graph = ig.Graph(n=len(sorted_gene_ids), edges=edge_indices)
    graph.vs["name"] = [gene_name_map[gene_id] for gene_id in sorted_gene_ids]
    graph.vs["gene_id"] = sorted_gene_ids
    graph.es["weight"] = weights

    return graph

# =============================================================================
# Community / weight utilities
# =============================================================================

def get_subgraph_by_community(
    graph: Union[nx.Graph, ig.Graph],
    communities: PartitionLike,
    community_id: int,
) -> Union[nx.Graph, ig.Graph]:
    """Return the induced subgraph of nodes belonging to a given community."""
    partition = normalize_partition(communities)
    nodes_in_community = [node for node, comm_id in partition.items() if comm_id == community_id]
    return subgraph_by_nodes(graph, nodes_in_community)

def check_graph_weights(graph: Union[nx.Graph, ig.Graph]) -> str:
    edge_data = list(iter_edges(graph))
    if not edge_data:
        return "no edges"

    has_weight_flags = ["weight" in data for _, _, data in edge_data]
    weights = [data.get("weight", 1) for _, _, data in edge_data]

    if all(has_weight_flags):
        return "unweighted" if len(set(weights)) == 1 else "weighted"

    return "partially weighted"

def display_communities_by_name(
    graph: Union[nx.Graph, ig.Graph],
    partition: PartitionLike,
    name_attr: str = "name",
) -> None:
    normalized = normalize_partition(partition)
    community_to_names: Dict[int, List[str]] = {}

    for node, community_id in normalized.items():
        gene_name = get_node_attr(graph, node, name_attr, str(node))
        community_to_names.setdefault(community_id, []).append(gene_name)

    for community_id in sorted(community_to_names):
        unique_names = sorted(set(community_to_names[community_id]))
        print(f'Community {community_id}: {" | ".join(unique_names)}')


def find_communities_of_genes(
    graph: Union[nx.Graph, ig.Graph],
    partition: PartitionLike,
    gene_names: Sequence[str],
    name_attr: str = "name",
) -> Dict[str, Union[int, str]]:
    normalized = normalize_partition(partition)
    name_to_node = build_attr_to_node_map(graph, name_attr)

    results: Dict[str, Union[int, str]] = {}
    for gene_name in gene_names:
        node_id = name_to_node.get(gene_name)
        if node_id is None:
            results[gene_name] = "Gene not found in the graph."
            continue

        community_id = normalized.get(node_id)
        if community_id is None:
            results[gene_name] = "Community for gene not found."
        else:
            results[gene_name] = community_id

    return results

def count_nodes_in_communities(partition: PartitionLike) -> None:
    normalized = normalize_partition(partition)
    community_counts = Counter(normalized.values())

    for community_id, count in sorted(community_counts.items()):
        print(f"Community {community_id}: {count} nodes")

# Backward-compatible aliases
find_communities_of_genes_nx = find_communities_of_genes
find_communities_of_genes_igraph = find_communities_of_genes


def _top_degree_nodes(subgraph, top_n):
    """
    Return node IDs of top-N nodes by degree within the given subgraph.
    Supports both NetworkX and igraph.
    """
    if top_n is None:
        if is_nx(subgraph):
            return list(subgraph.nodes())
        if is_ig(subgraph):
            return list(range(subgraph.vcount()))

    if is_nx(subgraph):
        degree_items = list(subgraph.degree())
        degree_items.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in degree_items[:top_n]]

    if is_ig(subgraph):
        degrees = subgraph.degree()
        ranked = sorted(enumerate(degrees), key=lambda x: x[1], reverse=True)
        return [node for node, _ in ranked[:top_n]]

    _raise_unsupported_graph_type(subgraph)


# =============================================================================
# Drawing helpers
# =============================================================================

def _draw_simple_graph(
    graph: Union[nx.Graph, ig.Graph],
    highlight_nodes: Optional[Iterable[NodeId]] = None,
    title: Optional[str] = None,
    name_attr: str = "name",
    layout_seed: int = 10,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Draw a graph after converting to NetworkX."""
    nx_graph = to_networkx_graph(graph)
    highlight_set = set(highlight_nodes or [])

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(nx_graph, seed=layout_seed)

    labels = {node: nx_graph.nodes[node].get(name_attr, str(node)) for node in nx_graph.nodes()}
    node_colors = ["red" if node in highlight_set else "lightblue" for node in nx_graph.nodes()]

    nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(nx_graph, pos, edge_color="gray", alpha=0.7)
    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=12)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.show()


def build_attr_to_node_map(graph, key):
    """Build a reverse map: node attribute value -> node ID."""
    mapping = {}
    for node, data in iter_nodes(graph):
        if key in data:
            mapping[data[key]] = node
    return mapping


def _draw_signed_subgraphs(
    pos_subgraph,
    neg_subgraph,
    highlight_keys=None,
    title=None,
    name_attr="name",
    match_attr="gene_id",
    layout="spring",
    layout_seed=10,
    figsize=(10, 8),
    node_size=500,
    font_size=10,
    label_top_n=None,
):
    pos_nx = to_networkx_graph(pos_subgraph)
    neg_nx = to_networkx_graph(neg_subgraph)

    highlight_keys = set(highlight_keys or [])

    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(pos_nx.nodes(data=True))
    combined_graph.add_nodes_from(neg_nx.nodes(data=True))
    combined_graph.add_edges_from(pos_nx.edges())
    combined_graph.add_edges_from(neg_nx.edges())

    # layout
    if layout == "spring":
        pos = nx.spring_layout(
            combined_graph,
            seed=layout_seed,
            k=2.0 / np.sqrt(max(combined_graph.number_of_nodes(), 2)),
            iterations=300,
        )
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(combined_graph)
    else:
        pos = nx.spring_layout(combined_graph, seed=layout_seed)

    plt.figure(figsize=figsize)

    # node colors
    node_colors = []
    for node in combined_graph.nodes():
        attrs = combined_graph.nodes[node]
        key = attrs.get(match_attr, attrs.get(name_attr, node))
        node_colors.append("red" if key in highlight_keys else "lightblue")

    # labels: optionally only top-degree nodes
    if label_top_n is None:
        label_nodes = set(combined_graph.nodes())
    else:
        degree_rank = sorted(combined_graph.degree(), key=lambda x: x[1], reverse=True)
        label_nodes = {n for n, _ in degree_rank[:label_top_n]}

    labels = {
        node: combined_graph.nodes[node].get(name_attr, str(node))
        for node in label_nodes
    }

    nx.draw_networkx_nodes(
        combined_graph,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        combined_graph,
        pos,
        edgelist=list(pos_nx.edges()),
        edge_color="red",
        width=1.5,
        alpha=0.35,
    )
    nx.draw_networkx_edges(
        combined_graph,
        pos,
        edgelist=list(neg_nx.edges()),
        edge_color="blue",
        width=1.5,
        alpha=0.35,
    )
    nx.draw_networkx_labels(
        combined_graph,
        pos,
        labels=labels,
        font_size=font_size,
    )

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Positive edge"),
        Line2D([0], [0], color="blue", lw=2, label="Negative edge"),
    ]
    plt.legend(handles=legend_elements)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

# =============================================================================
# Visualization utilities
# =============================================================================

def visualize_top_weighted_nodes(
    graph: Union[nx.Graph, ig.Graph],
    gene_name: str,
    top_n: int,
    name_attr: str = "name",
) -> None:
    """
    Visualize the target gene and its top-N weighted neighbors.
    """
    gene_id = find_node_by_attr(graph, name_attr, gene_name)
    if gene_id is None:
        print(f"No gene named {gene_name} found in the graph.")
        return

    edges = [
        (gene_id, neighbor, data.get("weight", 1))
        for neighbor, data in neighbors_with_attrs(graph, gene_id)
    ]
    top_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]

    top_nodes = {neighbor for _, neighbor, _ in top_edges}
    top_nodes.add(gene_id)

    subgraph = subgraph_by_nodes(graph, top_nodes)
    _draw_simple_graph(
        subgraph,
        highlight_nodes={gene_id},
        title=f"Top {top_n} genes connections to {gene_name}",
        name_attr=name_attr,
    )

def visualize_top_weighted_nodes_signed(
    pos_graph,
    neg_graph,
    gene_name,
    top_n,
    name_attr="name",
    match_attr="gene_id",
):
    """
    Visualize the target gene and its top-N weighted neighbors.

    Top-N calculation is based only on the positive graph.
    Positive edges are shown in red and negative edges in blue.

    This function safely supports NetworkX / igraph mixed use by matching
    nodes across graphs via match_attr (default: gene_id), not raw node IDs.
    """
    gene_id_pos = find_node_by_attr(pos_graph, name_attr, gene_name)
    if gene_id_pos is None:
        print(f"No gene named {gene_name} found in the positive graph.")
        return

    edges = [
        (gene_id_pos, neighbor, data.get("weight", 1))
        for neighbor, data in neighbors_with_attrs(pos_graph, gene_id_pos)
    ]
    top_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]

    top_nodes_pos = {neighbor for _, neighbor, _ in top_edges}
    top_nodes_pos.add(gene_id_pos)

    pos_subgraph = subgraph_by_nodes(pos_graph, top_nodes_pos)

    # Match positive-side selected nodes to negative graph via attribute
    pos_match_keys = {
        get_node_attr(
            pos_graph,
            node,
            match_attr,
            get_node_attr(pos_graph, node, name_attr, node),
        )
        for node in top_nodes_pos
    }

    neg_match_map = build_attr_to_node_map(neg_graph, match_attr)
    if not neg_match_map and match_attr != name_attr:
        neg_match_map = build_attr_to_node_map(neg_graph, name_attr)

    neg_nodes = [neg_match_map[key] for key in pos_match_keys if key in neg_match_map]
    neg_subgraph = subgraph_by_nodes(neg_graph, neg_nodes)

    highlight_key = get_node_attr(
        pos_graph,
        gene_id_pos,
        match_attr,
        get_node_attr(pos_graph, gene_id_pos, name_attr, gene_name),
    )

    _draw_signed_subgraphs(
        pos_subgraph,
        neg_subgraph,
        highlight_keys={highlight_key},
        title=f"Top {top_n} genes connections to {gene_name}",
        name_attr=name_attr,
        match_attr=match_attr,
    )


def visualize_top_weighted_nodes_between_genes(
    graph: Union[nx.Graph, ig.Graph],
    gene_names: Sequence[str],
    top_n: int,
    name_attr: str = "name",
) -> None:
    """
    Visualize top-N nodes ranked by summed connection weights from multiple genes.
    """
    name_to_node = build_attr_to_node_map(graph, name_attr)
    gene_ids = [name_to_node[name] for name in gene_names if name in name_to_node]

    if len(gene_ids) != len(gene_names):
        print("Some genes not found in the graph.")
        return

    node_weights: Dict[NodeId, float] = {}
    for gene_id in gene_ids:
        for neighbor, edge_data in neighbors_with_attrs(graph, gene_id):
            node_weights[neighbor] = node_weights.get(neighbor, 0.0) + edge_data.get("weight", 1)

    top_nodes = sorted(node_weights, key=node_weights.get, reverse=True)[:top_n]
    subgraph_nodes = set(top_nodes).union(gene_ids)
    subgraph = subgraph_by_nodes(graph, subgraph_nodes)

    _draw_simple_graph(
        subgraph,
        highlight_nodes=set(gene_ids),
        title=f"Top {top_n} Weighted Connections for {', '.join(gene_names)}",
        name_attr=name_attr,
    )


def visualize_top_weighted_nodes_between_genes_signed(
    pos_graph,
    neg_graph,
    gene_names,
    top_n,
    name_attr="name",
    match_attr="gene_id",
):
    """
    Visualize top-N nodes ranked by summed positive-edge weights
    from multiple genes.

    Top-N calculation is based only on the positive graph.
    Positive edges are shown in red and negative edges in blue.

    This function safely supports NetworkX / igraph mixed use by matching
    nodes across graphs via match_attr (default: gene_id), not raw node IDs.
    """
    name_to_node_pos = build_attr_to_node_map(pos_graph, name_attr)
    gene_ids_pos = [name_to_node_pos[name] for name in gene_names if name in name_to_node_pos]

    if len(gene_ids_pos) != len(gene_names):
        print("Some genes not found in the positive graph.")
        return

    node_weights = {}
    for gene_id in gene_ids_pos:
        for neighbor, edge_data in neighbors_with_attrs(pos_graph, gene_id):
            node_weights[neighbor] = node_weights.get(neighbor, 0.0) + edge_data.get("weight", 1)

    top_nodes_pos = sorted(node_weights, key=node_weights.get, reverse=True)[:top_n]
    subgraph_nodes_pos = set(top_nodes_pos).union(gene_ids_pos)

    pos_subgraph = subgraph_by_nodes(pos_graph, subgraph_nodes_pos)

    # Match positive-side selected nodes to negative graph via attribute
    pos_match_keys = {
        get_node_attr(
            pos_graph,
            node,
            match_attr,
            get_node_attr(pos_graph, node, name_attr, node),
        )
        for node in subgraph_nodes_pos
    }

    neg_match_map = build_attr_to_node_map(neg_graph, match_attr)
    if not neg_match_map and match_attr != name_attr:
        neg_match_map = build_attr_to_node_map(neg_graph, name_attr)

    neg_nodes = [neg_match_map[key] for key in pos_match_keys if key in neg_match_map]
    neg_subgraph = subgraph_by_nodes(neg_graph, neg_nodes)

    highlight_keys = {
        get_node_attr(
            pos_graph,
            node,
            match_attr,
            get_node_attr(pos_graph, node, name_attr, node),
        )
        for node in gene_ids_pos
    }

    _draw_signed_subgraphs(
        pos_subgraph,
        neg_subgraph,
        highlight_keys=highlight_keys,
        title=f"Top {top_n} Weighted Connections for {', '.join(gene_names)}",
        name_attr=name_attr,
        match_attr=match_attr,
    )

def visualize_module(
    graph,
    partition,
    community_id,
    name_attr="name",
):
    """
    Visualize a specific module/community from a graph.

    Parameters
    ----------
    graph : networkx.Graph or igraph.Graph
        Input graph.
    partition : dict, list-like, or Leiden/igraph partition object
        Community assignment.
    community_id : int
        Target community ID to visualize.
    name_attr : str
        Node attribute used for labels.
    """
    normalized = normalize_partition(partition)
    nodes_in_community = [node for node, comm in normalized.items() if comm == community_id]

    if not nodes_in_community:
        print(f"Community {community_id} is empty or not found.")
        return

    subgraph = subgraph_by_nodes(graph, nodes_in_community)

    _draw_simple_graph(
        subgraph,
        highlight_nodes=None,
        title=f"Module {community_id}",
        name_attr=name_attr,
    )

def visualize_module_signed(
    pos_graph,
    neg_graph,
    partition,
    community_id,
    name_attr="name",
    match_attr="gene_id",
):
    """
    Visualize a specific module/community from signed graphs.

    Positive edges are shown in red, negative edges in blue.

    If `partition` is a Leiden/igraph partition object, its vertex indices are
    interpreted on `partition.graph`, then mapped back to pos_graph/neg_graph
    using `match_attr` (default: gene_id).
    """
    normalized = normalize_partition(partition)

    # partition object が持つ元グラフ（aligned graph）上の node IDs
    part_graph = partition.graph if hasattr(partition, "graph") else pos_graph

    # community に属する node IDs を partition 側の graph 上で取得
    part_nodes = [node for node, comm in normalized.items() if comm == community_id]

    if not part_nodes:
        print(f"Community {community_id} is empty or not found.")
        return

    # partition graph 上の node -> match key（通常 gene_id）
    part_match_keys = {
        get_node_attr(
            part_graph,
            node,
            match_attr,
            get_node_attr(part_graph, node, name_attr, node),
        )
        for node in part_nodes
    }

    # pos_graph / neg_graph 側へ写像
    pos_match_map = build_attr_to_node_map(pos_graph, match_attr)
    neg_match_map = build_attr_to_node_map(neg_graph, match_attr)

    if not pos_match_map and match_attr != name_attr:
        pos_match_map = build_attr_to_node_map(pos_graph, name_attr)
    if not neg_match_map and match_attr != name_attr:
        neg_match_map = build_attr_to_node_map(neg_graph, name_attr)

    pos_nodes = [pos_match_map[key] for key in part_match_keys if key in pos_match_map]
    neg_nodes = [neg_match_map[key] for key in part_match_keys if key in neg_match_map]

    if not pos_nodes:
        print(f"Community {community_id} could not be mapped onto the positive graph.")
        return

    pos_subgraph = subgraph_by_nodes(pos_graph, pos_nodes)
    neg_subgraph = subgraph_by_nodes(neg_graph, neg_nodes)

    _draw_signed_subgraphs(
        pos_subgraph,
        neg_subgraph,
        highlight_keys=set(),
        title=f"Module {community_id}",
        layout="kamada_kawai",
        figsize=(14, 12),
        name_attr=name_attr,
        match_attr=match_attr,
    )

def visualize_module_of_gene(
    graph,
    partition,
    gene_name,
    name_attr="name",
):
    """
    Visualize the module/community containing the specified gene.
    """
    normalized = normalize_partition(partition)
    node = find_node_by_attr(graph, name_attr, gene_name)

    if node is None:
        print(f"No gene named {gene_name} found in the graph.")
        return

    if node not in normalized:
        print(f"No community assignment found for gene {gene_name}.")
        return

    community_id = normalized[node]
    visualize_module(graph, normalized, community_id, name_attr=name_attr)

def visualize_module_of_gene_signed(
    pos_graph,
    neg_graph,
    partition,
    gene_name,
    name_attr="name",
    match_attr="gene_id",
):
    """
    Visualize the signed module/community containing the specified gene.

    Works for dict/list partitions and Leiden partition objects.
    """
    normalized = normalize_partition(partition)

    # pos_graph 側で query gene を探す
    pos_node = find_node_by_attr(pos_graph, name_attr, gene_name)
    if pos_node is None:
        print(f"No gene named {gene_name} found in the positive graph.")
        return

    query_key = get_node_attr(
        pos_graph,
        pos_node,
        match_attr,
        get_node_attr(pos_graph, pos_node, name_attr, gene_name),
    )

    # partition 側 graph 上で同じ match_attr を持つ node を探す
    part_graph = partition.graph if hasattr(partition, "graph") else pos_graph
    part_node = find_node_by_attr(part_graph, match_attr, query_key)

    if part_node is None and match_attr != name_attr:
        part_node = find_node_by_attr(part_graph, name_attr, gene_name)

    if part_node is None:
        print(f"No matching node for gene {gene_name} found in the partition graph.")
        return

    if part_node not in normalized:
        print(f"No community assignment found for gene {gene_name}.")
        return

    community_id = normalized[part_node]

    visualize_module_signed(
        pos_graph,
        neg_graph,
        partition,
        community_id,
        name_attr=name_attr,
        match_attr=match_attr,
    )

def visualize_module_top_degree_nodes(
    graph,
    partition,
    community_id,
    top_n=30,
    name_attr="name",
):
    """
    Visualize only the top-N degree nodes within a given module/community.

    Parameters
    ----------
    graph : networkx.Graph or igraph.Graph
        Input graph.
    partition : dict, list-like, or Leiden/igraph partition object
        Community assignment.
    community_id : int
        Target community ID.
    top_n : int
        Number of highest-degree nodes to keep within the module.
    name_attr : str
        Node attribute used for labels.
    """
    normalized = normalize_partition(partition)
    nodes_in_community = [node for node, comm in normalized.items() if comm == community_id]

    if not nodes_in_community:
        print(f"Community {community_id} is empty or not found.")
        return

    module_subgraph = subgraph_by_nodes(graph, nodes_in_community)
    top_nodes = _top_degree_nodes(module_subgraph, top_n)
    top_subgraph = subgraph_by_nodes(module_subgraph, top_nodes)

    _draw_simple_graph(
        top_subgraph,
        highlight_nodes=None,
        title=f"Module {community_id} (top {len(top_nodes)} by degree)",
        name_attr=name_attr,
    )


def visualize_module_top_degree_nodes_signed(
    pos_graph,
    neg_graph,
    partition,
    community_id,
    top_n=30,
    name_attr="name",
    match_attr="gene_id",
):
    """
    Visualize only the top-N degree nodes within a given signed module/community.

    Top-N ranking is computed within the positive graph's module subgraph.
    Positive edges are shown in red and negative edges in blue.

    This function supports Leiden partition objects whose node indices belong
    to partition.graph, not necessarily to pos_graph / neg_graph directly.
    """
    normalized = normalize_partition(partition)
    part_graph = partition.graph if hasattr(partition, "graph") else pos_graph

    # partition 側 graph 上で community のノードを取得
    part_nodes = [node for node, comm in normalized.items() if comm == community_id]

    if not part_nodes:
        print(f"Community {community_id} is empty or not found.")
        return

    # partition graph 上の node -> match key（通常 gene_id）
    part_match_keys = {
        get_node_attr(
            part_graph,
            node,
            match_attr,
            get_node_attr(part_graph, node, name_attr, node),
        )
        for node in part_nodes
    }

    # pos_graph / neg_graph 側へ写像
    pos_match_map = build_attr_to_node_map(pos_graph, match_attr)
    neg_match_map = build_attr_to_node_map(neg_graph, match_attr)

    if not pos_match_map and match_attr != name_attr:
        pos_match_map = build_attr_to_node_map(pos_graph, name_attr)
    if not neg_match_map and match_attr != name_attr:
        neg_match_map = build_attr_to_node_map(neg_graph, name_attr)

    pos_nodes = [pos_match_map[key] for key in part_match_keys if key in pos_match_map]
    neg_nodes = [neg_match_map[key] for key in part_match_keys if key in neg_match_map]

    if not pos_nodes:
        print(f"Community {community_id} could not be mapped onto the positive graph.")
        return

    # positive graph 側で module subgraph を作り、その中で degree 上位 N を選ぶ
    pos_module_subgraph = subgraph_by_nodes(pos_graph, pos_nodes)
    top_nodes_pos = _top_degree_nodes(pos_module_subgraph, top_n)
    pos_top_subgraph = subgraph_by_nodes(pos_module_subgraph, top_nodes_pos)

    # 選ばれた positive 側 top nodes を negative 側へ再マップ
    top_match_keys = {
        get_node_attr(
            pos_graph,
            node,
            match_attr,
            get_node_attr(pos_graph, node, name_attr, node),
        )
        for node in top_nodes_pos
    }

    neg_top_nodes = [neg_match_map[key] for key in top_match_keys if key in neg_match_map]
    neg_top_subgraph = subgraph_by_nodes(neg_graph, neg_top_nodes)

    _draw_signed_subgraphs(
        pos_top_subgraph,
        neg_top_subgraph,
        highlight_keys=set(),
        title=f"Module {community_id} (top {len(top_nodes_pos)} by degree)",
        name_attr=name_attr,
        match_attr=match_attr,
    )


def visualize_module_of_gene_top_degree_nodes(
    graph,
    partition,
    gene_name,
    top_n=30,
    name_attr="name",
):
    """
    Visualize top-N degree nodes within the module containing the given gene.
    """
    normalized = normalize_partition(partition)
    node = find_node_by_attr(graph, name_attr, gene_name)

    if node is None:
        print(f"No gene named {gene_name} found in the graph.")
        return

    if node not in normalized:
        print(f"No community assignment found for gene {gene_name}.")
        return

    community_id = normalized[node]
    visualize_module_top_degree_nodes(
        graph,
        normalized,
        community_id,
        top_n=top_n,
        name_attr=name_attr,
    )

def visualize_module_of_gene_top_degree_nodes_signed(
    pos_graph,
    neg_graph,
    partition,
    gene_name,
    top_n=30,
    name_attr="name",
    match_attr="gene_id",
):
    """
    Visualize top-N degree nodes within the signed module containing the given gene.

    Works for dict/list partitions and Leiden partition objects.
    """
    normalized = normalize_partition(partition)

    # pos_graph 側で query gene を探す
    pos_node = find_node_by_attr(pos_graph, name_attr, gene_name)
    if pos_node is None:
        print(f"No gene named {gene_name} found in the positive graph.")
        return

    query_key = get_node_attr(
        pos_graph,
        pos_node,
        match_attr,
        get_node_attr(pos_graph, pos_node, name_attr, gene_name),
    )

    # partition 側 graph 上で同じ遺伝子を探す
    part_graph = partition.graph if hasattr(partition, "graph") else pos_graph
    part_node = find_node_by_attr(part_graph, match_attr, query_key)

    if part_node is None and match_attr != name_attr:
        part_node = find_node_by_attr(part_graph, name_attr, gene_name)

    if part_node is None:
        print(f"No matching node for gene {gene_name} found in the partition graph.")
        return

    if part_node not in normalized:
        print(f"No community assignment found for gene {gene_name}.")
        return

    community_id = normalized[part_node]

    visualize_module_top_degree_nodes_signed(
        pos_graph,
        neg_graph,
        partition,
        community_id,
        top_n=top_n,
        name_attr=name_attr,
        match_attr=match_attr,
    )

# =============================================================================
# Entropy utilities
# =============================================================================

def calc_entropy(partition: PartitionLike) -> float:
    """
    Compute normalized entropy of community size distribution.

    Returns 0.0 if there is only one community or no nodes.
    """
    normalized = normalize_partition(partition)
    community_sizes = Counter(normalized.values())
    sizes = np.array(list(community_sizes.values()), dtype=float)

    if len(sizes) <= 1:
        return 0.0

    probs = sizes / sizes.sum()
    entropy = -np.sum(probs * np.log(probs))
    normalized_entropy = entropy / np.log(len(sizes))
    return float(normalized_entropy)


def calc_entropy_igraph(partition: ig.clustering.VertexClustering) -> float:
    """
    Compute normalized entropy from an igraph VertexClustering object.
    """
    sizes = np.array(partition.sizes(), dtype=float)

    if len(sizes) <= 1:
        return 0.0

    probs = sizes / sizes.sum()
    entropy = -np.sum(probs * np.log(probs))
    normalized_entropy = entropy / np.log(len(sizes))
    return float(normalized_entropy)

# =============================================================================
# Synthetic signed network generation
# =============================================================================

def generate_signednetwork(
    community_size: int,
    num_communities: int,
    intra_edges: int,
    inter_edges: int,
    p1: float,
    p2: float,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Generate synthetic positive and negative signed graphs.

    Parameters
    ----------
    community_size
        Number of nodes per community.
    num_communities
        Number of communities.
    intra_edges
        Number of positive intra-community edges per community.
    inter_edges
        Number of candidate negative edges between adjacent communities.
    p1
        Fraction of positive intra-community edges to flip to negative.
    p2
        Fraction of negative inter-community edges to flip to positive.
    seed
        Random seed.
    """
    if seed is not None:
        np.random.seed(seed)

    pos_graph = nx.Graph()
    neg_graph = nx.Graph()

    # Positive intra-community graphs
    for community_idx in range(num_communities):
        tmp_graph = nx.gnm_random_graph(community_size, intra_edges)
        tmp_graph = nx.relabel_nodes(
            tmp_graph,
            {node: node + community_idx * community_size for node in tmp_graph.nodes()},
        )
        pos_graph = nx.compose(pos_graph, tmp_graph)

    # Negative inter-community edges between adjacent communities only
    total_nodes = community_size * num_communities
    inter_edge_candidates = [
        (i, j)
        for i in range(total_nodes)
        for j in range(i + 1, total_nodes)
        if abs(i // community_size - j // community_size) == 1
    ]

    selected_indices = np.random.choice(
        len(inter_edge_candidates),
        inter_edges * num_communities,
        replace=False,
    )
    negative_edges = [inter_edge_candidates[i] for i in selected_indices]
    neg_graph.add_edges_from(negative_edges)

    # Ensure the two graphs share the same node set
    pos_graph.add_nodes_from(neg_graph.nodes())
    neg_graph.add_nodes_from(pos_graph.nodes())

    # Flip some positive intra-community edges to negative
    positive_edges = list(pos_graph.edges())
    num_flip_pos_to_neg = int(intra_edges * p1)

    if positive_edges and num_flip_pos_to_neg > 0:
        selected_pos_indices = np.random.choice(len(positive_edges), num_flip_pos_to_neg, replace=False)
        for edge_index in selected_pos_indices:
            u, v = positive_edges[edge_index]
            same_community = (u // community_size) == (v // community_size)
            if same_community and pos_graph.degree(u) > 1 and pos_graph.degree(v) > 1:
                pos_graph.remove_edge(u, v)
                neg_graph.add_edge(u, v)

    # Flip some negative inter-community edges to positive
    negative_edges = list(neg_graph.edges())
    num_flip_neg_to_pos = int(inter_edges * p2)

    if negative_edges and num_flip_neg_to_pos > 0:
        selected_neg_indices = np.random.choice(len(negative_edges), num_flip_neg_to_pos, replace=False)
        for edge_index in selected_neg_indices:
            u, v = negative_edges[edge_index]
            different_community = (u // community_size) != (v // community_size)
            if different_community and neg_graph.degree(u) > 1 and neg_graph.degree(v) > 1:
                neg_graph.remove_edge(u, v)
                pos_graph.add_edge(u, v)

    return pos_graph, neg_graph