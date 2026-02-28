import numpy as np
import igraph as ig
import leidenalg as la

from signet.network_module import is_nx, is_ig, iter_nodes, iter_edges, get_node_attr

def _get_graph_gene_ids(graph, gene_id_attr="gene_id"):
    """
    Return a list of canonical gene IDs for a graph.

    Priority:
    1. node attribute gene_id
    2. node attribute name
    3. node ID itself
    """
    gene_ids = []
    for node, data in iter_nodes(graph):
        gene_id = data.get(gene_id_attr, data.get("name", node))
        gene_ids.append(gene_id)
    return gene_ids


def _build_geneid_to_name_map(graph, gene_id_attr="gene_id", name_attr="name"):
    """
    Build gene_id -> display_name map from either NetworkX or igraph.
    """
    mapping = {}
    for node, data in iter_nodes(graph):
        gene_id = data.get(gene_id_attr, data.get(name_attr, node))
        gene_name = data.get(name_attr, str(gene_id))
        mapping[gene_id] = gene_name
    return mapping


def _to_aligned_igraph(graph, all_gene_ids, gene_id_attr="gene_id", name_attr="name"):
    """
    Convert a NetworkX/igraph graph to an igraph.Graph whose vertex order is
    exactly aligned to all_gene_ids.
    """
    id_to_idx = {gid: i for i, gid in enumerate(all_gene_ids)}

    g = ig.Graph(len(all_gene_ids))
    g.vs["gene_id"] = list(all_gene_ids)
    g.vs["name"] = [str(gid) for gid in all_gene_ids]

    # name を元グラフから可能な範囲で引き継ぐ
    geneid_to_name = _build_geneid_to_name_map(graph, gene_id_attr=gene_id_attr, name_attr=name_attr)
    g.vs["name"] = [geneid_to_name.get(gid, str(gid)) for gid in all_gene_ids]

    edges = []
    weights = []
    has_any_weight = False

    if is_nx(graph):
        for u, v, data in iter_edges(graph):
            gid_u = get_node_attr(graph, u, gene_id_attr, u)
            gid_v = get_node_attr(graph, v, gene_id_attr, v)
            edges.append((id_to_idx[gid_u], id_to_idx[gid_v]))

            if "weight" in data:
                weights.append(data["weight"])
                has_any_weight = True
            else:
                weights.append(1.0)

    elif is_ig(graph):
        for u, v, data in iter_edges(graph):
            gid_u = get_node_attr(graph, u, gene_id_attr, u)
            gid_v = get_node_attr(graph, v, gene_id_attr, v)
            edges.append((id_to_idx[gid_u], id_to_idx[gid_v]))

            if "weight" in data:
                weights.append(data["weight"])
                has_any_weight = True
            else:
                weights.append(1.0)
    else:
        raise TypeError(f"Unsupported graph type: {type(graph)}")

    if edges:
        g.add_edges(edges)

    if edges and has_any_weight:
        g.es["weight"] = weights

    return g


def align_graphs(G_pos, G_neg, gene_id_attr="gene_id", name_attr="name"):
    """
    Align positive and negative graphs onto the same vertex set and order.

    Accepts either NetworkX or igraph as input.
    Returns two aligned igraph.Graph objects.
    """
    ids_pos = set(_get_graph_gene_ids(G_pos, gene_id_attr=gene_id_attr))
    ids_neg = set(_get_graph_gene_ids(G_neg, gene_id_attr=gene_id_attr))
    all_gene_ids = sorted(ids_pos | ids_neg)

    new_G_pos = _to_aligned_igraph(
        G_pos,
        all_gene_ids,
        gene_id_attr=gene_id_attr,
        name_attr=name_attr,
    )
    new_G_neg = _to_aligned_igraph(
        G_neg,
        all_gene_ids,
        gene_id_attr=gene_id_attr,
        name_attr=name_attr,
    )

    return new_G_pos, new_G_neg


def find_partition_signed_CPM(
    G_pos,
    G_neg,
    alpha=0.5,
    res_pos=0.5,
    res_neg=0.5,
    seed=None,
    gene_id_attr="gene_id",
    name_attr="name",
    return_both=False
):
    """
    Signed Leiden using CPM partitions.

    Input graphs may be NetworkX or igraph.
    Internal computation is performed on aligned igraph graphs.
    """
    aligned_G_pos, aligned_G_neg = align_graphs(
        G_pos,
        G_neg,
        gene_id_attr=gene_id_attr,
        name_attr=name_attr,
    )

    n_nodes = aligned_G_pos.vcount()
    initial_membership = list(range(n_nodes))

    part_pos = la.CPMVertexPartition(
        aligned_G_pos,
        resolution_parameter=res_pos,
        weights="weight" if "weight" in aligned_G_pos.es.attributes() else None,
        initial_membership=initial_membership,
    )

    part_neg = la.CPMVertexPartition(
        aligned_G_neg,
        resolution_parameter=res_neg,
        weights="weight" if "weight" in aligned_G_neg.es.attributes() else None,
        initial_membership=initial_membership,
    )

    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)

    w_pos = alpha
    w_neg = -(1.0 - alpha)

    diff = 1
    while diff > 0:
        diff = optimiser.optimise_partition_multiplex(
            partitions=[part_pos, part_neg],
            layer_weights=[w_pos, w_neg],
        )

    if return_both:
        return part_pos, part_neg
    return part_pos


def find_partition_signed_modularity(
    G_pos,
    G_neg,
    alpha=1,
    res_pos=1.0,
    res_neg=1.0,
    seed=None,
    gene_id_attr="gene_id",
    name_attr="name",
    return_both=False
):
    """
    Signed Leiden using RBConfigurationVertexPartition.

    Input graphs may be NetworkX or igraph.
    Returns partitions on aligned igraph graphs.
    """
    aligned_G_pos, aligned_G_neg = align_graphs(
        G_pos,
        G_neg,
        gene_id_attr=gene_id_attr,
        name_attr=name_attr,
    )

    part_pos = la.find_partition(
        aligned_G_pos,
        la.RBConfigurationVertexPartition,
        weights="weight" if "weight" in aligned_G_pos.es.attributes() else None,
        resolution_parameter=res_pos,
        seed=seed,
    )

    part_neg = la.find_partition(
        aligned_G_neg,
        la.RBConfigurationVertexPartition,
        weights="weight" if "weight" in aligned_G_neg.es.attributes() else None,
        resolution_parameter=res_neg,
        seed=seed,
    )

    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)

    diff = 1
    while diff > 0:
        diff = optimiser.optimise_partition_multiplex(
            [part_pos, part_neg],
            layer_weights=[alpha, -(1-alpha)]
        )

    if return_both:
        return part_pos, part_neg
    return part_pos


def find_partition_signed_ModularityVertexPartition(
    G_pos,
    G_neg,
    weight_1,
    weight_2,
    seed=None,
    gene_id_attr="gene_id",
    name_attr="name",
):
    """
    Signed Leiden using ModularityVertexPartition.

    Input graphs may be NetworkX or igraph.
    Returns partitions on aligned igraph graphs.
    """
    aligned_G_pos, aligned_G_neg = align_graphs(
        G_pos,
        G_neg,
        gene_id_attr=gene_id_attr,
        name_attr=name_attr,
    )

    part_pos = la.ModularityVertexPartition(
        aligned_G_pos,
        weights="weight" if "weight" in aligned_G_pos.es.attributes() else None,
    )
    part_neg = la.ModularityVertexPartition(
        aligned_G_neg,
        weights="weight" if "weight" in aligned_G_neg.es.attributes() else None,
    )

    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)

    diff = 1
    while diff > 0:
        diff = optimiser.optimise_partition_multiplex(
            [part_pos, part_neg],
            layer_weights=[weight_1, weight_2],
        )

    return part_pos, part_neg