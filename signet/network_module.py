import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import igraph as ig

def load_graph_from_EEISP(filename, threshold):
    G = nx.Graph()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_id1 = parts[0]
            gene_id2 = parts[1]
            gene_name1 = parts[2]
            gene_name2 = parts[3]
            weight = float(parts[4])
            if weight >= threshold:
                G.add_node(gene_id1, name=gene_name1)
                G.add_node(gene_id2, name=gene_name2)
                G.add_edge(gene_id1, gene_id2, weight=weight)
    return G

def load_graph_from_EEISP_igraph(filename, threshold):
    edges = []
    weights = []
    gene_ids = set()
    gene_name_map = {}

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            gene_id1 = parts[0]
            gene_id2 = parts[1]
            gene_name1 = parts[2]
            gene_name2 = parts[3]
            weight = float(parts[4])

            if weight >= threshold:
                edges.append((gene_id1, gene_id2))
                weights.append(weight)

                gene_ids.add(gene_id1)
                gene_ids.add(gene_id2)
                gene_name_map[gene_id1] = gene_name1
                gene_name_map[gene_id2] = gene_name2

    gene_ids = sorted(gene_ids)
    id_to_index = {gid: i for i, gid in enumerate(gene_ids)}

    edge_indices = [(id_to_index[u], id_to_index[v]) for u, v in edges]

    g = ig.Graph(n=len(gene_ids), edges=edge_indices)
    g.vs["gene_id"] = gene_ids
    g.vs["gene_name"] = [gene_name_map[gid] for gid in gene_ids]
    g.es["weight"] = weights

    return g



def count_nodes_in_communities(partition):
    community_counts = {}
    for node, community in partition.items():
        if community in community_counts:
            community_counts[community] += 1
        else:
            community_counts[community] = 1

    for community, count in sorted(community_counts.items()):
        print(f"Community {community}: {count} nodes")

def find_communities_of_genes_nx(graph, partition, gene_names):
    results = {}

    # gene_name -> node_id の逆引きを一度作る（効率・安全性のため）
    name_to_node = {
        data.get("name"): node
        for node, data in graph.nodes(data=True)
        if "name" in data
    }

    for gene_name in gene_names:
        node_id = name_to_node.get(gene_name)

        if node_id is None:
            results[gene_name] = "Gene not found in the graph."
            continue

        community_id = partition.get(node_id)
        if community_id is None:
            results[gene_name] = "Community for gene not found."
        else:
            results[gene_name] = community_id

    return results


def find_communities_of_genes_igraph(graph, partition, gene_names):
    results = {}
    for gene_name in gene_names:

        node_id = graph.vs.find(name=gene_name).index if graph.vs.find(name=gene_name) else None
        if node_id is None:
            results[gene_name] = "Gene not found in the graph."
            continue

        community_id = partition[node_id] if node_id in partition else None
        if community_id is None:
            results[gene_name] = "Community for gene not found."
        else:
            results[gene_name] = community_id
    return results


def generate_signednetwork(community_size, num_communities, intra_edges, inter_edges, p1, p2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    G_positive = nx.Graph()
    G_negative = nx.Graph()

    for i in range(num_communities):
        G_tmp = nx.gnm_random_graph(community_size, intra_edges)
        G_tmp = nx.relabel_nodes(G_tmp, {node: node + i*community_size for node in G_tmp.nodes()})
        G_positive = nx.compose(G_positive, G_tmp)

    inter_edge_candidates = [(i, j) for i in range(community_size*num_communities) for j in range(i+1, community_size*num_communities) if abs(i//community_size - j//community_size) == 1]
    inter_edge_selected = np.random.choice(len(inter_edge_candidates), inter_edges * num_communities, replace=False)
    negative_edges = [inter_edge_candidates[i] for i in inter_edge_selected]
    G_negative.add_edges_from(negative_edges)

    G_positive.add_nodes_from(G_negative.nodes())
    G_negative.add_nodes_from(G_positive.nodes())

    positive_edges = list(G_positive.edges())
    for edge_index in np.random.choice(len(positive_edges), int(intra_edges * p1), replace=False):
        edge = positive_edges[edge_index]
        if edge[0]//community_size == edge[1]//community_size and G_positive.degree(edge[0]) > 1 and G_positive.degree(edge[1]) > 1:
            G_positive.remove_edge(*edge)
            G_negative.add_edge(*edge)

    negative_edges = list(G_negative.edges())
    for edge_index in np.random.choice(len(negative_edges), int(inter_edges * p2), replace=False):
        edge = negative_edges[edge_index]
        if edge[0]//community_size != edge[1]//community_size and G_negative.degree(edge[0]) > 1 and G_negative.degree(edge[1]) > 1:
            G_negative.remove_edge(*edge)
            G_positive.add_edge(*edge)

    return G_positive, G_negative


def calc_entropy(partition):
    cluster_sizes = list(partition.values())
    unique, counts = np.unique(cluster_sizes, return_counts=True)
    cluster_size_distribution = dict(zip(unique, counts))

    sizes = list(cluster_size_distribution.values())
    total_nodes = sum(sizes)
    num_clusters = len(sizes)

    entropy = -sum((size / total_nodes) * np.log(size / total_nodes) for size in sizes)
    normalized_entropy = entropy / np.log(num_clusters)

    return normalized_entropy

def calc_entropy_igraph(partition):
    """
    partition: igraph.clustering.VertexClustering
               (e.g. Leiden / Louvain の結果)
    """

    # 各クラスタのサイズ
    sizes = partition.sizes()
    total_nodes = sum(sizes)
    num_clusters = len(sizes)

    if num_clusters <= 1:
        return 0.0

    probs = np.array(sizes) / total_nodes
    entropy = -np.sum(probs * np.log(probs))
    normalized_entropy = entropy / np.log(num_clusters)

    return normalized_entropy