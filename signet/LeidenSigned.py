import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg as la

def align_graphs(G_pos, G_neg):
    """
    G_pos と G_neg のノードを和集合で統一し、インデックス順序を完全に一致させた
    新しいグラフを作成して返します。
    """
    ids_pos = set(G_pos.vs["gene_id"]) if "gene_id" in G_pos.vs.attributes() else set(range(G_pos.vcount()))
    ids_neg = set(G_neg.vs["gene_id"]) if "gene_id" in G_neg.vs.attributes() else set(range(G_neg.vcount()))

    all_gene_ids = sorted(list(ids_pos | ids_neg))

    # ID -> 新インデックス のマッピング
    id_to_idx = {gid: i for i, gid in enumerate(all_gene_ids)}

    # 2. 新しい空グラフを作成（ノード数は和集合のサイズ）
    new_G_pos = ig.Graph(len(all_gene_ids))
    new_G_neg = ig.Graph(len(all_gene_ids))

    # 属性（gene_id, gene_name）のコピー（必要であれば）
    new_G_pos.vs["gene_id"] = all_gene_ids
    new_G_neg.vs["gene_id"] = all_gene_ids
    # name属性の引き継ぎ処理は省略（元のグラフから引く必要があります）

    # 3. エッジのコピー（IDベースでインデックスを変換）
    # G_pos のエッジを移行
    edges_pos = []
    weights_pos = []
    has_weight_pos = "weight" in G_pos.es.attributes()

    # 元のIDを使ってエッジを取得し、新インデックスに変換
    original_ids_pos = G_pos.vs["gene_id"]
    for edge in G_pos.es:
        src_id = original_ids_pos[edge.source]
        tgt_id = original_ids_pos[edge.target]
        edges_pos.append((id_to_idx[src_id], id_to_idx[tgt_id]))
        if has_weight_pos:
            weights_pos.append(edge["weight"])

    new_G_pos.add_edges(edges_pos)
    if has_weight_pos: new_G_pos.es["weight"] = weights_pos

    # G_neg のエッジを移行
    edges_neg = []
    weights_neg = []
    has_weight_neg = "weight" in G_neg.es.attributes()
    original_ids_neg = G_neg.vs["gene_id"]

    for edge in G_neg.es:
        src_id = original_ids_neg[edge.source]
        tgt_id = original_ids_neg[edge.target]
        edges_neg.append((id_to_idx[src_id], id_to_idx[tgt_id]))
        if has_weight_neg:
            weights_neg.append(edge["weight"])

    new_G_neg.add_edges(edges_neg)
    if has_weight_neg: new_G_neg.es["weight"] = weights_neg

    return new_G_pos, new_G_neg

# CPM（Constant Potts Model）
def find_partition_signed_CPM(G_pos, G_neg, alpha=0.5, res_pos=0.5, res_neg=0.5, seed=None):
    # グラフのアライメント（インデックスの統一）
    aligned_G_pos, aligned_G_neg = align_graphs(G_pos, G_neg)
    n_nodes = aligned_G_pos.vcount()
    initial_membership = list(range(n_nodes))

    part_pos = la.CPMVertexPartition(
        aligned_G_pos,
        resolution_parameter=res_pos,
        weights='weight' if 'weight' in aligned_G_pos.es.attributes() else None,
        initial_membership=initial_membership
    )

    part_neg = la.CPMVertexPartition(
        aligned_G_neg,
        resolution_parameter=res_neg,
        weights='weight' if 'weight' in aligned_G_neg.es.attributes() else None,
        initial_membership=initial_membership
    )

    # マルチプレックス最適化
    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)

    # 重要: 負のグラフの重みはマイナスにする
    # weight_1, weight_2 を直接渡すより、alphaで制御するほうが安全です
    w_pos = alpha
    w_neg = -1.0 * (1.0 - alpha)

    # optimize_partition_multiplex は内部で improvement がなくなるまで回りますが、
    # 念のため n_iterations を指定するか、diffループを使います
    diff = 1
    while diff > 0:
        diff = optimiser.optimise_partition_multiplex(
            partitions=[part_pos, part_neg],
            layer_weights=[w_pos, w_neg]
        )

    return part_pos


def find_partition_signed_CPM_old(G_pos, G_neg, weight_1, weight_2, res_pos=1.0, res_neg=1.0, seed=None):
    optimiser = la.Optimiser()

    # G_neg にあって G_pos にない gene_id を追加
    pos_gene_ids = set(G_pos.vs["gene_id"])
    neg_gene_ids = set(G_neg.vs["gene_id"])

    for gid in neg_gene_ids - pos_gene_ids:
        idx = G_neg.vs.find(gene_id=gid).index
        G_pos.add_vertex(
            gene_id=gid,
            gene_name=G_neg.vs[idx]["gene_name"]
        )

    # G_pos にあって G_neg にない gene_id を追加
    for gid in pos_gene_ids - neg_gene_ids:
        idx = G_pos.vs.find(gene_id=gid).index
        G_neg.add_vertex(
            gene_id=gid,
            gene_name=G_pos.vs[idx]["gene_name"]
        )

    part_pos = la.find_partition(G_pos, la.CPMVertexPartition, resolution_parameter=res_pos, seed=seed);
    part_neg = la.find_partition(G_neg, la.CPMVertexPartition, resolution_parameter=res_neg, seed=seed);

    diff = 1
    while diff > 0:
         diff = optimiser.optimise_partition_multiplex(
              [part_pos, part_neg],
              layer_weights=[weight_1, weight_2]);

    return part_pos, part_neg


def find_partition_signed_modularity(G_pos, G_neg, weight_1, weight_2, res_pos=1.0, res_neg=1.0, seed=None):
    optimiser = la.Optimiser()

    # G_neg にあって G_pos にない gene_id を追加
    pos_gene_ids = set(G_pos.vs["gene_id"])
    neg_gene_ids = set(G_neg.vs["gene_id"])

    for gid in neg_gene_ids - pos_gene_ids:
        idx = G_neg.vs.find(gene_id=gid).index
        G_pos.add_vertex(
            gene_id=gid,
            gene_name=G_neg.vs[idx]["gene_name"]
        )

    # G_pos にあって G_neg にない gene_id を追加
    for gid in pos_gene_ids - neg_gene_ids:
        idx = G_pos.vs.find(gene_id=gid).index
        G_neg.add_vertex(
            gene_id=gid,
            gene_name=G_pos.vs[idx]["gene_name"]
        )

    part_pos = la.find_partition(G_pos,
                                la.RBConfigurationVertexPartition,
                                weights="weight",
                                resolution_parameter=res_pos,
                                seed=seed
    )
    part_neg = la.find_partition(G_neg,
                                la.RBConfigurationVertexPartition,
                                weights="weight",
                                resolution_parameter=res_neg,
                                seed=seed
    )

    diff = 1
    while diff > 0:
         diff = optimiser.optimise_partition_multiplex(
              [part_pos, part_neg],
              layer_weights=[weight_1, weight_2]);

    return part_pos, part_neg


def find_partition_signed_ModularityVertexPartition(G_pos, G_neg, weight_1, weight_2):
    optimiser = la.Optimiser()

    # G_neg にあって G_pos にない gene_id を追加
    pos_gene_ids = set(G_pos.vs["gene_id"])
    neg_gene_ids = set(G_neg.vs["gene_id"])

    for gid in neg_gene_ids - pos_gene_ids:
        idx = G_neg.vs.find(gene_id=gid).index
        G_pos.add_vertex(
            gene_id=gid,
            gene_name=G_neg.vs[idx]["gene_name"]
        )

    # G_pos にあって G_neg にない gene_id を追加
    for gid in pos_gene_ids - neg_gene_ids:
        idx = G_pos.vs.find(gene_id=gid).index
        G_neg.add_vertex(
            gene_id=gid,
            gene_name=G_pos.vs[idx]["gene_name"]
        )

    part_pos = la.ModularityVertexPartition(G_pos, weights='weight');
    part_neg = la.ModularityVertexPartition(G_neg, weights='weight');

    diff = 1
    while diff > 0:
         diff = optimiser.optimise_partition_multiplex(
              [part_pos, part_neg],
              layer_weights=[weight_1, weight_2]);

    return part_pos, part_neg
