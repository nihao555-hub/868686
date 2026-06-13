# -*- coding: utf-8 -*-
"""
实验五：聚类算法（K-Means / 层次聚类 / 密度聚类 DBSCAN）
============================================================
要求：
  1. 自己编程实现层次聚类法（凝聚式 Agglomerative，本程序支持 single/
     complete/average 三种连接方式）；
  2. 自己编程实现密度聚类法（DBSCAN）；
  3. 对 K-Means、层次聚类、密度聚类进行对比分析，用表格 + 图形展示
     （含各算法参数、取值与优缺点）。

数据集：ex7data2.csv（Andrew Ng ML 课程，300 个二维样本，含 3 个自然簇）。
三种算法均从零用 numpy 实现；轮廓系数（silhouette）仅作为评价指标调用 sklearn。

运行：python clustering.py
输出：terminal 打印对比表 + 生成多张对比图 png + comparison.json
"""
import os
import json
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

RNG = np.random.RandomState(42)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def find_dataset():
    """定位 ex7data2.csv（在“聚类”目录下的 练1/data 中）。"""
    roots = [OUT_DIR, os.environ.get('LAB5_ROOT', '')]
    for start in roots:
        if not start or not os.path.isdir(start):
            continue
        for dirpath, _dirs, files in os.walk(start):
            for fn in files:
                if fn == 'ex7data2.csv':
                    return os.path.join(dirpath, fn)
        # 向上一层再找
        parent = os.path.dirname(start)
        for dirpath, _dirs, files in os.walk(parent):
            for fn in files:
                if fn == 'ex7data2.csv':
                    return os.path.join(dirpath, fn)
    raise FileNotFoundError('找不到 ex7data2.csv')


def load_data():
    path = find_dataset()
    data = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:        # 跳过表头 X1,X2
                continue
            parts = line.replace(',', ' ').split()
            data.append([float(parts[0]), float(parts[1])])
    return np.array(data)


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def make_moons(n=300, noise=0.06, seed=42):
    """生成双月牙形（非凸）数据集，用于展示算法对任意形状的适应能力。"""
    rng = np.random.RandomState(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = np.linspace(0, np.pi, n1)
    outer = np.c_[np.cos(t1), np.sin(t1)]
    t2 = np.linspace(0, np.pi, n2)
    inner = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5]
    X = np.vstack([outer, inner])
    X += rng.normal(scale=noise, size=X.shape)
    return X


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def pairwise_dist(X):
    """返回 n×n 欧氏距离矩阵。"""
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    d2 = np.maximum(d2, 0)
    return np.sqrt(d2)


# ============================================================================
# 1. K-Means（从零实现）
# ============================================================================
def kmeans(X, k, max_iter=100, tol=1e-6, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    # k-means++ 初始化
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        # 分配
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers]).T
        new_labels = np.argmin(dists, axis=1)
        # 更新
        new_centers = np.array([
            X[new_labels == j].mean(axis=0) if np.any(new_labels == j)
            else centers[j] for j in range(k)
        ])
        shift = np.max(np.abs(new_centers - centers))
        centers, labels = new_centers, new_labels
        if shift < tol:
            break
    # 计算 SSE
    sse = sum(np.sum((X[labels == j] - centers[j]) ** 2) for j in range(k))
    return labels, centers, sse, it + 1


# ============================================================================
# 2. 层次聚类 - 凝聚式（从零实现）
# ============================================================================
def hierarchical(X, k, linkage='average'):
    """
    凝聚式层次聚类：初始每个点为一簇，每次合并“距离最近”的两个簇，
    直到剩下 k 个簇。linkage: single(最小) / complete(最大) / average(平均)。
    """
    n = X.shape[0]
    D = pairwise_dist(X)
    clusters = {i: [i] for i in range(n)}     # 簇编号 -> 成员索引列表
    # 簇间距离表（用字典缓存，初始就是点间距离）
    cdist = {}
    ids = list(clusters.keys())
    for i in range(n):
        for j in range(i + 1, n):
            cdist[(i, j)] = D[i, j]

    next_id = n
    merge_history = []
    while len(clusters) > k:
        # 找到距离最近的两个簇
        (ci, cj), best = min(cdist.items(), key=lambda kv: kv[1])
        # 合并 ci, cj -> 新簇 next_id
        members = clusters[ci] + clusters[cj]
        merge_history.append((ci, cj, best, len(members)))
        del clusters[ci]
        del clusters[cj]
        # 删除涉及 ci/cj 的距离项
        cdist = {key: v for key, v in cdist.items()
                 if ci not in key and cj not in key}
        # 计算新簇与其它簇的距离
        for cid, mem in clusters.items():
            sub = D[np.ix_(members, mem)]
            if linkage == 'single':
                dist = sub.min()
            elif linkage == 'complete':
                dist = sub.max()
            else:  # average
                dist = sub.mean()
            key = (min(cid, next_id), max(cid, next_id))
            cdist[key] = dist
        clusters[next_id] = members
        next_id += 1

    # 输出标签
    labels = np.zeros(n, dtype=int)
    for lab, (cid, mem) in enumerate(clusters.items()):
        for idx in mem:
            labels[idx] = lab
    return labels, merge_history


# ============================================================================
# 3. 密度聚类 - DBSCAN（从零实现）
# ============================================================================
def dbscan(X, eps, min_pts):
    """
    DBSCAN：基于密度的聚类。
    返回 labels，其中 -1 表示噪声点。
    """
    n = X.shape[0]
    D = pairwise_dist(X)
    labels = np.full(n, -2, dtype=int)   # -2: 未访问, -1: 噪声, >=0: 簇号
    cluster_id = -1

    def region_query(p):
        return np.where(D[p] <= eps)[0]

    for p in range(n):
        if labels[p] != -2:
            continue
        neighbors = region_query(p)
        if len(neighbors) < min_pts:
            labels[p] = -1            # 暂标记为噪声
            continue
        cluster_id += 1
        labels[p] = cluster_id
        # 扩展簇（广度优先）
        seeds = list(neighbors)
        i = 0
        while i < len(seeds):
            q = seeds[i]
            if labels[q] == -1:        # 之前的噪声点 -> 变为边界点
                labels[q] = cluster_id
            if labels[q] == -2:
                labels[q] = cluster_id
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_pts:   # q 也是核心点
                    seeds.extend(list(q_neighbors))
            i += 1
    return labels


# ============================================================================
# 评价与可视化
# ============================================================================
def silhouette(X, labels):
    """轮廓系数（忽略噪声点 -1）。调用 sklearn 作为评价指标。"""
    try:
        from sklearn.metrics import silhouette_score
        mask = labels >= 0
        if len(set(labels[mask])) < 2:
            return float('nan')
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return float('nan')


def plot_clusters(ax, X, labels, title, centers=None):
    uniq = sorted(set(labels))
    cmap = plt.get_cmap('tab10')
    for idx, lab in enumerate(uniq):
        m = labels == lab
        if lab == -1:
            ax.scatter(X[m, 0], X[m, 1], c='lightgray', marker='x',
                       s=25, label='噪声')
        else:
            ax.scatter(X[m, 0], X[m, 1], color=cmap(idx % 10), s=25,
                       label=f'簇{lab}')
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='*',
                   s=220, edgecolor='black', label='质心', zorder=5)
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best')


def main():
    X = load_data()
    print(f'数据集 ex7data2：{X.shape[0]} 个样本，{X.shape[1]} 维')
    K = 3

    results = {}

    # ---- K-Means ----
    t0 = time.time()
    km_labels, km_centers, km_sse, km_iter = kmeans(X, K)
    km_t = time.time() - t0
    results['kmeans'] = dict(
        params=f'k={K}, k-means++初始化, 迭代{km_iter}次',
        n_clusters=int(len(set(km_labels))), sse=float(km_sse),
        silhouette=silhouette(X, km_labels), n_noise=0, time=km_t)

    # ---- 层次聚类 ----
    t0 = time.time()
    hc_labels, hist = hierarchical(X, K, linkage='average')
    hc_t = time.time() - t0
    results['hierarchical'] = dict(
        params=f'k={K}, linkage=average(平均连接)',
        n_clusters=int(len(set(hc_labels))), sse=None,
        silhouette=silhouette(X, hc_labels), n_noise=0, time=hc_t)

    # ---- DBSCAN ----
    # 经验法：基于第 k 近邻距离选 eps
    D = pairwise_dist(X)
    kth = np.sort(D, axis=1)[:, 4]      # 第4近邻
    eps = float(np.percentile(kth, 90))
    min_pts = 5
    t0 = time.time()
    db_labels = dbscan(X, eps=eps, min_pts=min_pts)
    db_t = time.time() - t0
    n_noise = int(np.sum(db_labels == -1))
    n_cl = len(set(db_labels[db_labels >= 0]))
    results['dbscan'] = dict(
        params=f'eps={eps:.3f}, min_pts={min_pts}',
        n_clusters=int(n_cl), sse=None,
        silhouette=silhouette(X, db_labels), n_noise=n_noise, time=db_t)

    # ---- 打印对比表 ----
    print('\n================== 三种聚类算法对比 ==================')
    header = f'{"算法":<14}{"簇数":<6}{"噪声点":<8}{"轮廓系数":<12}{"耗时(s)":<10}'
    print(header)
    name_map = {'kmeans': 'K-Means', 'hierarchical': '层次聚类(平均)',
                'dbscan': 'DBSCAN'}
    for key in ('kmeans', 'hierarchical', 'dbscan'):
        r = results[key]
        print(f'{name_map[key]:<14}{r["n_clusters"]:<6}{r["n_noise"]:<8}'
              f'{r["silhouette"]:<12.4f}{r["time"]:<10.4f}')
    print('参数：')
    for key in ('kmeans', 'hierarchical', 'dbscan'):
        print(f'  {name_map[key]}: {results[key]["params"]}')
    print('======================================================')

    with open(os.path.join(OUT_DIR, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- 可视化：原始数据 + 三种聚类结果 ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].scatter(X[:, 0], X[:, 1], c='steelblue', s=25)
    axes[0, 0].set_title('原始数据（未聚类）')
    plot_clusters(axes[0, 1], X, km_labels,
                  f'K-Means (k={K})\n轮廓系数={results["kmeans"]["silhouette"]:.3f}',
                  centers=km_centers)
    plot_clusters(axes[1, 0], X, hc_labels,
                  f'层次聚类(平均连接, k={K})\n轮廓系数={results["hierarchical"]["silhouette"]:.3f}')
    plot_clusters(axes[1, 1], X, db_labels,
                  f'DBSCAN (eps={eps:.2f}, minPts={min_pts})\n'
                  f'轮廓系数={results["dbscan"]["silhouette"]:.3f}, 噪声={n_noise}')
    fig.suptitle('三种聚类算法在 ex7data2 数据集上的对比', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, 'cluster_comparison.png'), dpi=150)
    plt.close(fig)

    # ---- K-Means 肘部法则图（辅助选 k）----
    sses = []
    Ks = range(1, 9)
    for kk in Ks:
        _, _, sse, _ = kmeans(X, kk)
        sses.append(sse)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(list(Ks), sses, 'bo-')
    ax.set_xlabel('簇数 k')
    ax.set_ylabel('SSE（簇内平方和）')
    ax.set_title('K-Means 肘部法则：选择最优 k')
    ax.axvline(3, color='red', linestyle='--', label='肘点 k=3')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'kmeans_elbow.png'), dpi=150)
    plt.close(fig)

    # ---- 非凸形状数据（双月牙）对比：突出三种算法的优缺点 ----
    Xm = make_moons(n=300, noise=0.06, seed=42)
    mk_labels, mk_centers, _, _ = kmeans(Xm, 2)
    mh_labels, _ = hierarchical(Xm, 2, linkage='single')   # 单连接擅长链状/任意形状
    # eps 由 k-距离法选取：5近邻距离中位数约 0.088，取 0.15 落在密度间隙之上
    eps_m = 0.15
    md_labels = dbscan(Xm, eps=eps_m, min_pts=5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    plot_clusters(axes[0], Xm, mk_labels,
                  f'K-Means(k=2)\n轮廓={silhouette(Xm, mk_labels):.3f}',
                  centers=mk_centers)
    plot_clusters(axes[1], Xm, mh_labels,
                  f'层次聚类(单连接,k=2)\n轮廓={silhouette(Xm, mh_labels):.3f}')
    plot_clusters(axes[2], Xm, md_labels,
                  f'DBSCAN(eps={eps_m:.2f},minPts=5)\n轮廓={silhouette(Xm, md_labels):.3f}, '
                  f'噪声={int(np.sum(md_labels==-1))}')
    fig.suptitle('非凸形状（双月牙）数据：DBSCAN 与单连接层次聚类可识别任意形状，K-Means 失效', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, 'nonconvex_comparison.png'), dpi=150)
    plt.close(fig)

    print('[plot] 已保存 cluster_comparison.png, kmeans_elbow.png, nonconvex_comparison.png')


if __name__ == '__main__':
    main()
