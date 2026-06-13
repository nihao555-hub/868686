# -*- coding: utf-8 -*-
"""实验报告6：降维（Dimensionality Reduction）

基本要求：自选高维数据，进行三种降维处理，并分别在降至 2 维和 3 维时对比可视化：
  · 只使用 PCA
  · 只使用 t-SNE
  · 将 PCA 与 t-SNE 结合（先 PCA 预降维，再 t-SNE）

数据集：sklearn 手写数字 digits（1797 个样本，64 维，10 个类别）。
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import labcommon as C

LAB = os.path.dirname(C.find_file("Lab6.pdf"))
OUT = os.path.join(LAB, "实验报告6-降维")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}
CMAP = "tab10"


def scatter2d(ax, emb, y, title):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=CMAP, s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def scatter3d(ax, emb, y, title):
    sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=y, cmap=CMAP, s=8, alpha=0.7)
    ax.set_title(title)
    return sc


def run():
    digits = load_digits()
    X, y = digits.data, digits.target
    R["shape"] = X.shape
    R["n_classes"] = len(np.unique(y))
    Xs = StandardScaler().fit_transform(X)

    # ---- PCA ----
    t0 = time.time()
    pca2 = PCA(n_components=2, random_state=0).fit(Xs)
    pca2_emb = pca2.transform(Xs)
    pca3_emb = PCA(n_components=3, random_state=0).fit_transform(Xs)
    R["pca_time"] = time.time() - t0
    R["pca2_var"] = pca2.explained_variance_ratio_.sum()
    pca_full = PCA(random_state=0).fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    R["pca30_var"] = cum[29]
    R["n_for_90"] = int(np.argmax(cum >= 0.90) + 1)

    # ---- t-SNE only ----
    t0 = time.time()
    tsne2_emb = TSNE(n_components=2, init="random", perplexity=30,
                     random_state=0).fit_transform(Xs)
    tsne3_emb = TSNE(n_components=3, init="random", perplexity=30,
                     random_state=0).fit_transform(Xs)
    R["tsne_time"] = time.time() - t0

    # ---- PCA + t-SNE ----
    t0 = time.time()
    X_pca30 = PCA(n_components=30, random_state=0).fit_transform(Xs)
    comb2_emb = TSNE(n_components=2, init="pca", perplexity=30,
                     random_state=0).fit_transform(X_pca30)
    comb3_emb = TSNE(n_components=3, init="pca", perplexity=30,
                     random_state=0).fit_transform(X_pca30)
    R["comb_time"] = time.time() - t0

    # ---- 解释方差曲线 ----
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(range(1, len(cum) + 1), cum, color="#2980b9")
    ax.axhline(0.90, color="#c0392b", ls="--", label="90% 累计方差")
    ax.axvline(R["n_for_90"], color="#27ae60", ls=":",
               label=f"{R['n_for_90']} 个主成分")
    ax.set_xlabel("主成分个数")
    ax.set_ylabel("累计解释方差比例")
    ax.set_title("PCA 累计解释方差曲线")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pca_variance.png"), bbox_inches="tight")
    plt.close()

    # ---- 2D 对比 ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    scatter2d(axes[0], pca2_emb, y, f"PCA (2D)\n解释方差={R['pca2_var']:.1%}")
    scatter2d(axes[1], tsne2_emb, y, "t-SNE (2D)")
    sc = scatter2d(axes[2], comb2_emb, y, "PCA(30) + t-SNE (2D)")
    fig.colorbar(sc, ax=axes, fraction=0.025, label="数字类别")
    fig.suptitle("三种降维方法降至 2 维的可视化对比", fontsize=14)
    plt.savefig(os.path.join(FIG, "compare_2d.png"), bbox_inches="tight")
    plt.close()

    # ---- 3D 对比 ----
    fig = plt.figure(figsize=(14, 4.8))
    for i, (emb, title) in enumerate([
        (pca3_emb, "PCA (3D)"),
        (tsne3_emb, "t-SNE (3D)"),
        (comb3_emb, "PCA(30) + t-SNE (3D)"),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        scatter3d(ax, emb, y, title)
    fig.suptitle("三种降维方法降至 3 维的可视化对比", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "compare_3d.png"), bbox_inches="tight")
    plt.close()


def build_report():
    rep = C.Report("报告6", "降维算法（PCA / t-SNE）")
    rep.h("一、实验目的")
    rep.bullets([
        "掌握降维算法的基本原理（PCA 线性降维、t-SNE 非线性流形降维）及 Python 实现；",
        "理解线性降维与非线性降维的差异及各自适用场景；",
        "能够运用降维算法对高维数据进行 2 维/3 维可视化并对比分析。",
    ])

    rep.h("二、实验内容")
    rep.para(f"自选高维数据集（sklearn 手写数字 digits，形状 {R['shape']}，"
             f"共 {R['n_classes']} 个类别，每个样本为 8×8=64 维灰度向量），"
             f"分别采用以下三种降维方法，并在降至 2 维和 3 维时对比可视化结果：")
    rep.bullets([
        "只使用 PCA（主成分分析，线性降维）；",
        "只使用 t-SNE（t 分布随机邻域嵌入，非线性降维）；",
        "将 PCA 与 t-SNE 结合（先用 PCA 预降维到 30 维去噪/提速，再用 t-SNE）。",
    ])

    rep.h("三、实验步骤")
    rep.bullets([
        "数据准备：加载 digits 数据集并用 StandardScaler 标准化；",
        "PCA 降维：分别降至 2 维和 3 维，并绘制累计解释方差曲线确定信息保留量；",
        "t-SNE 降维：以原始 64 维为输入，perplexity=30，降至 2 维和 3 维；",
        "PCA+t-SNE：先用 PCA 将 64 维降至 30 维，再用 t-SNE（init='pca'）降维；",
        "可视化对比：按类别着色绘制三种方法的 2D/3D 散点并对比聚类效果与耗时。",
    ])

    rep.h("四、程序设计的核心代码")
    rep.code(
        "Xs = StandardScaler().fit_transform(X)              # 标准化\n"
        "# 1) 只用 PCA\n"
        "pca2 = PCA(n_components=2).fit_transform(Xs)\n"
        "# 2) 只用 t-SNE\n"
        "tsne2 = TSNE(n_components=2, init='random', perplexity=30).fit_transform(Xs)\n"
        "# 3) PCA + t-SNE（先 PCA 预降维到 30 维，再 t-SNE）\n"
        "X30  = PCA(n_components=30).fit_transform(Xs)\n"
        "comb2 = TSNE(n_components=2, init='pca', perplexity=30).fit_transform(X30)",
        caption="① 三种降维方法（2D；3D 只需将 n_components 改为 3）")

    rep.h("五、实验结果")
    rep.h2("5.1 PCA 解释方差")
    rep.para(f"PCA 前 2 个主成分仅保留约 {R['pca2_var']:.1%} 的方差，前 30 个主成分"
             f"累计保留约 {R['pca30_var']:.1%}，达到 90% 累计方差需要约 "
             f"{R['n_for_90']} 个主成分。说明 digits 数据本征维度较高，"
             f"仅靠 2 维线性投影难以充分区分 10 个数字。")
    rep.image(os.path.join(FIG, "pca_variance.png"), 5.6,
              "图1 PCA 累计解释方差曲线")

    rep.h2("5.2 降至 2 维的对比")
    rep.image(os.path.join(FIG, "compare_2d.png"), 6.6,
              "图2 PCA / t-SNE / PCA+t-SNE 降至 2 维的可视化对比")
    rep.para("可以看到：PCA(2D) 各数字类别相互重叠严重，只能区分少数差异大的数字；"
             "t-SNE(2D) 将同类数字紧凑地聚成清晰的“团簇”，类间间隔明显；"
             "PCA+t-SNE(2D) 聚类效果与纯 t-SNE 相当，但因先做了 PCA 预降维而更稳定。")

    rep.h2("5.3 降至 3 维的对比")
    rep.image(os.path.join(FIG, "compare_3d.png"), 6.6,
              "图3 PCA / t-SNE / PCA+t-SNE 降至 3 维的可视化对比")
    rep.para("3 维下结论一致：t-SNE 与 PCA+t-SNE 形成 10 个分离良好的簇，"
             "而 PCA 仍有较多类别交叠。")

    rep.h2("5.4 三种方法耗时对比")
    rep.table(["降维方法", "说明", "耗时(2D+3D)"], [
        ["PCA", "线性、可解释、速度极快", f"{R['pca_time']:.2f}s"],
        ["t-SNE", "非线性、聚类好、较慢", f"{R['tsne_time']:.1f}s"],
        ["PCA+t-SNE", "先 PCA 去噪提速，再 t-SNE", f"{R['comb_time']:.1f}s"],
    ], caption="表1 三种降维方法对比", col_widths=[1.8, 3.0, 1.6])

    rep.h("六、实验体会")
    rep.para(
        "本实验对比了 PCA 与 t-SNE 两类降维方法，收获颇深。①原理差异：PCA 是线性降维，"
        "通过最大化方差/特征值分解寻找正交主成分，全局结构与可解释性强、速度快，但无法"
        "刻画非线性流形，因此在 digits 上 2 维投影类别严重重叠；t-SNE 是非线性方法，"
        "通过保持高维空间的局部邻域概率分布，能将同类样本聚成紧凑簇，可视化效果远优于 PCA，"
        "但计算开销大、结果依赖 perplexity 等超参且不保留全局距离。②结合策略：先用 PCA 将"
        "64 维预降到 30 维（保留约 "
        f"{R['pca30_var']:.0%} 的方差）再做 t-SNE，可去除噪声、降低计算量、使 t-SNE 更稳定，"
        "是工程中常用的组合，兼顾了效率与可视化质量。③总结：PCA 适合做预处理与全局分析，"
        "t-SNE 适合做高维数据的探索性可视化，二者结合往往效果最佳。")

    rep.teacher_eval()
    docx = os.path.join(OUT, "机器学习实验报告6-降维.docx")
    rep.save(docx)
    return docx


if __name__ == "__main__":
    run()
    docx = build_report()
    print("pca2_var=%.3f pca30_var=%.3f n90=%d times: pca=%.2f tsne=%.1f comb=%.1f" % (
        R["pca2_var"], R["pca30_var"], R["n_for_90"],
        R["pca_time"], R["tsne_time"], R["comb_time"]))
    print("SAVED:", docx)
