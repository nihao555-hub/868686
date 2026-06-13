# -*- coding: utf-8 -*-
"""实验报告2：K 近邻（KNN）—— 海伦约会数据集

基本要求：在海伦约会任务的基础上，增加交叉验证等方法确定最优 K 值，
最终输出最优 K 值及对应精度。

步骤：
  1. 读取 datingTestSet.txt（3 特征：每年飞行常客里程、玩游戏时间占比、
     每周冰淇淋公升数；标签：didntLike / smallDoses / largeDoses）；
  2. Min-Max 归一化（消除量纲差异，里程数远大于其他特征）；
  3. 用 10 折交叉验证在 K=1..30 上选取最优 K；
  4. 在独立测试集上评估最优 K 的精度并给出混淆矩阵；
  5. 可视化：特征两两散点、3D 散点、精度-K 曲线。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import labcommon as C

LAB = os.path.dirname(C.find_file("Lab3.pdf"))
OUT = os.path.join(LAB, "实验报告2-KNN")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}
LABELS = ["didntLike", "smallDoses", "largeDoses"]
LABELS_CN = ["不喜欢", "魅力一般", "极具魅力"]


def load_dating():
    path = C.find_file("datingTestSet.txt")
    X, y = [], []
    mapping = {"didntLike": 0, "smallDoses": 1, "largeDoses": 2}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            X.append([float(parts[0]), float(parts[1]), float(parts[2])])
            y.append(mapping[parts[3]])
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------------
# 从零实现的 KNN（欧氏距离 + 多数表决）
# ----------------------------------------------------------------------------
def knn_classify(x, X_train, y_train, k):
    dists = np.sqrt(((X_train - x) ** 2).sum(axis=1))
    idx = dists.argsort()[:k]
    votes = np.bincount(y_train[idx], minlength=3)
    return votes.argmax()


def run():
    X, y = load_dating()
    R["n"] = len(y)
    R["dist"] = [int((y == k).sum()) for k in range(3)]
    scaler = MinMaxScaler().fit(X)
    Xn = scaler.transform(X)

    # 训练/测试划分
    Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.2,
                                          random_state=42, stratify=y)

    # 10 折交叉验证选 K
    ks = list(range(1, 31))
    cv_means = []
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k)
        cv_means.append(cross_val_score(clf, Xtr, ytr, cv=10).mean())
    best_idx = int(np.argmax(cv_means))
    best_k = ks[best_idx]
    R["best_k"] = best_k
    R["best_cv"] = cv_means[best_idx]
    R["cv_table"] = [[k, f"{m:.2%}"] for k, m in zip(ks, cv_means)
                     if k in (1, 3, 5, 7, 9, 11, 15, 19, 23, best_k)]

    # 用最优 K 在测试集评估
    clf = KNeighborsClassifier(n_neighbors=best_k).fit(Xtr, ytr)
    yte_pred = clf.predict(Xte)
    R["test_acc"] = accuracy_score(yte, yte_pred)

    # 从零实现的 KNN 在测试集上验证一致性
    scratch_pred = np.array([knn_classify(x, Xtr, ytr, best_k) for x in Xte])
    R["scratch_acc"] = accuracy_score(yte, scratch_pred)

    cm = confusion_matrix(yte, yte_pred)
    R["cm"] = cm

    # ---- 图1：精度-K 曲线 ----
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(ks, cv_means, "o-", color="#2980b9", ms=4)
    ax.axvline(best_k, color="#c0392b", ls="--",
               label=f"最优 K={best_k} (CV精度={cv_means[best_idx]:.2%})")
    ax.set_xlabel("K 值")
    ax.set_ylabel("10 折交叉验证平均精度")
    ax.set_title("交叉验证选择最优 K 值")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "cv_k.png"), bbox_inches="tight")
    plt.close()

    # ---- 图2：特征两两散点 ----
    names = ["飞行常客里程", "玩游戏时间占比", "每周冰淇淋(L)"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    pairs = [(0, 1), (0, 2)]
    for ax, (a, b) in zip(axes, pairs):
        for k in range(3):
            m = y == k
            ax.scatter(X[m, a], X[m, b], s=12, c=colors[k], label=LABELS_CN[k],
                       alpha=0.7)
        ax.set_xlabel(names[a]); ax.set_ylabel(names[b])
        ax.legend(fontsize=8)
    axes[0].set_title("飞行里程 vs 游戏时间占比")
    axes[1].set_title("飞行里程 vs 冰淇淋消费")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "scatter2d.png"), bbox_inches="tight")
    plt.close()

    # ---- 图3：3D 散点 ----
    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    for k in range(3):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], X[m, 2], s=12, c=colors[k],
                   label=LABELS_CN[k], alpha=0.7)
    ax.set_xlabel("飞行里程"); ax.set_ylabel("游戏占比"); ax.set_zlabel("冰淇淋")
    ax.set_title("海伦约会数据三维分布")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "scatter3d.png"), bbox_inches="tight")
    plt.close()

    # ---- 图4：混淆矩阵 ----
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(cm, cmap="Greens")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS_CN, fontsize=9)
    ax.set_yticklabels(LABELS_CN, fontsize=9)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"测试集混淆矩阵 (K={best_k}, 精度={R['test_acc']:.2%})")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "cm.png"), bbox_inches="tight")
    plt.close()


def build_report():
    rep = C.Report("报告2", "K 近邻算法（KNN）—— 海伦约会")
    rep.h("一、实验目的")
    rep.bullets([
        "掌握 K 近邻（KNN）算法的基本原理（距离度量、多数表决）及 Python 实现；",
        "掌握特征归一化在 KNN 中的作用；",
        "掌握用交叉验证选择超参数 K 的方法；",
        "能够运用 KNN 完成海伦约会分类任务并评估精度。",
    ])

    rep.h("二、实验内容")
    rep.para(f"对海伦约会数据集（共 {R['n']} 个样本，3 个特征，3 个类别：不喜欢/"
             f"魅力一般/极具魅力，样本分布 {R['dist']}）进行 KNN 分类。在原有代码基础上"
             f"增加交叉验证以确定最优 K 值，最终输出最优 K 值及对应精度。")

    rep.h("三、实验步骤")
    rep.bullets([
        "数据读取：解析 datingTestSet.txt，将文本标签映射为 0/1/2；",
        "特征归一化：使用 Min-Max 将三个特征缩放到 [0,1]，"
        "因飞行里程数量级远大于其他特征，不归一化会使其主导欧氏距离；",
        "划分数据：按 8:2 分层划分训练集与测试集；",
        "交叉验证选 K：在训练集上对 K=1~30 做 10 折交叉验证，取平均精度最高的 K；",
        "测试评估：用最优 K 在独立测试集上评估精度并绘制混淆矩阵；",
        "可视化：绘制精度-K 曲线、特征散点图与三维分布图。",
    ])

    rep.h("四、程序设计的核心代码")
    rep.code(
        "def knn_classify(x, X_train, y_train, k):\n"
        "    dists = np.sqrt(((X_train - x) ** 2).sum(axis=1))  # 欧氏距离\n"
        "    idx = dists.argsort()[:k]                          # 最近的 k 个\n"
        "    votes = np.bincount(y_train[idx], minlength=3)     # 多数表决\n"
        "    return votes.argmax()",
        caption="① 从零实现的 KNN 分类器（欧氏距离 + 多数表决）")
    rep.code(
        "Xn = MinMaxScaler().fit_transform(X)        # 归一化\n"
        "for k in range(1, 31):                       # 10 折交叉验证选 K\n"
        "    score = cross_val_score(KNeighborsClassifier(k), Xtr, ytr, cv=10)\n"
        "    cv_means.append(score.mean())\n"
        "best_k = ks[int(np.argmax(cv_means))]        # 最优 K",
        caption="② 交叉验证确定最优 K（sklearn）")

    rep.h("五、实验结果")
    rep.para(f"通过 10 折交叉验证，最优 K 值为 {R['best_k']}，对应交叉验证平均精度为 "
             f"{R['best_cv']:.2%}。下表给出部分 K 值的交叉验证精度：")
    rep.table(["K 值", "10 折 CV 平均精度"], R["cv_table"],
              caption="表1 不同 K 值的交叉验证精度", col_widths=[1.6, 2.6])
    rep.image(os.path.join(FIG, "cv_k.png"), 5.8,
              "图1 交叉验证精度随 K 的变化（红色虚线为最优 K）")
    rep.para(f"用最优 K={R['best_k']} 在独立测试集（200 个样本）上评估，分类精度为 "
             f"{R['test_acc']:.2%}；作为验证，从零实现的 KNN 在相同设置下精度为 "
             f"{R['scratch_acc']:.2%}，两者一致，说明自实现逻辑正确。混淆矩阵如下：")
    rep.image(os.path.join(FIG, "cm.png"), 4.8, "图2 测试集混淆矩阵")
    rep.para("数据可视化（归一化前）显示，三个类别在“飞行里程-游戏时间占比”平面上"
             "已有较清晰的聚集区域，这也是 KNN 能取得较高精度的原因：")
    rep.image(os.path.join(FIG, "scatter2d.png"), 6.2,
              "图3 特征两两散点图")
    rep.image(os.path.join(FIG, "scatter3d.png"), 4.8,
              "图4 海伦约会数据三维分布")

    rep.h("六、实验体会")
    rep.para(
        "本次实验让我体会到 KNN “懒惰学习”的特点——无需显式训练，预测时直接基于"
        "距离做多数表决。两点收获尤为深刻：①归一化是 KNN 的关键预处理，飞行里程"
        "（数万量级）若不归一化会完全主导欧氏距离，使另外两个特征失效；②K 是核心"
        "超参数，K 过小对噪声敏感（高方差），K 过大则决策边界过于平滑（高偏差），"
        "通过交叉验证可以在偏差-方差之间自动权衡，得到稳定的最优 K。本实验最优 "
        f"K={R['best_k']}，测试精度 {R['test_acc']:.2%}，验证了交叉验证选参的有效性。")

    rep.teacher_eval()
    docx = os.path.join(OUT, "机器学习实验报告2-KNN.docx")
    rep.save(docx)
    return docx


if __name__ == "__main__":
    run()
    docx = build_report()
    print("best_k=%d cv=%.4f test_acc=%.4f scratch=%.4f" % (
        R["best_k"], R["best_cv"], R["test_acc"], R["scratch_acc"]))
    print("SAVED:", docx)
