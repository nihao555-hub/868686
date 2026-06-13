# -*- coding: utf-8 -*-
"""实验报告7：支持向量机（SVM）

基本要求：
  1. 简述练习2（自实现 SMO）与练习3（sklearn SVC）算法实现的区别和联系，
     并分别总结各自涉及的参数，以表格形式展示。
  2. 对不同核函数、不同参数下的实验结果进行对比分析，展示 SVM 在多分类任务上的
     可视化结果（决策边界、背景色、支持向量、参数值）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from os import listdir
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import labcommon as C

LAB = os.path.dirname(C.find_file("Lab7.pdf"))
OUT = os.path.join(LAB, "实验报告7-SVM")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}


# ----------------------------------------------------------------------------
# 练习3：手写数字识别（sklearn SVC）
# ----------------------------------------------------------------------------
def img2vector(filename):
    vect = np.zeros(1024)
    with open(filename) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                vect[32 * i + j] = int(line[j])
    return vect


def load_digits_txt(folder):
    files = listdir(folder)
    X = np.zeros((len(files), 1024))
    y = np.zeros(len(files), dtype=int)
    for i, fn in enumerate(files):
        y[i] = int(fn.split("_")[0])
        X[i] = img2vector(os.path.join(folder, fn))
    return X, y


def digit_recognition():
    base = os.path.dirname(C.find_file("Lab7.pdf"))
    train_dir = os.path.join(base, "练习3", "trainingDigits")
    test_dir = os.path.join(base, "练习3", "testDigits")
    Xtr, ytr = load_digits_txt(train_dir)
    Xte, yte = load_digits_txt(test_dir)
    R["digit_ntr"], R["digit_nte"] = len(ytr), len(yte)
    rows = []
    best = (None, -1)
    for name, kw in [("线性核 linear", dict(kernel="linear", C=200)),
                     ("多项式核 poly(d=3)", dict(kernel="poly", degree=3, C=200)),
                     ("高斯核 rbf(默认γ)", dict(kernel="rbf", C=200, gamma="scale")),
                     ("高斯核 rbf(γ=0.001)", dict(kernel="rbf", C=200, gamma=0.001))]:
        clf = SVC(**kw).fit(Xtr, ytr)
        acc = clf.score(Xte, yte)
        rows.append([name, f"{int(clf.n_support_.sum())}", f"{acc:.2%}"])
        if acc > best[1]:
            best = (name, acc)
    R["digit_rows"] = rows
    R["digit_best"] = best


# ----------------------------------------------------------------------------
# 可视化工具
# ----------------------------------------------------------------------------
def plot_decision(ax, clf, X, y, title, multiclass=False):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    if multiclass:
        cmap_bg = ListedColormap(["#FFDDDD", "#DDFFDD", "#DDDDFF"])
        cmap_pt = ListedColormap(["#d62728", "#2ca02c", "#1f77b4"])
        ax.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_bg)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pt, s=20, edgecolors="k",
                   linewidths=0.3)
    else:
        ax.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22, edgecolors="k",
                   linewidths=0.3)
        # 决策边界 / 间隔
        try:
            dz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            ax.contour(xx, yy, dz, levels=[-1, 0, 1], colors="k",
                       linestyles=["--", "-", "--"], linewidths=[0.8, 1.3, 0.8])
        except Exception:
            pass
    # 支持向量
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors="k",
               linewidths=1.1, label="支持向量")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def binary_kernel_compare():
    df = pd.read_csv(C.find_file("svmdata2.csv"))
    X = StandardScaler().fit_transform(df[["X1", "X2"]].values)
    y = df["y"].values
    configs = [
        ("线性核\nlinear, C=1", dict(kernel="linear", C=1)),
        ("多项式核\npoly(d=3), C=1", dict(kernel="poly", degree=3, C=1)),
        ("高斯核\nrbf, C=1, γ=scale", dict(kernel="rbf", C=1, gamma="scale")),
        ("高斯核\nrbf, C=100, γ=10", dict(kernel="rbf", C=100, gamma=10)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0))
    rows = []
    for ax, (title, kw) in zip(axes, configs):
        clf = SVC(**kw).fit(X, y)
        acc = clf.score(X, y)
        plot_decision(ax, clf, X, y, f"{title}\n精度={acc:.2%}, SV={clf.n_support_.sum()}")
        rows.append([title.replace("\n", " "), f"{clf.n_support_.sum()}", f"{acc:.2%}"])
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("二分类（非线性数据 svmdata2）不同核函数/参数的决策边界对比", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "binary_kernels.png"), bbox_inches="tight")
    plt.close()
    R["binary_rows"] = rows


def rbf_param_grid():
    df = pd.read_csv(C.find_file("svmdata2.csv"))
    X = StandardScaler().fit_transform(df[["X1", "X2"]].values)
    y = df["y"].values
    Cs = [0.1, 1, 100]
    gammas = [0.1, 1, 10]
    fig, axes = plt.subplots(len(Cs), len(gammas), figsize=(12, 11))
    for i, Cv in enumerate(Cs):
        for j, g in enumerate(gammas):
            clf = SVC(kernel="rbf", C=Cv, gamma=g).fit(X, y)
            plot_decision(axes[i, j], clf, X, y,
                          f"C={Cv}, γ={g}\n精度={clf.score(X, y):.2%}, "
                          f"SV={clf.n_support_.sum()}")
    fig.suptitle("高斯核 (RBF) 不同惩罚参数 C 与核宽参数 γ 的影响", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "rbf_grid.png"), bbox_inches="tight")
    plt.close()


def multiclass_iris():
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data[:, 2:4])
    y = iris.target
    configs = [
        ("线性核 linear, C=1", dict(kernel="linear", C=1)),
        ("多项式核 poly(d=3), C=1", dict(kernel="poly", degree=3, C=1)),
        ("高斯核 rbf, C=1, γ=scale", dict(kernel="rbf", C=1, gamma="scale")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    rows = []
    for ax, (title, kw) in zip(axes, configs):
        clf = SVC(**kw).fit(X, y)
        acc = clf.score(X, y)
        plot_decision(ax, clf, X, y, f"{title}\n精度={acc:.2%}, SV总数={clf.n_support_.sum()}",
                      multiclass=True)
        rows.append([title, f"{clf.n_support_.sum()}", f"{acc:.2%}"])
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("SVM 多分类（Iris 花瓣特征）不同核函数的决策区域、背景色与支持向量",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "multiclass_iris.png"), bbox_inches="tight")
    plt.close()
    R["iris_rows"] = rows


def build_report():
    rep = C.Report("报告7", "支持向量机（SVM）")
    rep.h("一、实验目的")
    rep.bullets([
        "掌握支持向量机（SVM）的基本原理（最大间隔、对偶问题、核技巧、软间隔）及实现；",
        "理解 SMO 算法求解 SVM 对偶问题的过程；",
        "掌握不同核函数与参数对 SVM 分类性能与决策边界的影响；",
        "能够运用 SVM 完成二分类与多分类任务并进行可视化分析。",
    ])

    rep.h("二、实验内容")
    rep.bullets([
        "对比练习2（自实现 SMO 训练 SVM）与练习3（基于 sklearn SVC）的实现区别与联系，"
        "并以表格总结各自涉及的参数；",
        "在手写数字识别任务上用不同核函数训练 SVC 并对比精度；",
        "在二分类（svmdata2）与多分类（Iris）数据上，对不同核函数、不同参数（C、γ、"
        "多项式阶数）的决策边界、背景色与支持向量进行可视化对比。",
    ])

    rep.h("三、实验步骤")
    rep.bullets([
        "阅读练习2/练习3 源码，分析两种实现的算法流程与参数；",
        "手写数字识别：将 32×32 文本图像转为 1024 维向量，训练 SVC 并评估不同核函数；",
        "二分类可视化：标准化 svmdata2，训练不同核/参数的 SVC，绘制决策边界与支持向量；",
        "RBF 参数研究：在 C×γ 网格上训练 RBF-SVM，观察过拟合/欠拟合；",
        "多分类可视化：在 Iris 上训练多分类 SVM，绘制决策区域背景色与支持向量。",
    ])

    rep.h("四、练习2 与练习3 的区别和联系（要求1）")
    rep.h2("4.1 联系")
    rep.para("两者求解的都是同一个 SVM 模型：在软间隔意义下最大化分类间隔，最终决策函数"
             "均为 f(x)=sign(Σ αᵢ yᵢ K(xᵢ,x)+b)，都依赖核函数把数据映射到高维空间，"
             "并且最优解都只与少数支持向量有关。")
    rep.h2("4.2 区别")
    rep.table(["对比项", "练习2：自实现 SMO", "练习3：sklearn SVC"], [
        ["实现方式", "纯 NumPy 手写完整 Platt SMO", "调用 libsvm 封装的 SVC"],
        ["求解算法", "SMO：每次选两个 αᵢ 解析优化", "libsvm 优化的 SMO（C/C++）"],
        ["核函数", "手写 linear / rbf 核矩阵", "linear/poly/rbf/sigmoid 等内置"],
        ["分类能力", "二分类（多分类需 1-vs-rest 组合）", "原生支持多分类(OvO)"],
        ["代码量/速度", "代码量大、Python 实现较慢", "几行代码、底层 C 实现快"],
        ["用途", "理解 SVM 与 SMO 原理", "工程应用、快速建模"],
    ], caption="表1 练习2 与练习3 实现对比", col_widths=[1.4, 2.7, 2.7])

    rep.h2("4.3 各自涉及的参数")
    rep.table(["参数", "含义", "出现位置"], [
        ["C", "软间隔惩罚系数，越大越不容忍误分类", "练习2 / 练习3"],
        ["toler", "KKT 条件容错率（迭代终止判据）", "练习2"],
        ["maxIter", "SMO 最大迭代轮数", "练习2"],
        ["kTup/kernel", "核函数类型（lin / rbf 等）", "练习2 / 练习3"],
        ["σ / gamma", "RBF 核宽参数，控制单样本影响范围", "练习2(σ) / 练习3(gamma)"],
        ["degree", "多项式核阶数", "练习3"],
        ["alphas, b", "对偶变量与偏置（训练得到的模型参数）", "练习2 / 练习3"],
    ], caption="表2 SVM/SMO 涉及的主要参数", col_widths=[1.6, 3.2, 2.0])

    rep.h("五、不同核函数与参数的对比分析（要求2）")
    rep.h2("5.1 手写数字识别（不同核函数）")
    rep.para(f"将练习3 的手写数字数据（训练 {R['digit_ntr']} 张、测试 "
             f"{R['digit_nte']} 张，10 类）转为 1024 维向量，用不同核函数训练 SVC，"
             f"测试精度如下表。其中 {R['digit_best'][0]} 表现最佳，精度 "
             f"{R['digit_best'][1]:.2%}（注：RBF 核宽 γ 取值过大或过小都会显著降低精度，"
             f"说明参数选择的重要性）。")
    rep.table(["核函数 / 参数", "支持向量数", "测试精度"], R["digit_rows"],
              caption="表3 不同核函数下手写数字识别精度", col_widths=[2.8, 1.6, 1.6])

    rep.h2("5.2 二分类：不同核函数的决策边界")
    rep.para("在非线性可分数据 svmdata2 上对比不同核函数的决策边界（实线为决策边界，"
             "虚线为间隔边界，空心圈为支持向量）：线性核无法刻画非线性边界，"
             "多项式核与高斯核能形成弯曲边界；当 rbf 的 C、γ 过大时边界过度贴合训练点，"
             "出现明显过拟合。")
    rep.image(os.path.join(FIG, "binary_kernels.png"), 6.8,
              "图1 不同核函数/参数的二分类决策边界与支持向量")
    rep.table(["核函数 / 参数", "支持向量数", "训练精度"], R["binary_rows"],
              caption="表4 不同核函数二分类结果", col_widths=[2.8, 1.6, 1.6])

    rep.h2("5.3 RBF 核的参数 C 与 γ 的影响")
    rep.para("固定 RBF 核，在 C×γ 网格上观察决策边界变化：γ 越大，单个样本影响范围越小，"
             "边界越“曲折”、越容易过拟合；C 越大，对误分类惩罚越重，间隔越窄、支持向量越少。"
             "二者共同决定模型复杂度，需通过交叉验证选取。")
    rep.image(os.path.join(FIG, "rbf_grid.png"), 6.2,
              "图2 RBF 核惩罚参数 C 与核宽 γ 的影响（3×3 网格）")

    rep.h2("5.4 多分类可视化（Iris）")
    rep.para("SVM 通过一对一(OvO)策略扩展到多分类。下图展示在 Iris 花瓣特征上，"
             "不同核函数的决策区域（背景色）、样本点与支持向量（空心圈），并标注核函数、"
             "参数与精度：")
    rep.image(os.path.join(FIG, "multiclass_iris.png"), 6.8,
              "图3 SVM 多分类决策区域、背景色与支持向量")
    rep.table(["核函数 / 参数", "支持向量总数", "精度"], R["iris_rows"],
              caption="表5 Iris 多分类不同核函数结果", col_widths=[2.8, 1.8, 1.4])

    rep.h("六、实验体会")
    rep.para(
        "本次实验我从原理与实践两方面深入理解了 SVM。通过对比自实现 SMO（练习2）与 "
        "sklearn SVC（练习3），认识到二者求解的是同一最大间隔模型，区别在于工程封装："
        "手写 SMO 帮助我理解了“每次挑选两个拉格朗日乘子 α 解析优化、用容错率判断 KKT "
        "条件、最终模型只由支持向量决定”的过程；而 sklearn 基于 libsvm 用几行代码即可"
        "完成训练并原生支持多分类。在核函数与参数实验中，我直观地看到：①核函数决定了"
        "决策边界的形状，线性核只能画直线，poly/rbf 才能拟合非线性边界；②惩罚参数 C "
        "控制对误分类的容忍度，③RBF 的 γ 控制核宽，γ 过大会使边界过度贴合训练样本而"
        "过拟合。这些参数共同决定模型复杂度，需要借助交叉验证在偏差与方差之间取得平衡。"
        "支持向量的可视化也让我理解了 SVM “稀疏解”的本质——决策边界只由少数关键样本支撑。")

    rep.teacher_eval()
    docx = os.path.join(OUT, "机器学习实验报告7-SVM.docx")
    rep.save(docx)
    return docx


if __name__ == "__main__":
    digit_recognition()
    binary_kernel_compare()
    rbf_param_grid()
    multiclass_iris()
    docx = build_report()
    print("digit_best=%s(%.4f) ntr=%d nte=%d" % (
        R["digit_best"][0], R["digit_best"][1], R["digit_ntr"], R["digit_nte"]))
    print("SAVED:", docx)
