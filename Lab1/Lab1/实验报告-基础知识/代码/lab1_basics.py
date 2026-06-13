# -*- coding: utf-8 -*-
"""实验1：基础知识与 Scikit-learn（破冰练习）

实验目标：安装并熟悉 Python 环境，完成 Python / NumPy / Pandas /
数据可视化 / SciPy / Scikit-learn 的破冰练习。
本脚本以可运行的方式演示各模块的核心用法，并产出可视化结果。
"""
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, linalg
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import labcommon as C

LAB = os.path.dirname(C.find_file("Lab1.pdf"))
OUT = os.path.join(LAB, "实验报告-基础知识")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}


def numpy_demo():
    a = np.arange(1, 13).reshape(3, 4)
    R["np_sum_axis0"] = a.sum(axis=0).tolist()
    R["np_mean"] = float(a.mean())
    # 广播
    b = a * np.array([1, 10, 100, 1000])
    R["np_broadcast_row0"] = b[0].tolist()
    # 线性代数：解 Ax=b
    A = np.array([[3.0, 2.0], [1.0, 2.0]])
    rhs = np.array([12.0, 8.0])
    x = linalg.solve(A, rhs)
    R["linalg_solve"] = [round(v, 3) for v in x]
    # 随机数与统计
    rng = np.random.default_rng(0)
    s = rng.normal(50, 10, 1000)
    R["np_describe"] = (round(s.mean(), 2), round(s.std(), 2))


def pandas_demo():
    csv = ("name,math,physics,chinese\n"
           "Alice,92,88,79\nBob,75,80,95\nCindy,88,91,70\n"
           "David,60,55,85\nEve,99,97,90\n")
    df = pd.read_csv(io.StringIO(csv))
    df["total"] = df[["math", "physics", "chinese"]].sum(axis=1)
    df["rank"] = df["total"].rank(ascending=False).astype(int)
    R["pd_head"] = df.to_string(index=False)
    R["pd_mean"] = df[["math", "physics", "chinese"]].mean().round(1).to_dict()
    return df


def viz_demo(df):
    # 图1：matplotlib 多种图表
    x = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(x, np.sin(x), label="sin"); axes[0, 0].plot(x, np.cos(x), label="cos")
    axes[0, 0].set_title("折线图：正弦/余弦"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    rng = np.random.default_rng(1)
    axes[0, 1].scatter(rng.normal(size=120), rng.normal(size=120),
                       c=rng.random(120), cmap="viridis", alpha=0.7)
    axes[0, 1].set_title("散点图：二维随机点")

    axes[1, 0].hist(rng.normal(50, 10, 1000), bins=30, color="#3498db",
                    edgecolor="white")
    axes[1, 0].set_title("直方图：正态分布采样")

    axes[1, 1].bar(df["name"], df["total"], color="#e67e22")
    axes[1, 1].set_title("柱状图：学生总分"); axes[1, 1].set_ylabel("总分")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "matplotlib_basics.png"), bbox_inches="tight")
    plt.close()

    # 图2：iris 相关性热图 + 箱线图
    iris = load_iris()
    dfi = pd.DataFrame(iris.data, columns=["花萼长", "花萼宽", "花瓣长", "花瓣宽"])
    corr = dfi.corr()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    im = axes[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(dfi.columns, rotation=45); axes[0].set_yticklabels(dfi.columns)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                         color="black", fontsize=8)
    axes[0].set_title("Iris 特征相关性热图")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    dfi["类别"] = iris.target
    for k in range(3):
        axes[1].scatter(dfi[dfi.类别 == k]["花瓣长"], dfi[dfi.类别 == k]["花瓣宽"],
                        label=iris.target_names[k], alpha=0.7)
    axes[1].set_xlabel("花瓣长"); axes[1].set_ylabel("花瓣宽")
    axes[1].set_title("Iris 花瓣特征散点"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pandas_seaborn.png"), bbox_inches="tight")
    plt.close()


def sklearn_demo():
    iris = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.3,
                                          random_state=42, stratify=iris.target)
    clf = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    R["sk_acc"] = acc
    cm = confusion_matrix(yte, clf.predict(Xte))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(iris.target_names, rotation=30, fontsize=8)
    ax.set_yticklabels(iris.target_names, fontsize=8)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    ax.set_title(f"Iris KNN 分类混淆矩阵 (精度={acc:.2%})")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "sklearn_iris.png"), bbox_inches="tight")
    plt.close()


def build_report():
    rep = C.Report("1", "基础知识与 Scikit-learn（破冰练习）")
    rep.h("一、实验目的")
    rep.bullets([
        "安装 Python 编辑器，熟悉 Python 开发环境；",
        "完成 Python 破冰练习，掌握 Python 基础语法；",
        "掌握 NumPy 数值计算、Pandas 数据处理、Matplotlib/Seaborn 可视化、"
        "SciPy 科学计算的基本用法；",
        "了解机器学习库 Scikit-learn 的基本使用流程。",
    ])

    rep.h("二、实验内容")
    rep.bullets([
        "Python 基础：变量、容器、控制流、函数等破冰练习；",
        "NumPy：数组创建与运算、广播、线性代数、随机数与统计；",
        "Pandas：DataFrame 的构建、统计、排序与派生列；",
        "数据可视化：折线图、散点图、直方图、柱状图、相关性热图；",
        "Scikit-learn：以 Iris 数据集为例完成“加载-划分-训练-评估”全流程。",
    ])

    rep.h("三、实验步骤")
    rep.bullets([
        "搭建环境：安装 Python 3.12 及 NumPy/Pandas/Matplotlib/SciPy/scikit-learn；",
        "NumPy 练习：创建 3×4 数组并做按轴求和、广播运算与线性方程组求解；",
        "Pandas 练习：构造学生成绩表，计算总分、均值、排名；",
        "可视化练习：绘制 4 类基础图表及 Iris 特征相关性热图、散点图；",
        "Scikit-learn 练习：在 Iris 上训练 KNN 分类器并输出精度与混淆矩阵。",
    ])

    rep.h("四、程序设计的核心代码")
    rep.code(
        "a = np.arange(1, 13).reshape(3, 4)        # 3x4 数组\n"
        "a.sum(axis=0); a.mean()                    # 按列求和 / 全局均值\n"
        "b = a * np.array([1, 10, 100, 1000])       # 广播\n"
        "x = scipy.linalg.solve(A, rhs)             # 解线性方程组 Ax=b",
        caption="① NumPy / SciPy 数值计算")
    rep.code(
        "df = pd.read_csv(...)                       # 读取数据\n"
        "df['total'] = df[['math','physics','chinese']].sum(axis=1)\n"
        "df['rank']  = df['total'].rank(ascending=False).astype(int)",
        caption="② Pandas 数据处理")
    rep.code(
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y)\n"
        "clf = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)\n"
        "acc = accuracy_score(yte, clf.predict(Xte))",
        caption="③ Scikit-learn 机器学习流程")

    rep.h("五、实验结果")
    rep.h2("5.1 NumPy / SciPy")
    rep.bullets([
        f"3×4 数组按列求和：{R['np_sum_axis0']}，全局均值：{R['np_mean']}；",
        f"广播运算第 0 行结果：{R['np_broadcast_row0']}；",
        f"线性方程组 3x+2y=12, x+2y=8 的解：x={R['linalg_solve']}；",
        f"1000 个正态采样的均值/标准差约为：{R['np_describe']}。",
    ])
    rep.h2("5.2 Pandas")
    rep.para("学生成绩表（含总分与排名）：")
    rep.code(R["pd_head"], caption="DataFrame 输出")
    rep.para(f"各科平均分：{R['pd_mean']}。")
    rep.h2("5.3 数据可视化")
    rep.image(os.path.join(FIG, "matplotlib_basics.png"), 6.4,
              "图1 Matplotlib 基础图表：折线/散点/直方图/柱状图")
    rep.image(os.path.join(FIG, "pandas_seaborn.png"), 6.4,
              "图2 Iris 特征相关性热图与花瓣散点图")
    rep.h2("5.4 Scikit-learn")
    rep.para(f"在 Iris 数据集上用 KNN(k=5) 进行三分类，测试集精度为 {R['sk_acc']:.2%}，"
             f"混淆矩阵如下：")
    rep.image(os.path.join(FIG, "sklearn_iris.png"), 4.4,
              "图3 Iris KNN 分类混淆矩阵")

    rep.h("六、实验体会")
    rep.para(
        "通过破冰练习，我完成了 Python 科学计算生态的入门。NumPy 的向量化与广播"
        "机制让数组运算既简洁又高效，避免了显式循环；Pandas 的 DataFrame 提供了"
        "类 SQL/Excel 的数据处理能力，groupby、rank、缺失值处理等在后续实验中频繁用到；"
        "Matplotlib/Seaborn 让数据“可见”，是理解数据分布和模型结果的关键工具；"
        "SciPy 补充了线性代数、统计、优化等科学计算能力。最后通过 Scikit-learn 体验了"
        "机器学习“加载数据→划分→训练→评估”的标准流程，统一的 fit/predict/score 接口"
        "为后续各实验打下了基础。")

    rep.teacher_eval()
    docx = os.path.join(OUT, "机器学习实验报告-基础知识.docx")
    rep.save(docx)
    return docx


if __name__ == "__main__":
    numpy_demo()
    df = pandas_demo()
    viz_demo(df)
    sklearn_demo()
    docx = build_report()
    print("sk_acc=%.4f" % R["sk_acc"])
    print("SAVED:", docx)
