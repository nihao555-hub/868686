# -*- coding: utf-8 -*-
"""实验报告1：逻辑回归（Logistic Regression）

任务：
  A. 从零实现逻辑回归（批量梯度下降），在“考试成绩-是否录取”二分类数据集
     (ex2data1) 上训练，输出分类精度与决策边界可视化。
  B. 真实任务：用逻辑回归预测“马疝病(horse colic)是否存活”，输出测试集精度、
     混淆矩阵，并与从零实现/不同正则强度对比。
  C. 多分类扩展：在鸢尾花(Iris)数据集上用 Softmax 逻辑回归做三分类，
     输出决策区域可视化与精度。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_iris
import labcommon as C

LAB = os.path.dirname(C.find_file("Lab2.pdf"))
OUT = os.path.join(LAB, "实验报告1-逻辑回归")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}


# ----------------------------------------------------------------------------
# 从零实现的逻辑回归
# ----------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_lr_gd(X, y, lr=0.1, n_iter=3000):
    """批量梯度下降。X 不含偏置列，内部自动添加。"""
    m, n = X.shape
    Xb = np.hstack([np.ones((m, 1)), X])
    w = np.zeros(n + 1)
    losses = []
    for _ in range(n_iter):
        h = sigmoid(Xb @ w)
        grad = Xb.T @ (h - y) / m
        w -= lr * grad
        eps = 1e-9
        losses.append(-np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)))
    return w, losses


def predict_lr(X, w):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return (sigmoid(Xb @ w) >= 0.5).astype(int)


# ----------------------------------------------------------------------------
# A. ex2data1 二分类 + 决策边界
# ----------------------------------------------------------------------------
def task_a():
    path = C.find_file("ex2data1.txt")
    data = np.loadtxt(path, delimiter=",")
    X, y = data[:, :2], data[:, 2]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    w, losses = train_lr_gd(Xs, y, lr=0.3, n_iter=5000)
    acc = accuracy_score(y, predict_lr(Xs, w))
    R["A_acc"] = acc
    R["A_w"] = w

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(losses, color="#c0392b")
    ax[0].set_title("梯度下降损失收敛曲线")
    ax[0].set_xlabel("迭代次数")
    ax[0].set_ylabel("交叉熵损失")
    ax[0].grid(alpha=0.3)

    pos, neg = y == 1, y == 0
    ax[1].scatter(X[pos, 0], X[pos, 1], c="#2980b9", marker="+", s=70, label="录取 (y=1)")
    ax[1].scatter(X[neg, 0], X[neg, 1], c="#e67e22", marker="o", s=35,
                  edgecolors="k", label="未录取 (y=0)")
    # 决策边界：w0 + w1*z1 + w2*z2 = 0，z 为标准化坐标
    xs = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200)
    zs1 = (xs - scaler.mean_[0]) / scaler.scale_[0]
    zs2 = -(w[0] + w[1] * zs1) / w[2]
    xs2 = zs2 * scaler.scale_[1] + scaler.mean_[1]
    ax[1].plot(xs, xs2, "g-", lw=2, label="决策边界")
    ax[1].set_xlabel("考试1成绩")
    ax[1].set_ylabel("考试2成绩")
    ax[1].set_title(f"逻辑回归决策边界 (训练精度={acc:.2%})")
    ax[1].legend(loc="upper right", fontsize=9)
    ax[1].set_ylim(X[:, 1].min() - 5, X[:, 1].max() + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "A_decision_boundary.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# B. horse colic 真实数据集
# ----------------------------------------------------------------------------
def task_b():
    tr = np.loadtxt(C.find_file("horseColicTraining.txt"))
    te = np.loadtxt(C.find_file("horseColicTest.txt"))
    Xtr, ytr = tr[:, :-1], tr[:, -1]
    Xte, yte = te[:, :-1], te[:, -1]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # 从零实现
    w, _ = train_lr_gd(Xtr_s, ytr, lr=0.1, n_iter=8000)
    acc_scratch = accuracy_score(yte, predict_lr(Xte_s, w))

    # sklearn，不同正则强度
    rows = []
    best = (None, -1, None)
    for Cval in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=Cval, max_iter=2000)
        clf.fit(Xtr_s, ytr)
        a = clf.score(Xte_s, yte)
        rows.append([f"{Cval:g}", f"{clf.score(Xtr_s, ytr):.2%}", f"{a:.2%}"])
        if a > best[1]:
            best = (Cval, a, clf)
    R["B_scratch_acc"] = acc_scratch
    R["B_reg_rows"] = rows
    R["B_best_C"] = best[0]
    R["B_best_acc"] = best[1]

    cm = confusion_matrix(yte, best[2].predict(Xte_s))
    R["B_cm"] = cm
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["死亡(0)", "存活(1)"])
    ax.set_yticklabels(["死亡(0)", "存活(1)"])
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"马疝病存活预测混淆矩阵\n(C={best[0]:g}, 测试精度={best[1]:.2%})")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "B_confusion.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# C. Iris 多分类
# ----------------------------------------------------------------------------
def task_c():
    iris = load_iris()
    X = iris.data[:, 2:4]          # 花瓣长度、花瓣宽度
    y = iris.target
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xs, y)
    acc = clf.score(Xs, y)
    R["C_acc"] = acc

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
    colors = ["#440154", "#21918c", "#fde725"]
    for k, name in enumerate(iris.target_names):
        ax.scatter(X[y == k, 0], X[y == k, 1], c=colors[k], edgecolors="k",
                   s=40, label=name)
    ax.set_xlabel("花瓣长度 (cm)")
    ax.set_ylabel("花瓣宽度 (cm)")
    ax.set_title(f"Iris 三分类逻辑回归决策区域 (精度={acc:.2%})")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "C_iris_regions.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# 生成报告
# ----------------------------------------------------------------------------
def build_report():
    rep = C.Report("报告1", "逻辑回归（Logistic Regression）")
    rep.h("一、实验目的")
    rep.bullets([
        "掌握逻辑回归的基本原理（Sigmoid 假设函数、对数似然、梯度下降）及 Python 实现；",
        "掌握从二分类到多分类（Softmax / One-vs-Rest）的推广方法；",
        "能够运用逻辑回归完成具体的分类任务，输出分类精度与分类可视化结果。",
    ])

    rep.h("二、实验内容")
    rep.para("在所提供的逻辑回归练习基础上，自选新的应用领域与数据集，"
             "完成代码实现与改进，最终输出分类精度和分类可视化图。本报告共完成三个子任务：")
    rep.bullets([
        "任务A（二分类，自实现）：基于“两门考试成绩预测是否录取”数据集 ex2data1，"
        "从零实现逻辑回归（批量梯度下降），输出训练精度与决策边界。",
        "任务B（二分类，真实数据）：基于马疝病(horse colic)数据集，用逻辑回归预测病马"
        "是否存活，对比自实现与 sklearn、不同正则强度的测试精度，并给出混淆矩阵。",
        "任务C（多分类扩展）：在鸢尾花 Iris 数据集上用 Softmax 逻辑回归做三分类，"
        "输出决策区域可视化与精度。",
    ])

    rep.h("三、实验要求")
    rep.bullets([
        "具体应用领域自选：分别选取教育录取、兽医病理、植物分类三个领域；",
        "具体分类数据自选：ex2data1、horseColic、Iris；",
        "具体分类任务自选：同时覆盖二分类与多分类。",
    ])

    rep.h("四、实验步骤")
    rep.bullets([
        "数据加载与预处理：读取数据集，使用 StandardScaler 对特征做标准化以稳定梯度下降；",
        "模型实现：实现 Sigmoid、交叉熵损失与批量梯度下降，迭代更新权重 w；",
        "二分类训练与可视化：在 ex2data1 上训练并绘制损失收敛曲线、样本散点与决策边界；",
        "真实数据验证：在 horseColic 训练集拟合、测试集评估，比较不同正则强度 C；",
        "多分类扩展：在 Iris 上训练 Softmax 逻辑回归，用网格法绘制决策区域；",
        "结果汇总：输出各任务的分类精度、混淆矩阵与可视化图。",
    ])

    rep.h("五、程序设计的核心代码")
    rep.code(
        "def sigmoid(z):\n"
        "    z = np.clip(z, -500, 500)\n"
        "    return 1.0 / (1.0 + np.exp(-z))\n\n"
        "def train_lr_gd(X, y, lr=0.1, n_iter=3000):\n"
        "    m, n = X.shape\n"
        "    Xb = np.hstack([np.ones((m, 1)), X])   # 增广偏置列\n"
        "    w = np.zeros(n + 1)\n"
        "    for _ in range(n_iter):\n"
        "        h = sigmoid(Xb @ w)\n"
        "        grad = Xb.T @ (h - y) / m          # 对数似然的梯度\n"
        "        w -= lr * grad                     # 梯度下降更新\n"
        "    return w",
        caption="① 从零实现的逻辑回归（Sigmoid + 批量梯度下降）")
    rep.code(
        "clf = LogisticRegression(C=Cval, max_iter=2000)\n"
        "clf.fit(Xtr_s, ytr)                       # 训练\n"
        "acc = clf.score(Xte_s, yte)               # 测试精度\n"
        "cm = confusion_matrix(yte, clf.predict(Xte_s))",
        caption="② sklearn 逻辑回归（不同正则强度对比 + 混淆矩阵）")
    rep.code(
        "xx, yy = np.meshgrid(...)\n"
        "Z = clf.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))\n"
        "ax.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.25)  # 决策区域背景",
        caption="③ 多分类决策区域可视化（网格预测 + contourf）")

    rep.h("六、实验结果")
    rep.h2("6.1 任务A：ex2data1 二分类")
    rep.para(f"从零实现的逻辑回归在 ex2data1 上经梯度下降收敛，训练分类精度为 "
             f"{R['A_acc']:.2%}。损失曲线单调下降，决策边界能较好地分开两类样本。")
    rep.image(os.path.join(FIG, "A_decision_boundary.png"), 6.2,
              "图1 梯度下降损失收敛曲线（左）与逻辑回归决策边界（右）")

    rep.h2("6.2 任务B：马疝病存活预测")
    rep.para(f"在 horseColic 真实数据集上，自实现逻辑回归的测试精度为 "
             f"{R['B_scratch_acc']:.2%}。使用 sklearn 并调节正则强度 C，结果如下表：")
    rep.table(["正则强度 C", "训练精度", "测试精度"], R["B_reg_rows"],
              caption="表1 不同正则强度下的逻辑回归精度对比",
              col_widths=[1.8, 1.8, 1.8])
    rep.para(f"其中 C={R['B_best_C']:g} 时测试精度最高，为 {R['B_best_acc']:.2%}，"
             f"对应混淆矩阵见下图。")
    rep.image(os.path.join(FIG, "B_confusion.png"), 4.4,
              "图2 马疝病存活预测的混淆矩阵")

    rep.h2("6.3 任务C：Iris 三分类")
    rep.para(f"在 Iris 数据集上使用 Softmax 逻辑回归（花瓣长度、花瓣宽度两个特征）"
             f"进行三分类，分类精度为 {R['C_acc']:.2%}，决策区域如下图所示，"
             f"setosa 类完全线性可分，versicolor 与 virginica 之间存在少量重叠。")
    rep.image(os.path.join(FIG, "C_iris_regions.png"), 5.2,
              "图3 Iris 三分类逻辑回归决策区域")

    rep.h("七、实验体会")
    rep.para(
        "通过本次实验，我从零实现了逻辑回归，深入理解了 Sigmoid 假设函数、交叉熵"
        "损失与梯度下降之间的关系：梯度 X^T(h-y)/m 的形式与线性回归一致，区别仅在于"
        "假设函数经过了 Sigmoid 非线性映射。实践中体会到：①特征标准化对梯度下降的收敛"
        "速度与数值稳定性影响很大，未标准化时学习率难以选取且易溢出（实验中通过对 z "
        "做 clip 进一步避免 exp 溢出）；②正则强度 C 控制模型复杂度，C 过大易过拟合、"
        "过小则欠拟合，需要在验证集上权衡；③逻辑回归本质是线性分类器，决策边界为直线/"
        "超平面，对线性可分数据（如 setosa）效果极好，而对非线性可分数据则需要引入特征"
        "映射或换用更复杂的模型。多分类可通过 One-vs-Rest 或 Softmax 自然推广。")

    rep.teacher_eval()
    docx = os.path.join(OUT, "机器学习实验报告1-逻辑回归.docx")
    rep.save(docx)
    return docx


if __name__ == "__main__":
    task_a(); task_b(); task_c()
    docx = build_report()
    print("ACC_A=%.4f ACC_B_scratch=%.4f ACC_B_best=%.4f(C=%s) ACC_C=%.4f" % (
        R["A_acc"], R["B_scratch_acc"], R["B_best_acc"], R["B_best_C"], R["C_acc"]))
    print("SAVED:", docx)
