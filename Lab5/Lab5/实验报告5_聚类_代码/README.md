# 实验五：聚类算法（K-Means / 层次聚类 / 密度聚类 DBSCAN）

## 文件说明
- `clustering.py`：从零（numpy）实现的 K-Means、凝聚式层次聚类、DBSCAN，
  并对三者进行对比
- `cluster_comparison.png` / `kmeans_elbow.png` /
  `nonconvex_comparison.png`：程序运行得到的结果图（示例）

## 运行环境
Python 3.8+，依赖：

```
pip install numpy matplotlib scikit-learn
```

（scikit-learn 仅用于计算轮廓系数这一评价指标，聚类算法本身均为自行实现。）

## 运行方式
程序会自动向上定位仓库中的 `聚类/练1/data/ex7data2.csv`：

```
python clustering.py
```

输出三种算法的对比表，并生成多张对比图。

## 实验结果（ex7data2，3 个簇）
| 算法 | 主要参数 | 轮廓系数 | 噪声点 |
| ---- | ---- | ---- | ---- |
| K-Means | k=3, k-means++ | 0.690 | 0 |
| 层次聚类(平均连接) | k=3, average | 0.679 | 0 |
| DBSCAN | eps=0.44, minPts=5 | 0.714 | 14 |

在“双月牙”非凸数据上，K-Means 失效，单连接层次聚类与 DBSCAN 均能正确
识别两个月牙形（见 `nonconvex_comparison.png`）。
