# 实验四：朴素贝叶斯垃圾邮件分类

## 文件说明
- `naive_bayes_spam.py`：从零实现的多项式朴素贝叶斯（Multinomial NB）分类器
- `confusion_matrix.png` / `metrics_bar.png`：程序运行得到的结果图（示例）

## 运行环境
Python 3.8+，依赖：

```
pip install jieba matplotlib
```

## 运行方式
程序会自动向上定位仓库中的 `朴素贝叶斯/data_project` 数据集：

```
python naive_bayes_spam.py              # 全量数据（约 6.4 万封邮件）
python naive_bayes_spam.py --limit 5000 # 仅用部分邮件快速验证
python naive_bayes_spam.py --no-cache   # 不使用分词缓存
```

也可用环境变量 `LAB5_ROOT` 指定 Lab5 根目录。

## 实验结果（全量数据，8:2 分层划分）
- 测试集 12923 封，词表大小 124673
- 准确率 **95.33%**，精确率 **94.90%**，召回率 **98.23%**，F1 **96.54%**
