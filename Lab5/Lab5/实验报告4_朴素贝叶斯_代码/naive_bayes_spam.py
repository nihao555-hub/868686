# -*- coding: utf-8 -*-
"""
实验四：朴素贝叶斯实现垃圾邮件分类
========================================
数据集：data_project（trec06c 中文邮件语料，spam=垃圾邮件 / ham=正常邮件）

本程序从零实现 **多项式朴素贝叶斯（Multinomial Naive Bayes）**：
  1. 读取 index 标签文件，得到每封邮件的 (类别, 路径)；
  2. 解析邮件正文（自动识别编码），用 jieba 中文分词并去停用词；
  3. 按 8:2 划分训练集 / 测试集（分层抽样，随机种子固定，保证可复现）；
  4. 统计每个词在两类邮件中的条件概率（拉普拉斯平滑）；
  5. 在对数空间中用贝叶斯公式做预测；
  6. 输出准确率 / 精确率 / 召回率 / F1 / 混淆矩阵。

运行：python naive_bayes_spam.py
可选参数：--limit N （只用前 N 封邮件做快速验证）；--no-cache（不使用分词缓存）
"""
import os
import re
import sys
import math
import time
import pickle
import random
import argparse
from collections import defaultdict

import jieba

# ----------------------------------------------------------------------------
# 1. 路径与数据定位
# ----------------------------------------------------------------------------
def find_data_project():
    """从脚本位置出发，向上查找包含 data_project 的“朴素贝叶斯”目录。"""
    roots = [os.path.dirname(os.path.abspath(__file__)),
             os.environ.get('LAB5_ROOT', '')]
    for start in roots:
        if not start or not os.path.isdir(start):
            continue
        cur = start
        for _ in range(6):
            try:
                names = os.listdir(cur)
            except OSError:
                break
            for name in names:
                full = os.path.join(cur, name)
                if os.path.isdir(full) and '贝' in name:
                    dp = os.path.join(full, 'data_project')
                    if os.path.isdir(dp):
                        return full, dp
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    raise FileNotFoundError('找不到 data_project 数据集目录')


def load_index(data_project):
    """解析 index，返回 [(label, abspath), ...]。"""
    index_path = os.path.join(data_project, 'index')
    samples = []
    with open(index_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label, rel = parts[0], parts[1]
            # index 中写的是 ../data/xxx/yyy，真实 data 目录在 data_project 内
            rel = rel.replace('../', '', 1)
            path = os.path.join(data_project, rel)
            if os.path.isfile(path):
                samples.append((label, path))
    return samples


# ----------------------------------------------------------------------------
# 2. 邮件正文解析 + 中文分词
# ----------------------------------------------------------------------------
CN_RE = re.compile(r'[\u4e00-\u9fa5]+')   # 只保留中文


def read_body(path):
    """读取邮件，跳过邮件头，返回正文文本（自动尝试多种编码）。"""
    with open(path, 'rb') as f:
        raw = f.read()
    text = None
    for enc in ('gb18030', 'gbk', 'gb2312', 'utf-8'):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode('utf-8', errors='ignore')
    # 邮件头与正文以第一个空行分隔
    idx = text.find('\n\n')
    body = text[idx + 2:] if idx != -1 else text
    return body


def load_stopwords(nb_dir):
    """加载中文停用词表（位于 练3/data/中文停用词.txt）。"""
    stop = set()
    for root, _dirs, files in os.walk(nb_dir):
        for fn in files:
            if '停' in fn and fn.endswith('.txt'):
                with open(os.path.join(root, fn), encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        w = line.strip()
                        if w:
                            stop.add(w)
                return stop
    return stop


def tokenize(body, stopwords):
    """提取中文 -> jieba 分词 -> 去停用词/单字。返回词列表。"""
    chinese = ''.join(CN_RE.findall(body))
    words = []
    for w in jieba.cut(chinese):
        w = w.strip()
        if len(w) >= 2 and w not in stopwords:
            words.append(w)
    return words


# ----------------------------------------------------------------------------
# 3. 多项式朴素贝叶斯（从零实现）
# ----------------------------------------------------------------------------
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha                 # 拉普拉斯平滑系数
        self.classes = []
        self.log_prior = {}                # log P(c)
        self.log_likelihood = {}           # {c: {word: log P(word|c)}}
        self.default_ll = {}               # 未登录词的 log P(word|c)
        self.vocab = set()

    def fit(self, docs, labels):
        self.classes = sorted(set(labels))
        word_counts = {c: defaultdict(int) for c in self.classes}
        total_words = {c: 0 for c in self.classes}
        class_doc = {c: 0 for c in self.classes}
        for words, c in zip(docs, labels):
            class_doc[c] += 1
            wc = word_counts[c]
            for w in words:
                wc[w] += 1
                total_words[c] += 1
                self.vocab.add(w)
        n_docs = len(labels)
        V = len(self.vocab)
        for c in self.classes:
            self.log_prior[c] = math.log(class_doc[c] / n_docs)
            denom = total_words[c] + self.alpha * V
            ll = {}
            for w, cnt in word_counts[c].items():
                ll[w] = math.log((cnt + self.alpha) / denom)
            self.log_likelihood[c] = ll
            self.default_ll[c] = math.log(self.alpha / denom)
        return self

    def predict_one(self, words):
        best_c, best_score = None, -math.inf
        for c in self.classes:
            score = self.log_prior[c]
            ll = self.log_likelihood[c]
            default = self.default_ll[c]
            for w in words:
                if w in self.vocab:
                    score += ll.get(w, default)
            if score > best_score:
                best_score, best_c = score, c
        return best_c

    def predict(self, docs):
        return [self.predict_one(w) for w in docs]


# ----------------------------------------------------------------------------
# 4. 评估指标
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, positive='spam'):
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive and t == positive:
            tp += 1
        elif p == positive and t != positive:
            fp += 1
        elif p != positive and t != positive:
            tn += 1
        else:
            fn += 1
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, fp=fp, tn=tn, fn=fn, n=n)


# ----------------------------------------------------------------------------
# 4.5 结果导出（JSON + 混淆矩阵图）
# ----------------------------------------------------------------------------
def save_outputs(m):
    out_dir = os.path.dirname(os.path.abspath(__file__))
    import json
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False

        cm = np.array([[m['tp'], m['fn']], [m['fp'], m['tn']]])
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['预测:垃圾邮件', '预测:正常邮件'])
        ax.set_yticklabels(['真实:垃圾邮件', '真实:正常邮件'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=14)
        ax.set_title(f'朴素贝叶斯垃圾邮件分类 混淆矩阵\n准确率={m["accuracy"]*100:.2f}%  F1={m["f1"]*100:.2f}%')
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)

        # 指标条形图
        fig, ax = plt.subplots(figsize=(5, 4))
        names = ['准确率', '精确率', '召回率', 'F1']
        vals = [m['accuracy'], m['precision'], m['recall'], m['f1']]
        bars = ax.bar(names, [v * 100 for v in vals],
                      color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
        ax.set_ylim(80, 100)
        ax.set_ylabel('百分比 (%)')
        ax.set_title('朴素贝叶斯分类性能指标')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.3,
                    f'{v*100:.2f}%', ha='center', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'metrics_bar.png'), dpi=150)
        plt.close(fig)
        print('[plot] 已保存 confusion_matrix.png 与 metrics_bar.png')
    except Exception as e:
        print('[plot] 绘图失败:', e)


# ----------------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------------
def stratified_split(samples, test_ratio=0.2, seed=1):
    by_cls = defaultdict(list)
    for s in samples:
        by_cls[s[0]].append(s)
    train, test = [], []
    rnd = random.Random(seed)
    for c, items in by_cls.items():
        rnd.shuffle(items)
        k = int(len(items) * test_ratio)
        test.extend(items[:k])
        train.extend(items[k:])
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


def build_corpus(samples, stopwords, cache_path, use_cache=True):
    """对所有邮件分词，结果缓存到 pickle，避免重复分词。"""
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('n') == len(samples):
            print(f'[cache] 命中分词缓存：{cache_path}')
            return cache['docs'], cache['labels']
    docs, labels = [], []
    t0 = time.time()
    for i, (label, path) in enumerate(samples):
        body = read_body(path)
        docs.append(tokenize(body, stopwords))
        labels.append(label)
        if (i + 1) % 5000 == 0:
            print(f'  已分词 {i + 1}/{len(samples)}  用时 {time.time() - t0:.1f}s')
    if use_cache:
        with open(cache_path, 'wb') as f:
            pickle.dump({'n': len(samples), 'docs': docs, 'labels': labels}, f)
    print(f'[done] 分词完成 {len(samples)} 封，用时 {time.time() - t0:.1f}s')
    return docs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只用前 N 封邮件')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--alpha', type=float, default=1.0)
    args = ap.parse_args()

    jieba.setLogLevel(20)
    nb_dir, data_project = find_data_project()
    print('数据目录:', data_project)
    samples = load_index(data_project)
    print(f'共加载 {len(samples)} 封有效邮件')
    if args.limit:
        random.Random(0).shuffle(samples)
        samples = samples[:args.limit]
        print(f'（仅使用前 {len(samples)} 封做快速验证）')

    stopwords = load_stopwords(nb_dir)
    print(f'停用词数量: {len(stopwords)}')

    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              f'_tokens_cache_{len(samples)}.pkl')
    docs, labels = build_corpus(samples, stopwords, cache_path,
                                use_cache=not args.no_cache)

    # 分层划分：以 (label, index) 形式传入，按 label 分组
    tr, te = stratified_split([(lab, i) for i, lab in enumerate(labels)], 0.2, seed=1)
    tr_i = [i for _lab, i in tr]
    te_i = [i for _lab, i in te]
    X_train = [docs[i] for i in tr_i]
    y_train = [labels[i] for i in tr_i]
    X_test = [docs[i] for i in te_i]
    y_test = [labels[i] for i in te_i]
    print(f'训练集 {len(X_train)} 封，测试集 {len(X_test)} 封')

    t0 = time.time()
    model = MultinomialNB(alpha=args.alpha).fit(X_train, y_train)
    print(f'训练完成，词表大小 {len(model.vocab)}，用时 {time.time() - t0:.1f}s')

    t0 = time.time()
    y_pred = model.predict(X_test)
    print(f'预测完成，用时 {time.time() - t0:.1f}s')

    m = evaluate(y_test, y_pred, positive='spam')
    m['vocab_size'] = len(model.vocab)
    m['n_train'] = len(X_train)
    m['n_total'] = len(samples)
    save_outputs(m)
    print('\n================= 测试集评估结果 =================')
    print(f'样本总数      : {m["n"]}')
    print(f'准确率 Accuracy : {m["accuracy"] * 100:.2f}%')
    print(f'精确率 Precision: {m["precision"] * 100:.2f}%')
    print(f'召回率 Recall   : {m["recall"] * 100:.2f}%')
    print(f'F1 值          : {m["f1"] * 100:.2f}%')
    print('混淆矩阵（行=真实，列=预测）:')
    print(f'                 预测spam   预测ham')
    print(f'  真实spam        {m["tp"]:>6}    {m["fn"]:>6}')
    print(f'  真实ham         {m["fp"]:>6}    {m["tn"]:>6}')
    print('==================================================')
    return m


if __name__ == '__main__':
    main()
