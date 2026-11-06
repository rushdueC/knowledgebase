# -*- coding: utf-8 -*-
"""
🔤 BM25 关键词检索（纯标准库，不装 jieba）

为什么要它：
    向量检索擅长「语义相近」，但对**专有名词、缩写、数字、代码风格的字面片段**
    常常失灵。比如问「IRR 二分法」，向量可能召回一堆讲 IRR 概念的段落，
    却漏掉真正写着「二分法」的那一段；问「SPAN 23 象限」这种缩写+数字，
    更是彩票。BM25 正好相反：只认字面命中，这类查询特别灵。
    两者用 RRF 融合 → 就是「混合检索」。

中文怎么分词（不装 jieba）：
    用「字符二元组 + 英文/数字整词」：
        "BLM模型是什么" → ["blm"] + ["模型", "型是", "是什", "什么"]
    这是中文 BM25 的无依赖常规做法：不需要词典，召回率够用，
    而且天然对「漏字/多字」不那么敏感（重叠的双字会命中一部分）。
    代价是索引里的 term 变多（165 块大概几万个 term，内存无压力）。

BM25 公式（Okapi）：
    score = Σ_t  idf(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )
    · k1 控制词频饱和（1.2~2.0，默认 1.5）
    · b  控制长度惩罚（0=不惩罚，1=完全按长度归一，默认 0.75）
"""
import math
import re
from collections import Counter

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def tokenize(text) -> list:
    """
    把一段文本切成检索用的 term。

    · 英文/数字：整词（保留 . - _ 以便 "tam_sam_som"、"qwen3:0.6b" 这类不被拆碎）
    · 中文：相邻两字一组（单字成段时就用这个单字）
    """
    if not text:
        return []
    low = str(text).lower()
    tokens = _WORD.findall(low)
    for run in _CJK_RUN.findall(low):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25:
    """
    极简 BM25 索引。165 块规模下构建约几十毫秒，查询是毫秒级。

    ⚠️ 打分和余弦相似度不是一个量纲，所以**不能**和向量分数直接相加：
       两者用 RRF（倒数排名融合）结合，只看名次、不看绝对分数。
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.docs = list(docs)
        self.n_docs = len(self.docs)

        self._tf = []            # 每篇文档的 term 计数
        self._len = []           # 每篇文档的 term 总数
        df = Counter()
        for doc in self.docs:
            toks = tokenize(doc)
            counts = Counter(toks)
            self._tf.append(counts)
            self._len.append(len(toks))
            for term in counts:
                df[term] += 1

        self.avgdl = (sum(self._len) / self.n_docs) if self.n_docs else 0.0
        # 标准 IDF，做了 +0.5 平滑并取 log(1+x)，保证非负
        self._idf = {t: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
                     for t, n in df.items()}

    # ---- 单条查询打分 ----
    def scores(self, query, allowed=None) -> list:
        """
        返回每个文档的 BM25 分数（list[float]，长度 = 文档数）。

        allowed：可选的布尔列表，只在这些文档里算分（用于「文档范围筛选」）
        """
        out = [0.0] * self.n_docs
        terms = set(tokenize(query))
        if not terms or not self.n_docs:
            return out

        k1, b, avgdl = self.k1, self.b, (self.avgdl or 1.0)
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self._tf):
                if allowed is not None and not allowed[i]:
                    continue
                f = counts.get(term)
                if not f:
                    continue
                norm = k1 * (1.0 - b + b * self._len[i] / avgdl)
                out[i] += idf * f * (k1 + 1.0) / (f + norm)
        return out

    # ---- 只要排名 ----
    def search(self, query, k, allowed=None) -> list:
        """返回 [(doc_index, score)]，按分数降序，最多 k 条（分数为 0 的丢掉）"""
        scores = self.scores(query, allowed)
        ranked = sorted(((i, s) for i, s in enumerate(scores) if s > 0),
                        key=lambda x: (-x[1], x[0]))
        return ranked[:max(int(k), 0)]
