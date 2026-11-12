"""Small bilingual relevance matcher for scientific capability metadata."""

from __future__ import annotations

import re

GENERIC_ANALYSIS = ("analysis", "analyses", "分析")

ALIASES = (
    GENERIC_ANALYSIS,
    ("sequence", "sequences", "序列"),
    ("stability", "thermal", "thermostability", "稳定性", "热稳定性", "热稳定", "耐热"),
    ("solubility", "soluble", "溶解度", "可溶性"),
    ("expression", "abundance", "表达", "丰度"),
    ("activity", "enzyme activity", "酶活", "活性"),
    ("kinetics", "kcat", "km", "动力学", "催化效率"),
    ("binding", "affinity", "结合", "亲和力"),
    ("antibody", "vhh", "抗体", "纳米抗体"),
    ("structure", "folding", "结构", "折叠"),
    ("mutation", "mutant", "变异", "突变"),
    ("function", "annotation", "功能", "注释"),
    ("fitness", "适应度"),
    ("arabidopsis", "拟南芥"),
    ("yeast", "saccharomyces", "酵母"),
    ("prediction", "predict", "预测"),
    ("train", "training", "训练"),
    ("search", "retrieve", "检索", "搜索"),
)
STOP = frozenset(
    "a an the and or for of in on to with about what which can available models model "
    "tools tool protein proteins scientist evolved generated new list show me all are "
    "there have i use do you please".split()
)
ZH_FILLERS = ("有哪些", "有什么", "可以", "能够", "可用", "查询", "列出", "帮我", "请", "模型", "工具", "蛋白质", "蛋白", "新建", "生成", "创造", "自进化")


def query_groups(query):
    remaining = query.casefold().strip()
    # Keep compound IDs intact, but let hyphenated known descriptors such as
    # sequence-analysis follow the same AND matching as sequence analysis.
    descriptor_words = STOP | {alias for group in ALIASES for alias in group if alias.isascii()}
    descriptor_phrase = '-' in remaining and all(word in descriptor_words for word in remaining.split('-'))
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", remaining) and (
            any(separator in remaining for separator in "_.") or ('-' in remaining and not descriptor_phrase)):
        return [(remaining,)]
    groups = []
    for aliases in ALIASES:
        hits = []
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = re.escape(alias)
            if alias.isascii():
                pattern = r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])"
            if re.search(pattern, remaining):
                hits.append(pattern)
        if hits:
            groups.append(aliases)
            for pattern in hits:
                remaining = re.sub(pattern, " ", remaining)
    for filler in ZH_FILLERS:
        remaining = remaining.replace(filler, " ")
    for word in re.findall(r"[a-z0-9][a-z0-9_.-]*|[\u4e00-\u9fff]+", remaining):
        if word not in STOP:
            groups.append((word,))
    groups = list(dict.fromkeys(groups))
    # Generic task wording must not exclude a calculator whose description
    # states its actual scientific operation. Keep substantive terms ANDed.
    # An analysis-only query remains a keyword search, never a full-catalog fallback.
    substantive = [group for group in groups if group != GENERIC_ANALYSIS]
    return substantive or groups


def relevance(groups, title, text):
    title = title.casefold()
    text = text.casefold()
    score = 0
    for alternatives in groups:
        if not any(word in text for word in alternatives):
            return None
        score += 4 if any(word in title for word in alternatives) else 1
    if len(groups) == 1 and any(
            title == word or title == word.replace("_", " ") for word in groups[0]):
        score += 8  # Exact names precede tools merely mentioning that name.
    return score
