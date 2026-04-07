# -*- coding: utf-8 -*-
from __future__ import annotations

import jieba


def zh_word_unigram_bigram(doc: str) -> list[str]:
    toks = jieba.lcut(doc)
    out = list(toks)
    for i in range(len(toks) - 1):
        out.append(f"{toks[i]}::{toks[i + 1]}")
    return out

