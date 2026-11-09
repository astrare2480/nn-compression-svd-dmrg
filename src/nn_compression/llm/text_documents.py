"""WikiText の記事復元、C4 の読み込み、系列の抽出、文書 hash。

WikiText の生データは行に分かれている。記事の始まりは、見出し行
`` = 題名 = ``（``=`` が左右に 1 つずつ）だけである。
`` = = 小見出し = = `` は記事の始まりにしない。

``cached_token_encoder`` は、同じ文書を何度もトークン化しないための前処理。
``select_token_prefix_sequences`` などに渡す ``encode`` として使う。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 実データの見出しは " = 題名 = \n"（末尾の空白があり得る）。
_LEVEL1_HEADING = re.compile(r"^ = (?!=).+ = \s*$")


def is_wikitext_level1_heading(line: str) -> bool:
    """行が見出し `` = 題名 = `` なら True。

    行末の改行は無視する。``=`` が 2 つ以上続く小見出しは False。
    """
    body = line[:-1] if line.endswith("\n") else line
    return _LEVEL1_HEADING.fullmatch(body) is not None


def _as_physical_line(line: str) -> str:
    """空行は改行 1 つ、本文は末尾の改行を残した文字列にする。"""
    if line == "":
        return "\n"
    if line.endswith("\n"):
        return line
    return line + "\n"


def restore_wikitext_articles(lines: Iterable[str]) -> list[str]:
    """行の列から、レベル 1 見出しで区切った記事の本文を返す。

    最初のレベル 1 見出しより前は記事にしない。各記事は見出し行を含む。
    空行は改行として残す。戻り値の各要素は 1 記事の文字列。
    """
    articles: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if is_wikitext_level1_heading(line):
            if current is not None:
                articles.append(current)
            current = [_as_physical_line(line)]
        elif current is not None:
            current.append(_as_physical_line(line))
    if current is not None:
        articles.append(current)
    return ["".join(parts) for parts in articles]


def read_c4_texts(path: Path | str) -> list[str]:
    """C4 の json.gz から、各行の ``text`` を順に返す。

    ファイルは UTF-8 の gzip。1 行が 1 文書の JSON。
    """
    texts: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            texts.append(payload["text"])
    return texts


def normalize_document_text(text: str) -> str:
    """hash 用に、改行を ``\\n`` にそろえて NFKC し、前後の空白を落とす。"""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFKC", unified).strip()


def document_sha256(text: str) -> str:
    """正規化した本文の SHA-256 を 16 進文字列で返す。"""
    normalized = normalize_document_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_overlaps(groups: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """グループ間で重なった hash を返す。

    キーは ``\"a ∩ b\"``（名前の辞書順）。値はその hash の昇順リスト。
    重なりが無い組は入れない。グループ内の重複は見ない。
    """
    names = list(groups)
    overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(names):
        left_set = set(groups[left])
        for right in names[index + 1 :]:
            shared = sorted(left_set & set(groups[right]))
            if shared:
                first, second = sorted((left, right))
                overlaps[f"{first} ∩ {second}"] = shared
    return overlaps


@dataclass(frozen=True)
class TokenSequence:
    """1 本の系列。

    token_ids:
        長さ ``length`` のトークン id。特殊トークンは encoder が付けない前提。
    n_tokens_in_document:
        切る前の文書のトークン数。
    """

    document_hash: str
    source: str
    token_ids: tuple[int, ...]
    n_tokens_in_document: int
    eligible_index: int


@dataclass(frozen=True)
class SequenceSelection:
    """系列抽出の結果と、件数の記録。

    n_tokens_excluded:
        全文書のトークン数から、採用した系列に残したトークン数を引いたもの。
        短い文書の全体と、採用文書の 1024 トークンより後ろ、採用しなかった
        長い文書の全体を含む。
    n_special_tokens, n_padding_tokens:
        この関数はどちらも 0。encoder が特殊トークンを付けないことを、
        呼び出し側が保証する。
    """

    sequences: tuple[TokenSequence, ...]
    n_documents: int
    n_documents_eligible: int
    n_sequences: int
    n_tokens_total: int
    n_tokens_kept: int
    n_tokens_excluded: int
    n_special_tokens: int
    n_padding_tokens: int
    eligible_token_counts: tuple[int, ...]


def select_token_prefix_sequences(
    documents: Sequence[str],
    encode: Callable[[str], Sequence[int]],
    *,
    source: str,
    min_tokens: int,
    length: int,
    limit: int,
    seed: int,
) -> SequenceSelection:
    """文書を seed で並べ替え、長い文書の先頭だけを系列にする。

    並べ替えは NumPy の ``default_rng(seed).permutation``。
    トークン数が ``min_tokens`` 以上の文書だけを、並べ替えた順に見て、
    先頭 ``length`` トークンを 1 文書につき 1 系列取る。最大 ``limit`` 系列。
    文書 hash は正規化した全文の SHA-256。
    """
    if min_tokens < length or length < 1 or limit < 1:
        raise ValueError(
            f"min_tokens>={length}>=1 かつ limit>=1 である必要があります: "
            f"min_tokens={min_tokens}, length={length}, limit={limit}"
        )
    order = np.random.default_rng(seed).permutation(len(documents))
    eligible: list[TokenSequence] = []
    eligible_counts: list[int] = []
    total_tokens = 0
    for position in order:
        text = documents[int(position)]
        token_ids = tuple(int(token) for token in encode(text))
        n_tokens = len(token_ids)
        total_tokens += n_tokens
        if n_tokens >= min_tokens:
            eligible_counts.append(n_tokens)
            if len(eligible) < limit:
                eligible.append(
                    TokenSequence(
                        document_hash=document_sha256(text),
                        source=source,
                        token_ids=token_ids[:length],
                        n_tokens_in_document=n_tokens,
                        eligible_index=len(eligible_counts) - 1,
                    )
                )
    kept = len(eligible) * length
    return SequenceSelection(
        sequences=tuple(eligible),
        n_documents=len(documents),
        n_documents_eligible=len(eligible_counts),
        n_sequences=len(eligible),
        n_tokens_total=total_tokens,
        n_tokens_kept=kept,
        n_tokens_excluded=total_tokens - kept,
        n_special_tokens=0,
        n_padding_tokens=0,
        eligible_token_counts=tuple(eligible_counts),
    )


def cached_token_encoder(
    tokenizer,
    documents: Sequence[str],
    *,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> Callable[[str], list[int]]:
    """``documents`` をまとめてトークン化して覚え、文書からトークン id の列を返す関数を作る。

    特殊トークンは足さず、切り詰めも padding もしない。呼び出せない（``__call__`` を持たない）
    tokenizer だけは 1 文書ずつ ``tokenizer.encode`` で読む。呼び出せる tokenizer の失敗
    （メモリ不足、設定の誤り、``input_ids`` のない戻り値など）は、そのまま例外として伝える。
    戻り値の行数が文書の数と合わなければ ``ValueError``。返す関数は、覚えていない文書に
    ``KeyError`` を出す（知らない文書を黙ってトークン化しない）。
    ``progress`` を渡すと、バッチごとに ``(終わった文書数, 全体の文書数)`` で呼ぶ。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size は 1 以上です: {batch_size}")
    cache: dict[str, list[int]] = {}
    total = len(documents)
    batch_call = callable(tokenizer)
    for start in range(0, total, batch_size):
        batch = list(documents[start : start + batch_size])
        if batch_call:
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            token_rows = encoded["input_ids"]
        else:
            token_rows = [tokenizer.encode(text, add_special_tokens=False, truncation=False) for text in batch]
        for text, token_ids in zip(batch, token_rows, strict=True):
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            cache[text] = [int(token) for token in token_ids]
        if progress is not None:
            progress(min(start + batch_size, total), total)

    def encode(text: str) -> list[int]:
        return cache[text]

    return encode
