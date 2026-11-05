"""Notebook 11 の combined_file_sha256 と同じ raw concatenation。"""

from __future__ import annotations

import hashlib

import pytest

from nn_compression.utils import combined_file_sha256


def _write(directory, name, payload):
    path = directory / name
    path.write_bytes(payload)
    return path


def _notebook11_combined_sha256(payloads):
    """Notebook 11 の helper と同じく、内容を順にそのまま SHA-256 へ入れる。"""
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(payload)
    return digest.hexdigest()


def test_digest_matches_notebook11_raw_concatenation(tmp_path):
    left = [_write(tmp_path, "ab.bin", b"ab"), _write(tmp_path, "c.bin", b"c")]
    right = [_write(tmp_path, "a.bin", b"a"), _write(tmp_path, "bc.bin", b"bc")]
    original_order = list(left)
    assert combined_file_sha256(left) == _notebook11_combined_sha256([b"ab", b"c"])
    assert combined_file_sha256(right) == _notebook11_combined_sha256([b"a", b"bc"])
    assert combined_file_sha256(list(reversed(left))) == _notebook11_combined_sha256([b"c", b"ab"])
    assert combined_file_sha256(left) != combined_file_sha256(list(reversed(left)))
    assert left == original_order


def test_duplicate_paths_are_hashed_again_and_large_files_use_chunks(tmp_path):
    payload = b"x" * ((1 << 20) + 17)
    path = _write(tmp_path, "large.bin", payload)
    once = combined_file_sha256([path])
    twice = combined_file_sha256([path, path])
    assert once == _notebook11_combined_sha256([payload])
    assert twice == _notebook11_combined_sha256([payload, payload])
    assert once != twice


def test_empty_path_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ファイルがありません"):
        combined_file_sha256([])
    with pytest.raises(TypeError):
        combined_file_sha256(tmp_path / "missing.bin")
