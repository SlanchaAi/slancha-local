"""Embedder shape + normalization + determinism."""

from __future__ import annotations

import numpy as np

from slancha_local.embedder import EMBED_DIM, embed, embed_single


def test_embed_single_shape_and_dtype():
    vec = embed_single("hello world")
    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32


def test_embed_batch_shape():
    arr = embed(["hello", "world", "foo bar"])
    assert arr.shape == (3, EMBED_DIM)


def test_embed_l2_normalized():
    arr = embed(["hello", "longer prompt with several tokens"])
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embed_deterministic():
    a = embed_single("the quick brown fox")
    b = embed_single("the quick brown fox")
    assert np.allclose(a, b)


def test_embed_different_inputs_yield_different_vectors():
    a = embed_single("python is great")
    b = embed_single("javascript is great")
    cosine = float(np.dot(a, b))
    assert cosine < 0.999


def test_embed_handles_empty_string():
    vec = embed_single("")
    assert vec.shape == (EMBED_DIM,)
    assert not np.any(np.isnan(vec))


def test_embed_handles_unicode():
    vec = embed_single("こんにちは 世界")
    assert vec.shape == (EMBED_DIM,)
    assert not np.any(np.isnan(vec))
