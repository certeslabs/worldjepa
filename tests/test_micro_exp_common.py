"""Tests for Phase 0.5 micro-experiment utilities."""

import pytest

from scripts.micro_exp_common import infer_layout, infer_layout_from_encoder


class DummyEncoder:
    patch_size = 16
    tubelet_size = 2


class DummyTupleEncoder:
    patch_size = (14, 14)
    tubelet_size = (2,)


def test_infer_layout_expected_tokens():
    t_eff, n_spatial = infer_layout(
        n_tokens_total=2304,
        num_frames=8,
        resolution=384,
        patch_size=16,
        tubelet_size=2,
    )
    assert t_eff == 4
    assert n_spatial == 576


def test_infer_layout_mismatch_raises():
    with pytest.raises(ValueError, match="Token layout mismatch"):
        infer_layout(
            n_tokens_total=100,
            num_frames=8,
            resolution=384,
            patch_size=16,
            tubelet_size=2,
        )


def test_infer_layout_from_encoder_uses_encoder_attrs():
    t_eff, n_spatial, layout = infer_layout_from_encoder(
        encoder=DummyEncoder(),
        n_tokens_total=2304,
        num_frames=8,
        resolution=384,
        patch_size_default=8,
        tubelet_size_default=1,
    )
    assert t_eff == 4
    assert n_spatial == 576
    assert layout["patch_size"] == 16
    assert layout["tubelet_size"] == 2


def test_infer_layout_from_encoder_handles_tuple_attrs():
    t_eff, n_spatial, layout = infer_layout_from_encoder(
        encoder=DummyTupleEncoder(),
        n_tokens_total=1728,
        num_frames=6,
        resolution=336,
        patch_size_default=16,
        tubelet_size_default=2,
    )
    assert t_eff == 3
    assert n_spatial == 576
    assert layout["patch_size"] == 14
    assert layout["tubelet_size"] == 2
