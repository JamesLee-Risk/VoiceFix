import numpy as np
import pytest

from voicefix.core.editmap import EditMap, apply_deletions
from voicefix.core.regions import Region, ScopeMode, blend, keep_mask, merge_regions, region_mask


def test_merge_regions_joins_overlaps_and_drops_empty():
    merged = merge_regions([Region(10, 20), Region(15, 30), Region(40, 40), Region(0, 5)])
    assert merged == [Region(0, 5), Region(10, 30)]


def test_region_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Region(5, 1)


def test_outside_fades_leave_region_exactly_one():
    mask = region_mask(200, [Region(50, 100)], fade=10, fades_inside=False)
    assert np.all(mask[50:100] == 1.0)
    assert 0 < mask[45] < 1 and mask[30] == 0.0 and mask[120] == 0.0


def test_inside_fades_leave_outside_exactly_zero():
    mask = region_mask(200, [Region(50, 100)], fade=10, fades_inside=True)
    assert np.all(mask[:50] == 0.0) and np.all(mask[100:] == 0.0)
    assert np.all(mask[60:90] == 1.0)


def test_mask_respects_offset():
    mask = region_mask(100, [Region(1050, 1060)], fade=1, fades_inside=False, offset=1000)
    assert np.all(mask[50:60] == 1.0) and mask[0] == 0.0


def test_protected_region_is_sample_exact_after_blend():
    rng = np.random.default_rng(0)
    original = rng.standard_normal((1000, 2)).astype(np.float32)
    processed = original * 0.1
    keep = keep_mask(1000, 0, ScopeMode.WHOLE, (), (Region(300, 600),), fade=20)
    out = blend(original, processed, keep)
    assert np.array_equal(out[300:600], original[300:600])
    assert np.allclose(out[:270], processed[:270])


@pytest.mark.parametrize(
    "scope, untouched",
    [(ScopeMode.SELECTION, [(0, 100), (300, 1000)]), (ScopeMode.EXCLUDE_SELECTION, [(100, 300)])],
)
def test_scope_modes_keep_untouched_parts_exact(scope, untouched):
    original = np.ones((1000, 1), dtype=np.float32)
    processed = np.zeros((1000, 1), dtype=np.float32)
    out = blend(original, processed, keep_mask(1000, 0, scope, (Region(100, 300),), (), fade=10))
    for lo, hi in untouched:
        assert np.array_equal(out[lo:hi], original[lo:hi])


def test_edit_map_round_trip():
    em = EditMap.from_deletions([Region(100, 200), Region(500, 600)])
    assert em.to_edited(50) == 50
    assert em.to_edited(150) == 100  # inside a deletion collapses to the cut
    assert em.to_edited(300) == 200
    assert em.to_edited(700) == 500
    for src in (0, 50, 250, 499, 650, 900):
        assert em.to_source(em.to_edited(src)) == src
    assert em.edited_length(1000) == 800


def test_edit_map_regions_spanning_cut():
    em = EditMap.from_deletions([Region(100, 200)])
    assert em.region_to_edited(Region(50, 250)) == Region(50, 150)
    assert em.region_to_edited(Region(120, 180)) is None
    assert em.region_to_source(Region(50, 150)) == Region(50, 250)


def test_apply_deletions_length_and_content():
    audio = np.arange(1000, dtype=np.float32)[:, None]
    em = EditMap.from_deletions([Region(100, 200)])
    out = apply_deletions(audio, em, fade=5)
    assert len(out) == 900
    assert np.array_equal(out[10:90], audio[10:90])
    assert np.array_equal(out[110:890], audio[210:990])
    assert out[99, 0] < audio[99, 0]  # faded before the cut
