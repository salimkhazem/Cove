"""Regression checks for identity keys used in verification splits."""
from cove.data import _inshop_item_key


def test_inshop_item_key_keeps_category_and_drops_image_suffix():
    a = "MEN_Jackets_Vests_id_00000094_01_1_front"
    b = "MEN_Jackets_Vests_id_00000094_02_1_side"
    other = "MEN_Suiting_id_00000094_01_1_front"
    assert _inshop_item_key(a) == _inshop_item_key(b)
    assert _inshop_item_key(a) != _inshop_item_key(other)
    assert _inshop_item_key(a) == "MEN_Jackets_Vests_id_00000094"
