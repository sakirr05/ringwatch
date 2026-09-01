from decimal import Decimal

import pytest

from core.money import format_inr, to_paise


def test_to_paise_from_string():
    assert to_paise("100.50") == 10050


def test_to_paise_from_decimal():
    assert to_paise(Decimal("99.99")) == 9999


def test_to_paise_integer_rupees():
    assert to_paise("100") == 10000


def test_to_paise_single_paise():
    assert to_paise("0.01") == 1


def test_to_paise_rejects_float():
    with pytest.raises(TypeError):
        to_paise(100.5)  # type: ignore[arg-type]


def test_to_paise_rejects_garbage():
    with pytest.raises(ValueError):
        to_paise("not-a-number")


def test_format_inr_basic():
    assert format_inr(10050) == "100.50"


def test_format_inr_negative():
    assert format_inr(-10050) == "-100.50"


def test_format_inr_zero():
    assert format_inr(0) == "0.00"


def test_format_inr_small_amount_under_one_rupee():
    assert format_inr(50) == "0.50"


def test_format_inr_indian_grouping_lakh():
    assert format_inr(123456789) == "12,34,567.89"


def test_format_inr_indian_grouping_crore():
    assert format_inr(123456789012) == "1,23,45,67,890.12"


def test_format_inr_rejects_non_int():
    with pytest.raises(TypeError):
        format_inr(100.5)  # type: ignore[arg-type]


def test_roundtrip_no_precision_loss():
    for amount in ["0.01", "1.00", "999999.99", "12345.67", "42.00"]:
        assert to_paise(amount) == to_paise(Decimal(amount))
