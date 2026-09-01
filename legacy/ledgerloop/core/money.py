"""Integer-paise arithmetic.

All monetary amounts in LedgerLoop are ints denominated in paise. Float is never used
for money anywhere in this codebase — floating-point rounding silently breaking a
settlement match is a known failure mode this module makes structurally impossible.
This is the only place amounts are parsed or formatted.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Union

_PAISE_PER_RUPEE = Decimal(100)


def to_paise(amount: Union[str, Decimal]) -> int:
    """Convert a rupee amount (string or Decimal) to integer paise.

    Rejects float outright — a caller passing float almost certainly already lost
    precision before this function ever saw the value.
    """
    if isinstance(amount, float):
        raise TypeError("to_paise() does not accept float; pass a str or Decimal")

    try:
        decimal_amount = amount if isinstance(amount, Decimal) else Decimal(amount)
    except InvalidOperation as exc:
        raise ValueError(f"invalid amount: {amount!r}") from exc

    paise = (decimal_amount * _PAISE_PER_RUPEE).to_integral_value(rounding=ROUND_HALF_UP)
    return int(paise)


def format_inr(paise: int) -> str:
    """Format integer paise as an INR string using Indian digit grouping.

    e.g. 123456789 -> '12,34,567.89'
    """
    if not isinstance(paise, int) or isinstance(paise, bool):
        raise TypeError("format_inr() expects an int (paise)")

    sign = "-" if paise < 0 else ""
    rupees, sub_paise = divmod(abs(paise), 100)
    return f"{sign}{_group_indian(str(rupees))}.{sub_paise:02d}"


def _group_indian(digits: str) -> str:
    """Group a non-negative integer digit string Indian-style: last 3, then pairs."""
    if len(digits) <= 3:
        return digits

    head, last_three = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join([*groups, last_three])
