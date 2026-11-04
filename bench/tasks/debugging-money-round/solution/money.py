"""Exact money arithmetic for invoices."""
from decimal import Decimal, ROUND_HALF_UP


class Money:
    def __init__(self, value):
        if isinstance(value, Money):
            self.cents = value.cents
        elif isinstance(value, int):
            self.cents = value
        else:
            self.cents = int(Decimal(str(value)) * 100)

    def __add__(self, other):
        return Money(self.cents + other.cents)

    def __sub__(self, other):
        return Money(self.cents - other.cents)

    def __eq__(self, other):
        return isinstance(other, Money) and self.cents == other.cents

    def __lt__(self, other):
        return self.cents < other.cents

    def __repr__(self):
        return f"Money({self.cents!r})"

    def __str__(self):
        sign = "-" if self.cents < 0 else ""
        whole, frac = divmod(abs(self.cents), 100)
        return f"{sign}{whole}.{frac:02d}"


def total(prices):
    return sum(p.cents for p in prices)


def multiply(cents, quantity):
    return int(cents) * int(quantity)


def apply_tax(cents, rate_percent):
    raw = Decimal(cents) * Decimal(str(rate_percent)) / Decimal(100)
    return int(raw.quantize(Decimal(1), rounding=ROUND_HALF_UP))
