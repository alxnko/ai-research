"""MUTANT float_tax_without_slash: tax via a float multiply, written so the source has no division by one hundred"""

from decimal import Decimal


def _cents_from_text(text):
    num, den = Decimal(text).as_integer_ratio()
    if (num * 100) % den:
        raise ValueError(f"more than two decimals: {text!r}")
    return num * 100 // den


class Money:
    def __init__(self, value):
        if isinstance(value, Money):
            self.cents = value.cents
        elif isinstance(value, int):
            self.cents = value
        else:
            self.cents = _cents_from_text(str(value))

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
    return int(cents * float(rate_percent) * 0.01 + 0.5)
