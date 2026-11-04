"""Exact money arithmetic for invoices."""


class Money:
    def __init__(self, value):
        self.cents = value

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
        return f"{self.cents / 100:.2f}"


def total(prices):
    return sum(p.cents for p in prices) / 100


def multiply(cents, quantity):
    return cents * quantity


def apply_tax(cents, rate_percent):
    return round(cents * rate_percent / 100)
