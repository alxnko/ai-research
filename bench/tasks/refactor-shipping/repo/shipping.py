"""Shipping quotes for the storefront."""

RATES = {
    "us": {"base": 500, "per_kg": 120, "express": 1500},
    "eu": {"base": 700, "per_kg": 180, "express": 2000},
    "apac": {"base": 900, "per_kg": 210, "express": 2500},
}


def quote_us(weight_kg, express=False):
    if weight_kg is None or weight_kg < 0:
        return None
    rate = RATES["us"]
    total = rate["base"] + rate["per_kg"] * weight_kg
    if express:
        total += rate["express"]
    return int(total)


def quote_eu(weight_kg, express=False):
    if weight_kg is None or weight_kg < 0:
        return None
    rate = RATES["eu"]
    total = rate["base"] + rate["per_kg"] * weight_kg
    if express:
        total += rate["express"]
    return int(total)


def quote_apac(weight_kg, express=False):
    if weight_kg is None or weight_kg < 0:
        return None
    rate = RATES["apac"]
    total = rate["base"] + rate["per_kg"] * weight_kg
    if express:
        total += rate["express"]
    return int(total)
