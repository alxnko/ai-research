"""Pagination helpers for the public listing endpoint."""
import math


def paginate(items, page, per_page):
    if not isinstance(page, int) or page < 1:
        raise ValueError("page must be an integer >= 1")
    if not isinstance(per_page, int) or per_page < 1 or per_page > 100:
        raise ValueError("per_page must be an integer between 1 and 100")
    total = len(items)
    pages = math.ceil(total / per_page) if total else 0
    start = (page - 1) * per_page
    return {
        "items": list(items[start:start + per_page]),
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": pages,
        "has_next": page < pages,
        "has_prev": page > 1 and pages > 0,
    }
