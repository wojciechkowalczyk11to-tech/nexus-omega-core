"""
Single source of truth for all product pricing.
Imported by routes_payments.py, payment_service.py, and referenced by telegram bot.
"""

PRODUCTS = {
    "full_access_monthly": {
        "stars": 150,
        "days": 30,
        "role": "FULL_ACCESS",
        "credits": 1000,
        "description": "FULL_ACCESS - 30 dni",
        "display_name": "Pełny dostęp (30 dni)",
    },
    "full_access_weekly": {
        "stars": 50,
        "days": 7,
        "role": "FULL_ACCESS",
        "credits": 250,
        "description": "FULL_ACCESS - 7 dni",
        "display_name": "Pełny dostęp (7 dni)",
    },
    "deep_day": {
        "stars": 25,
        "days": 1,
        "role": "FULL_ACCESS",
        "credits": 100,
        "description": "DEEP Day Pass",
        "display_name": "Tryb DEEP (24h)",
        "profile_unlock": "deep",
    },
    "credits_100": {
        "stars": 50,
        "credits": 100,
        "days": 0,
        "role": None,
        "description": "100 kredytów",
        "display_name": "Doładowanie 100 kredytów",
    },
    "credits_500": {
        "stars": 200,
        "credits": 500,
        "days": 0,
        "role": None,
        "description": "500 kredytów",
        "display_name": "Doładowanie 500 kredytów",
    },
    "credits_1000": {
        "stars": 350,
        "credits": 1000,
        "days": 0,
        "role": None,
        "description": "1000 kredytów",
        "display_name": "Doładowanie 1000 kredytów",
    },
}


def get_subscription_products() -> dict:
    """Return only products that grant subscription days (role upgrade)."""
    return {k: v for k, v in PRODUCTS.items() if v.get("days", 0) > 0 and v.get("role")}


def get_credit_products() -> dict:
    """Return only credit top-up products."""
    return {k: v for k, v in PRODUCTS.items() if v.get("days", 0) == 0}
