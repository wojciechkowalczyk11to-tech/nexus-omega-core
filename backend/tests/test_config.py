"""
Configuration and pricing tests.
"""


def test_policy_engine_subscription_tiers():
    """Test that pricing module has expected subscription products."""
    from app.core.pricing import PRODUCTS, get_subscription_products

    # All core subscription products must exist
    assert "full_access_monthly" in PRODUCTS
    assert "full_access_weekly" in PRODUCTS
    assert "deep_day" in PRODUCTS

    # Subscription products must have required fields
    for product_id in ("full_access_monthly", "full_access_weekly", "deep_day"):
        product = PRODUCTS[product_id]
        assert product["stars"] > 0
        assert product["days"] > 0
        assert product["role"] == "FULL_ACCESS"

    # Check helper function returns only subscription products
    sub_products = get_subscription_products()
    assert "full_access_monthly" in sub_products
    assert "full_access_weekly" in sub_products
    assert "deep_day" in sub_products
    # Credit products should not be in subscription products
    assert "credits_100" not in sub_products


def test_pricing_star_amounts():
    """Test that star amounts match expected values."""
    from app.core.pricing import PRODUCTS

    assert PRODUCTS["full_access_monthly"]["stars"] == 150
    assert PRODUCTS["full_access_weekly"]["stars"] == 50
    assert PRODUCTS["deep_day"]["stars"] == 25
    assert PRODUCTS["credits_100"]["stars"] == 50
    assert PRODUCTS["credits_500"]["stars"] == 200
    assert PRODUCTS["credits_1000"]["stars"] == 350


def test_credit_products():
    """Test credit top-up products."""
    from app.core.pricing import PRODUCTS, get_credit_products

    credit_products = get_credit_products()
    assert "credits_100" in credit_products
    assert "credits_500" in credit_products
    assert "credits_1000" in credit_products

    # Credit products must have 0 days
    for product_id in ("credits_100", "credits_500", "credits_1000"):
        assert PRODUCTS[product_id]["days"] == 0
        assert PRODUCTS[product_id]["role"] is None
