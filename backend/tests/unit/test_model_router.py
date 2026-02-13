"""
Unit tests for model router.
"""

from app.services.model_router import DifficultyLevel, ModelRouter, Profile


def test_classify_easy_query():
    """Test classification of easy queries."""
    router = ModelRouter()

    query = "Cześć"
    difficulty = router.classify_difficulty(query)

    assert difficulty == DifficultyLevel.EASY


def test_classify_medium_query():
    """Test classification of medium queries."""
    router = ModelRouter()

    query = "Jak działa silnik spalinowy?"
    difficulty = router.classify_difficulty(query)

    assert difficulty == DifficultyLevel.MEDIUM


def test_classify_hard_query_polish():
    """Test classification of hard queries in Polish."""
    router = ModelRouter()

    query = "Wyjaśnij szczegółowo architekturę mikroserwisów i porównaj z monolitem"
    difficulty = router.classify_difficulty(query)

    assert difficulty == DifficultyLevel.HARD


def test_classify_hard_query_english():
    """Test classification of hard queries in English."""
    router = ModelRouter()

    query = "Analyze the complexity of this algorithm and optimize it"
    difficulty = router.classify_difficulty(query)

    assert difficulty == DifficultyLevel.HARD


def test_classify_hard_query_by_length():
    """Test classification based on query length."""
    router = ModelRouter()

    # Very long query should be classified as hard
    query = " ".join(["word"] * 60)
    difficulty = router.classify_difficulty(query)

    assert difficulty == DifficultyLevel.HARD


def test_select_profile_eco_for_easy():
    """Test profile selection for easy queries."""
    router = ModelRouter()

    profile = router.select_profile(DifficultyLevel.EASY)

    assert profile == Profile.ECO


def test_select_profile_smart_for_medium():
    """Test profile selection for medium queries."""
    router = ModelRouter()

    profile = router.select_profile(DifficultyLevel.MEDIUM)

    assert profile == Profile.SMART


def test_select_profile_deep_for_hard_full_user():
    """Test profile selection for hard queries with FULL_ACCESS."""
    router = ModelRouter()

    profile = router.select_profile(DifficultyLevel.HARD, user_role="FULL_ACCESS")

    assert profile == Profile.DEEP


def test_select_profile_smart_for_hard_demo_user():
    """Test profile selection for hard queries with DEMO."""
    router = ModelRouter()

    profile = router.select_profile(DifficultyLevel.HARD, user_role="DEMO")

    # DEMO users get SMART instead of DEEP
    assert profile == Profile.SMART


def test_user_mode_override_eco():
    """Test user mode override to ECO."""
    router = ModelRouter()

    profile = router.select_profile(
        DifficultyLevel.HARD,
        user_mode="eco",
        user_role="FULL_ACCESS",
    )

    assert profile == Profile.ECO


def test_user_mode_override_deep_demo_fallback():
    """Test DEEP mode override for DEMO user falls back to SMART."""
    router = ModelRouter()

    profile = router.select_profile(
        DifficultyLevel.EASY,
        user_mode="deep",
        user_role="DEMO",
    )

    # DEMO cannot use DEEP, should fallback to SMART
    assert profile == Profile.SMART


def test_calculate_smart_credits_tier1():
    """Test smart credits calculation for ≤500 tokens."""
    router = ModelRouter()

    credits = router.calculate_smart_credits(400)

    assert credits == 1


def test_calculate_smart_credits_tier2():
    """Test smart credits calculation for ≤2000 tokens."""
    router = ModelRouter()

    credits = router.calculate_smart_credits(1500)

    assert credits == 2


def test_calculate_smart_credits_tier3():
    """Test smart credits calculation for >2000 tokens."""
    router = ModelRouter()

    credits = router.calculate_smart_credits(3000)

    assert credits == 4


def test_estimate_cost_gemini():
    """Test cost estimation for Gemini."""
    router = ModelRouter()

    estimate = router.estimate_cost(
        profile=Profile.ECO,
        provider="gemini",
        input_tokens=1000,
        output_tokens=500,
    )

    assert estimate.provider == "gemini"
    assert estimate.estimated_cost_usd > 0
    assert estimate.estimated_input_tokens == 1000
    assert estimate.estimated_output_tokens == 500


def test_estimate_cost_free_provider():
    """Test cost estimation for free provider."""
    router = ModelRouter()

    estimate = router.estimate_cost(
        profile=Profile.ECO,
        provider="groq",
        input_tokens=1000,
        output_tokens=500,
    )

    assert estimate.estimated_cost_usd == 0.0


def test_needs_confirmation_deep_full_user():
    """Test confirmation requirement for DEEP mode with FULL_ACCESS."""
    router = ModelRouter()

    needs_confirm = router.needs_confirmation(Profile.DEEP, "FULL_ACCESS")

    assert needs_confirm is True


def test_needs_confirmation_deep_admin():
    """Test no confirmation for ADMIN users."""
    router = ModelRouter()

    needs_confirm = router.needs_confirmation(Profile.DEEP, "ADMIN")

    assert needs_confirm is False


def test_needs_confirmation_eco():
    """Test no confirmation for ECO mode."""
    router = ModelRouter()

    needs_confirm = router.needs_confirmation(Profile.ECO, "FULL_ACCESS")

    assert needs_confirm is False
