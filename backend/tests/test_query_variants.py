from app.services.query_variants import QueryVariantService


def test_generic_variants_are_deterministic_and_bounded() -> None:
    service = QueryVariantService()
    expected = [
        "AI agents for visualizations",
        "AI agent visualization",
        "AI visualization",
        "artificial intelligence visualization",
    ]
    assert service.generate("  AI agents  for   visualizations ") == expected
    assert service.generate("AI agents for visualizations") == expected
    assert len(expected) <= 4


def test_variants_do_not_require_ai_or_a_specific_topic() -> None:
    variants = QueryVariantService().generate("Robots in hospitals for surgeries")
    assert variants[0] == "Robots in hospitals for surgeries"
    assert "Robot hospital surgery" in variants
    assert "Robot surgery" in variants
    assert len(variants) <= 4


def test_short_query_does_not_generate_duplicate_variants() -> None:
    assert QueryVariantService().generate("AI visualization") == [
        "AI visualization",
        "artificial intelligence visualization",
    ]
