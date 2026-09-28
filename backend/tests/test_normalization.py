import pytest

from app.services.normalization import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_metadata_text,
    normalize_title,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" HTTPS://doi.org/10.1000/ABC ", "10.1000/abc"),
        ("http://doi.org/10.5555/X.Y", "10.5555/x.y"),
        ("doi: 10.1/Test", "10.1/test"),
        (None, None),
    ],
)
def test_normalize_doi(raw: str | None, expected: str | None) -> None:
    assert normalize_doi(raw) == expected


def test_normalize_title_handles_unicode_punctuation_and_spaces() -> None:
    assert normalize_title("  Agent–Based: VISUALIZATION!  ") == "agent based visualization"


def test_normalize_arxiv_id_removes_version_and_url() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.01234v2") == "2401.01234"


@pytest.mark.parametrize(
    "raw",
    [
        "10.48550/arXiv.2508.07496",
        "https://doi.org/10.48550/arXiv.2508.07496v2",
    ],
)
def test_normalize_arxiv_id_from_external_arxiv_doi(raw: str) -> None:
    assert normalize_arxiv_id(raw) == "2508.07496"


@pytest.mark.parametrize(
    "raw",
    [
        "2401.01234",
        "arXiv:2401.01234v3",
        "/abs/2401.01234v2",
        "/pdf/2401.01234v1.pdf",
        "https://arxiv.org/pdf/2401.01234v4.pdf?download=1",
    ],
)
def test_normalize_arxiv_id_equivalent_forms(raw: str) -> None:
    assert normalize_arxiv_id(raw) == "2401.01234"


def test_normalize_arxiv_id_rejects_non_arxiv_url() -> None:
    assert normalize_arxiv_id("https://example.org/abs/2401.01234") is None


def test_normalize_doi_handles_nested_prefixes() -> None:
    assert normalize_doi("doi: https://doi.org/10.1000/ABC") == "10.1000/abc"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("An <italic>important</italic> result.", "An important result."),
        ("<p>Nested <bold>provider <italic>markup</italic></bold>.</p>", "Nested provider markup."),
        (
            "Score <inline-formula><tex-math><![CDATA[x^2 + y^2]]></tex-math></inline-formula>",
            "Score x^2 + y^2",
        ),
        ("Accuracy &gt; 90% &amp; stable.", "Accuracy > 90% & stable."),
        ("Already plain metadata.", "Already plain metadata."),
    ],
)
def test_normalize_metadata_text_removes_markup_and_preserves_content(
    raw: str, expected: str
) -> None:
    assert normalize_metadata_text(raw) == expected


def test_normalize_metadata_text_tolerates_malformed_markup() -> None:
    result = normalize_metadata_text("A <italic>recoverable result</italic><broken")
    assert result is not None
    assert "recoverable result" in result
