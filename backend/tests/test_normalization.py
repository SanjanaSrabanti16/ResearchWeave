import pytest

from app.services.normalization import normalize_arxiv_id, normalize_doi, normalize_title


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
