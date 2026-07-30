from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publication_documents_exist() -> None:
    expected = [
        "README.md",
        "REPRODUCING.md",
        "LICENSE",
        "docs/DATA_DICTIONARY.md",
        "docs/LICENSING.md",
        "docs/decisions/0005-publication-documentation.md",
    ]
    for relative in expected:
        path = ROOT / relative
        assert path.is_file()
        assert path.stat().st_size > 0


def test_readme_links_primary_documents() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "REPRODUCING.md" in text
    assert "docs/DATA_DICTIONARY.md" in text
    assert "docs/LICENSING.md" in text
    assert "make reproduce" in text


def test_reproducing_uses_pinned_pipeline() -> None:
    text = (ROOT / "REPRODUCING.md").read_text(encoding="utf-8")
    assert "make verify-source-snapshots" in text
    assert "make build-inputs" in text
    assert "make reproduce" in text
    assert "37 CSV" in text
    assert "5 PNG" in text
