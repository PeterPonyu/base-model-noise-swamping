#!/usr/bin/env python3
"""
Focused tests for visual_qa.py

Tests:
1. Fails gracefully when PDF not found
2. Processes a minimal fixture PDF
3. Detects source-newer-than-PDF
4. Detects forbidden text patterns
5. Detects missing graphics references
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def temp_package():
    """Create temporary package directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / "test-package"
        pkg_dir.mkdir()
        yield pkg_dir


@pytest.fixture
def minimal_pdf(temp_package):
    """Create minimal PDF fixture using pdflatex."""
    tex_content = r"""
\documentclass{article}
\begin{document}
Hello World
\end{document}
"""
    tex_file = temp_package / "minimal.tex"
    tex_file.write_text(tex_content)

    # Compile to PDF
    subprocess.run(
        ["pdflatex", "-interaction=batchmode", str(tex_file)],
        cwd=temp_package,
        capture_output=True
    )

    pdf_file = temp_package / "minimal.pdf"
    assert pdf_file.exists(), "PDF fixture creation failed"
    return pdf_file


def test_pdf_not_found(temp_package):
    """Test graceful failure when PDF doesn't exist."""
    result = subprocess.run(
        [
            "python",
            "submissions/visual_qa.py",
            str(temp_package),
            str(temp_package / "nonexistent.pdf")
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "Should fail when PDF not found"
    assert "not found" in result.stderr.lower(), "Should report missing PDF"


def test_minimal_fixture(temp_package, minimal_pdf):
    """Test processing a minimal fixture PDF."""
    result = subprocess.run(
        [
            "python",
            "submissions/visual_qa.py",
            str(temp_package),
            str(minimal_pdf),
            "--status", "honest-draft"
        ],
        capture_output=True,
        text=True,
        cwd="."
    )

    assert result.returncode == 0, f"Should succeed on valid PDF\nStderr: {result.stderr}"

    # Check outputs exist
    qa_dir = temp_package / "figures-qa"
    assert qa_dir.exists(), "figures-qa directory should be created"

    manifest_json = qa_dir / "manifest.json"
    assert manifest_json.exists(), "manifest.json should be created"

    manifest_md = qa_dir / "manifest.md"
    assert manifest_md.exists(), "manifest.md should be created"

    # Check at least one page PNG
    page_pngs = list(qa_dir.glob("test-package-main-page-*.png"))
    assert len(page_pngs) >= 1, "At least one page PNG should be generated"

    # Verify JSON structure
    with open(manifest_json) as f:
        data = json.load(f)

    assert data["package"] == "test-package"
    assert data["status"] == "honest-draft"
    assert "checks" in data
    assert "pdf_info" in data["checks"]
    assert "pages" in data["checks"]

    # Check page is non-empty
    pages_check = data["checks"]["pages"]
    assert pages_check["status"] in ["ok", "warning"]
    assert pages_check["total_pages"] >= 1


def test_source_newer_than_pdf(temp_package, minimal_pdf):
    """Test detection of source files newer than PDF."""
    # Touch a .tex file to make it newer
    tex_file = temp_package / "newer.tex"
    tex_file.write_text(r"\documentclass{article}\begin{document}New\end{document}")

    time.sleep(0.1)  # Ensure timestamp difference
    tex_file.touch()

    result = subprocess.run(
        [
            "python",
            "submissions/visual_qa.py",
            str(temp_package),
            str(minimal_pdf)
        ],
        capture_output=True,
        text=True,
        cwd="."
    )

    assert result.returncode == 0

    # Check manifest reports newer source
    manifest_json = temp_package / "figures-qa" / "manifest.json"
    with open(manifest_json) as f:
        data = json.load(f)

    freshness = data["checks"]["source_freshness"]
    assert freshness["status"] == "warning", "Should warn about newer sources"
    assert any("newer.tex" in s for s in freshness["newer_sources"])


def test_forbidden_text_patterns(temp_package):
    """Test detection of forbidden text patterns in PDF."""
    # Create PDF with forbidden patterns
    tex_content = r"""
\documentclass{article}
\begin{document}
This is a TODO item.
Path: /home/user/project/file.txt
Codename: B6 experiment
\end{document}
"""
    tex_file = temp_package / "forbidden.tex"
    tex_file.write_text(tex_content)

    subprocess.run(
        ["pdflatex", "-interaction=batchmode", str(tex_file)],
        cwd=temp_package,
        capture_output=True
    )

    pdf_file = temp_package / "forbidden.pdf"
    assert pdf_file.exists()

    result = subprocess.run(
        [
            "python",
            "submissions/visual_qa.py",
            str(temp_package),
            str(pdf_file)
        ],
        capture_output=True,
        text=True,
        cwd="."
    )

    assert result.returncode == 0

    # Check manifest reports forbidden patterns
    manifest_json = temp_package / "figures-qa" / "manifest.json"
    with open(manifest_json) as f:
        data = json.load(f)

    text_scan = data["checks"]["text_scan"]
    assert text_scan["status"] == "warning", "Should warn about forbidden patterns"
    assert "placeholder" in text_scan["hits"]
    assert "internal_path" in text_scan["hits"]
    assert "codename" in text_scan["hits"]


def test_missing_graphics(temp_package, minimal_pdf):
    """Test detection of missing includegraphics references."""
    # Create .tex with missing graphic reference
    tex_file = temp_package / "main.tex"
    tex_file.write_text(r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\includegraphics{missing-figure}
\end{document}
""")

    result = subprocess.run(
        [
            "python",
            "submissions/visual_qa.py",
            str(temp_package),
            str(minimal_pdf)
        ],
        capture_output=True,
        text=True,
        cwd="."
    )

    assert result.returncode == 0

    # Check manifest reports missing graphic
    manifest_json = temp_package / "figures-qa" / "manifest.json"
    with open(manifest_json) as f:
        data = json.load(f)

    graphics = data["checks"]["graphics"]
    assert graphics["status"] == "warning", "Should warn about missing graphics"
    assert graphics["missing_count"] > 0
    assert any("missing-figure" in m["graphic"] for m in graphics["missing"])


def test_does_not_overwrite_existing(temp_package, minimal_pdf):
    """Test that existing figures-qa files are not overwritten."""
    # Run once
    subprocess.run(
        [
            "python",
            "submissions/visual_qa.py",
            str(temp_package),
            str(minimal_pdf)
        ],
        capture_output=True,
        cwd="."
    )

    # Find generated page PNG
    qa_dir = temp_package / "figures-qa"
    page_pngs = list(qa_dir.glob("test-package-main-page-*.png"))
    assert len(page_pngs) >= 1

    first_png = page_pngs[0]
    original_mtime = first_png.stat().st_mtime

    time.sleep(0.1)

    # Run again
    subprocess.run(
        [
            "python",
            "submissions/visual_qa.py",
            str(temp_package),
            str(minimal_pdf)
        ],
        capture_output=True,
        cwd="."
    )

    # Check that file was not regenerated
    new_mtime = first_png.stat().st_mtime
    assert new_mtime == original_mtime, "Should not overwrite existing page PNG"

    # Check manifest reports as "existing"
    manifest_json = qa_dir / "manifest.json"
    with open(manifest_json) as f:
        data = json.load(f)

    pages = data["checks"]["pages"]
    assert any(p["status"] == "existing" for p in pages["pages"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
