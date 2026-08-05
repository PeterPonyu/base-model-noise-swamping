#!/usr/bin/env python3
"""Validate the final visual-review index and ready-package artifacts."""

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "FINAL-VISUAL-REVIEW-2026-08-04.md"
BANNER = "HONEST-STATE REVIEW DRAFT"

READY = (
    {
        "name": "D2",
        "package": ROOT / "d2-neurocomputing",
        "pdf": ROOT / "d2-neurocomputing" / "main-honest-review.pdf",
        "pages": 35,
        "figures": 5,
        "review_sha": "bd5e2f57382b49953e6a6fb76fe4c7a20e31d63fae17c03fa53b649db8608a1a",
        "page_sheet_sha": "3062ab55ec5e391d9e35b3eb55d6c1f9d669ec44a1f18ce8026a3922aab5451a",
        "figure_sheet_sha": "c6f7b475bc6db16361ba8a421de4e3e60b4cc8f58da9354f0fc1e6ba4b833536",
        "frozen": (
            (
                ROOT / "d2-neurocomputing" / "main.pdf",
                "5c48fa92ec69138da61f29e16ffde68c48bfe375d06b1c4194a8ca61703b9a18",
            ),
        ),
    },
    {
        "name": "B6",
        "package": ROOT / "ieee",
        "pdf": ROOT / "ieee" / "flat" / "main-honest-review.pdf",
        "pages": 14,
        "figures": 7,
        "review_sha": "0858794f832877800ad93d2ed194058a10912919743c2be0f291fb6dc1498d73",
        "page_sheet_sha": "626e57aeaa6c9ef6ca10c1a09f97d482df3dae3facadbfaf9af7bbedfedf15b9",
        "figure_sheet_sha": "2e53354e956ef9be02d93349a67fcf409659ebeb25e1d2a4e0e284075854b6fd",
        "frozen": (
            (
                ROOT / "ieee" / "main-as-submitted.pdf",
                "9fe0eb55adad0bf935db54188ddc8a84440f6df8482de3f3259212830bff5145",
            ),
            (
                ROOT / "ieee" / "flat" / "TETCI_main_manuscript.zip",
                "cad4851f6b792ada599e7c6a38c309ebe0d754763f87e8ac5ce4430c33f9f0c8",
            ),
        ),
    },
    {
        "name": "Frame-A",
        "package": ROOT / "frame-a-eswa",
        "pdf": ROOT / "frame-a-eswa" / "main-honest-review.pdf",
        "pages": 18,
        "figures": 10,
        "review_sha": "538bb8652ade48db4f9822b9751df366c81e8aef29fb1415dc9f85e11790984f",
        "page_sheet_sha": "28c6d2f9de7f99b5f4f9146f17e48dc4e0067e65d8eb73c359cd34ce9fbc5f3e",
        "figure_sheet_sha": "0997c5ed48a13c60920b41917502361d50d417cd42c0db704751e6aff1bf2c6e",
        "frozen": (
            (
                ROOT / "frame-a-eswa" / "main.pdf",
                "a16063536e1a318aee25104c7c6526e2201d1d369a1316cfc9b90c3bbefa673b",
            ),
        ),
    },
    {
        "name": "PaperB",
        "package": ROOT / "paper-b-neurocomputing",
        "pdf": ROOT / "paper-b-neurocomputing" / "main-honest-review.pdf",
        "pages": 29,
        "figures": 5,
        "review_sha": "a29de6a66ee03ac1fe78ad5c71f6cd012344a9b4ac2152536123a33008a52f1e",
        "page_sheet_sha": "017d87187964bb0a371533fc5571e4afa710193d4ef01654bf5740e5033e4e56",
        "figure_sheet_sha": "e6c61e0d40e945bc8061356c698e6670885e56dfc78d89c6c2b18aad2a29b1f8",
        "frozen": (
            (
                ROOT / "paper-b-neurocomputing" / "main.pdf",
                "b79bd033e1da5d1b05df845f9ed014b4aaaed94d5ce8a9f47786753c13c0299a",
            ),
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_text(path: Path, first_page_only: bool = False) -> str:
    command = ["pdftotext"]
    if first_page_only:
        command.extend(["-f", "1", "-l", "1"])
    command.extend([str(path), "-"])
    return subprocess.check_output(command, text=True)


def main() -> None:
    index_text = INDEX.read_text()
    for spec in READY:
        package = spec["package"]
        qa = package / "figures-qa"
        manifest = json.loads((qa / "manifest.json").read_text())
        figure_manifest = json.loads((qa / "figures-manifest.json").read_text())

        assert spec["pdf"].is_file(), spec["pdf"]
        manifest_pdf = Path(manifest["pdf_path"])
        if not manifest_pdf.is_absolute():
            manifest_pdf = ROOT.parent / manifest_pdf
        assert manifest_pdf.resolve() == spec["pdf"].resolve()
        assert manifest["checks"]["pdf_info"]["data"]["sha256"] == spec["review_sha"]
        assert sha256(spec["pdf"]) == spec["review_sha"]
        assert manifest["checks"]["pages"]["total_pages"] == spec["pages"]
        assert len(manifest["checks"]["pages"]["pages"]) == spec["pages"]
        assert not manifest["checks"]["pages"]["empty_pages"]
        assert figure_manifest["count"] == spec["figures"]
        assert len(figure_manifest["figures"]) == spec["figures"]
        assert sha256(qa / "contact-sheet.png") == spec["page_sheet_sha"]
        assert sha256(qa / "figures-contact-sheet.png") == spec["figure_sheet_sha"]
        assert BANNER in pdf_text(spec["pdf"], first_page_only=True)
        assert spec["review_sha"] in index_text
        assert spec["page_sheet_sha"] in index_text
        assert spec["figure_sheet_sha"] in index_text

        for page in manifest["checks"]["pages"]["pages"]:
            raster = qa / page["file"]
            assert raster.is_file() and raster.stat().st_size > 1000, raster
        for figure in figure_manifest["figures"]:
            raster = qa / figure["png"]
            assert raster.is_file() and raster.stat().st_size == figure["bytes"], raster
            assert sha256(raster) == figure["sha256"]

        for frozen_path, frozen_sha in spec["frozen"]:
            assert frozen_path.is_file(), frozen_path
            assert sha256(frozen_path) == frozen_sha
            assert frozen_sha in index_text

        print(f"{spec['name']}: {spec['pages']} pages, {spec['figures']} figures, hashes verified")


if __name__ == "__main__":
    main()
