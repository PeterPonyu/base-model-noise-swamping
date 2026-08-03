import json
import pathlib
import re
import subprocess

from PIL import Image, ImageChops, ImageStat

ROOT = pathlib.Path(__file__).parents[1]
CANONICAL = pathlib.Path(
    "/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness/results/quant_survival/aggregate/quant_survival_repair_v1.json"
)
EXPECTED_STEMS = {
    "fig01_codec_scope",
    "fig02_efficacy_survival",
    "fig03_rank_survival",
    "fig04_reconstruction_gap",
    "fig06_width_law",
}
EXPECTED_PDFS = {ROOT / "figures" / f"{stem}.pdf" for stem in EXPECTED_STEMS}
EXPECTED_PNGS = {ROOT / "figures-qa" / f"{stem}.png" for stem in EXPECTED_STEMS}
EXPECTED_TEXT = {
    "fig01_codec_scope": ("FP32 edit", "quantize", "efficacy", "out of scope"),
    "fig02_efficacy_survival": ("A", "B", "C", "D", "Absolute efficacy survival", "Conditional on FP32 success", "8-bit survival", "4-bit full-model K1", "0.904 PASS", "0.680 FAIL"),
    "fig03_rank_survival": ("A", "B", "C", "D", "Rank-survival estimands", "within-probe", "edit-level"),
    "fig04_reconstruction_gap": ("A", "B", "C", "Median |dW|/b", "Function-space gap", "Parameter-space gap", "K3 concentration is KILLED"),
    "fig06_width_law": ("A", "B", "C", "D", "Geometry-valid boundary by width", "Within-family ordering", "Old 1B-vs-Mistral confound", "H-Llama holds", "INCONCLUSIVE", "not layers within models"),
}


def command_output(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def assert_exact_files(directory, suffix, expected):
    actual = set(directory.glob(f"*{suffix}"))
    assert actual == expected, f"expected exactly {sorted(map(str, expected))}; got {sorted(map(str, actual))}"


def pdf_dimensions_points(pdf):
    info = command_output("pdfinfo", str(pdf))
    pages = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    size = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+) pts", info, re.MULTILINE)
    assert pages and pages.group(1) == "1", f"{pdf}: not single-page\n{info}"
    assert size, f"{pdf}: missing page dimensions"
    return float(size.group(1)), float(size.group(2))


def assert_vector_pdf(pdf):
    listing = command_output("pdfimages", "-list", str(pdf))
    data_rows = [line for line in listing.splitlines() if re.match(r"^\s*\d+\s+\d+\s+", line)]
    assert not data_rows, f"{pdf}: raster/image objects found\n{listing}"


def assert_pdf_text(pdf):
    text = command_output("pdftotext", "-layout", str(pdf), "-")
    normalized = " ".join(text.split())
    for label in EXPECTED_TEXT[pdf.stem]:
        assert label in normalized, f"{pdf}: missing text {label!r}"


def assert_png(png, pdf_points):
    with Image.open(png) as image:
        assert image.format == "PNG", png
        width, height = image.size
        expected_width = pdf_points[0] / 72 * 200
        expected_height = pdf_points[1] / 72 * 200
        assert abs(width - expected_width) <= 3 and abs(height - expected_height) <= 3, (
            f"{png}: {width}x{height} is not a 200-dpi rendering of {pdf_points} pt"
        )
        rgb = image.convert("RGB")
        white = Image.new("RGB", rgb.size, "white")
        bbox = ImageChops.difference(rgb, white).getbbox()
        assert bbox is not None, f"{png}: blank"
        left, top, right, bottom = bbox
        margins = (left, top, width - right, height - bottom)
        assert min(margins) >= 2, f"{png}: content touches/clips edge; margins={margins}"
        assert max(margins) <= max(width, height) * 0.16, f"{png}: excessive whitespace; margins={margins}"
        crop = rgb.crop(bbox)
        extrema = ImageStat.Stat(crop).extrema
        assert any(low < 245 for low, _ in extrema), f"{png}: non-white extent has no visible content"


canonical = json.loads(CANONICAL.read_text())
assert canonical["module_provenance"]["version"] == "1.2.1"
assert canonical["module_provenance"]["n_boot"] == 500

assert_exact_files(ROOT / "figures", ".pdf", EXPECTED_PDFS)
assert_exact_files(ROOT / "figures-qa", ".png", EXPECTED_PNGS)
for pdf in sorted(EXPECTED_PDFS):
    assert pdf.stat().st_size > 1000, f"{pdf}: blank or truncated"
    dimensions = pdf_dimensions_points(pdf)
    assert_vector_pdf(pdf)
    assert_pdf_text(pdf)
    assert_png(ROOT / "figures-qa" / f"{pdf.stem}.png", dimensions)

print("validation PASS: exactly 5 vector-only single-page PDFs and 5 nonblank 200-dpi QA PNGs; v1.2.1/n_boot=500")
