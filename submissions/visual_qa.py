#!/usr/bin/env python3
"""
Unified paper visual QA tool for multi-package submissions.

Usage:
    python visual_qa.py <package_dir> <pdf_path> [--status candidate|honest-draft|incomplete]

Generates:
    - figures-qa/<package>-main-page-###.png (200dpi per-page rasters)
    - figures-qa/contact-sheet.png (grid overview)
    - figures-qa/manifest.json (machine-readable QA report)
    - figures-qa/manifest.md (human-readable QA report)

Does NOT:
    - Clear existing figures-qa contents
    - Overwrite standalone figure QA files
    - Modify source files
    - Build LaTeX
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PaperQA:
    """Paper visual quality assurance."""

    def __init__(self, package_dir: Path, pdf_path: Path, status: Optional[str] = None):
        self.package_dir = package_dir.resolve()
        self.pdf_path = pdf_path.resolve()
        self.status = status
        self.package_name = self.package_dir.name
        self.qa_dir = self.package_dir / "figures-qa"
        self.report = {
            "package": self.package_name,
            "pdf_path": str(pdf_path),
            "status": status,
            "checks": {}
        }

    def run(self) -> Dict:
        """Execute all QA checks."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        # Ensure QA directory exists (don't clear it)
        self.qa_dir.mkdir(exist_ok=True)

        print(f"Running QA for {self.package_name}...")

        # Run all checks
        self.report["checks"]["pdf_info"] = self._check_pdf_info()
        self.report["checks"]["pdf_fonts"] = self._check_pdf_fonts()
        self.report["checks"]["text_scan"] = self._check_text_content()
        self.report["checks"]["latex"] = self._check_latex()
        self.report["checks"]["source_freshness"] = self._check_source_freshness()
        self.report["checks"]["graphics"] = self._check_graphics()

        # Generate page rasters
        self.report["checks"]["pages"] = self._generate_page_rasters()

        # Generate contact sheet
        self._generate_contact_sheet()

        # Write reports
        self._write_json_report()
        self._write_markdown_report()

        return self.report

    def _check_pdf_info(self) -> Dict:
        """Extract PDF metadata with pdfinfo."""
        try:
            result = subprocess.run(
                ["pdfinfo", str(self.pdf_path)],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split("\n")
            info = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    info[key.strip()] = value.strip()

            # Compute SHA256
            sha256 = hashlib.sha256()
            with open(self.pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            info["sha256"] = sha256.hexdigest()

            return {"status": "ok", "data": info}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": str(e)}

    def _check_pdf_fonts(self) -> Dict:
        """Check font embedding with pdffonts."""
        try:
            result = subprocess.run(
                ["pdffonts", str(self.pdf_path)],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) < 3:
                return {"status": "ok", "embedded_count": 0, "fonts": []}

            # Parse header and font rows
            fonts = []
            for line in lines[2:]:  # Skip header rows
                parts = line.split()
                if len(parts) >= 4:
                    fonts.append({
                        "name": parts[0],
                        "type": parts[1] if len(parts) > 1 else "",
                        "embedded": "yes" in line.lower()
                    })

            embedded_count = sum(1 for f in fonts if f["embedded"])
            all_embedded = len(fonts) == embedded_count

            return {
                "status": "ok",
                "total_fonts": len(fonts),
                "embedded_count": embedded_count,
                "all_embedded": all_embedded,
                "fonts": fonts
            }
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": str(e)}

    def _check_text_content(self) -> Dict:
        """Scan PDF text for forbidden patterns."""
        try:
            result = subprocess.run(
                ["pdftotext", str(self.pdf_path), "-"],
                capture_output=True,
                text=True,
                check=True
            )
            text = result.stdout

            # Forbidden patterns
            patterns = {
                "placeholder": r"(?i)(TODO|FIXME|XXX|PLACEHOLDER|\?\?\?)",
                "internal_path": r"(/home/|/Users/|C:\\|edit-harness/|branches/|\.omc/)",
                "codename": r"\b(B6|E[0-9]|D[0-9]|G[0-9]|H[0-9]|P[0-9]|MIX_[ABC]|run_[a-z0-9]+|ROME|MEMIT|AlphaEdit)\b"
            }

            hits = {}
            for name, pattern in patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    hits[name] = {
                        "count": len(matches),
                        "samples": list(set(matches))[:5]  # First 5 unique
                    }

            return {
                "status": "warning" if hits else "ok",
                "hits": hits
            }
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": str(e)}

    def _check_latex(self) -> Dict:
        """Check LaTeX build logs for errors/warnings."""
        # Look for .log file
        log_paths = list(self.package_dir.glob("*.log"))
        if not log_paths:
            # Try flat/ subdirectory for ieee
            flat_dir = self.package_dir / "flat"
            if flat_dir.exists():
                log_paths = list(flat_dir.glob("*.log"))

        if not log_paths:
            return {"status": "skip", "message": "No .log file found"}

        log_path = log_paths[0]  # Use first log found

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read()

            errors = re.findall(r"^! (.+)$", log_content, re.MULTILINE)
            overfull = re.findall(r"Overfull \\[hv]box", log_content)
            undefined = re.findall(r"undefined", log_content, re.IGNORECASE)

            return {
                "status": "ok",
                "log_file": str(log_path.relative_to(self.package_dir)),
                "errors": len(errors),
                "overfull": len(overfull),
                "undefined": len(undefined),
                "error_samples": errors[:3]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _check_source_freshness(self) -> Dict:
        """Check if source files are newer than PDF."""
        try:
            pdf_mtime = self.pdf_path.stat().st_mtime

            # Check main.tex and other .tex files
            tex_files = list(self.package_dir.glob("*.tex"))
            sections_dir = self.package_dir / "sections"
            if sections_dir.exists():
                tex_files.extend(sections_dir.glob("*.tex"))

            newer_files = []
            for tex_file in tex_files:
                if tex_file.stat().st_mtime > pdf_mtime:
                    newer_files.append(str(tex_file.relative_to(self.package_dir)))

            return {
                "status": "warning" if newer_files else "ok",
                "pdf_mtime": pdf_mtime,
                "newer_sources": newer_files
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _check_graphics(self) -> Dict:
        """Check for missing includegraphics references."""
        try:
            # Find all .tex files
            tex_files = list(self.package_dir.glob("*.tex"))
            sections_dir = self.package_dir / "sections"
            if sections_dir.exists():
                tex_files.extend(sections_dir.glob("*.tex"))

            missing = []
            for tex_file in tex_files:
                with open(tex_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Find includegraphics references
                graphics = re.findall(
                    r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                    content
                )

                for graphic in graphics:
                    # Try common extensions
                    for ext in ["", ".png", ".pdf", ".jpg", ".eps"]:
                        graphic_path = self.package_dir / f"{graphic}{ext}"
                        if graphic_path.exists():
                            break
                    else:
                        # Also check figures/, figures-r/, figures-src/
                        for subdir in ["figures", "figures-r", "figures-src", "figures-tex"]:
                            for ext in ["", ".png", ".pdf", ".jpg", ".eps"]:
                                graphic_path = self.package_dir / subdir / f"{Path(graphic).name}{ext}"
                                if graphic_path.exists():
                                    break
                            else:
                                continue
                            break
                        else:
                            missing.append({
                                "file": str(tex_file.relative_to(self.package_dir)),
                                "graphic": graphic
                            })

            return {
                "status": "warning" if missing else "ok",
                "missing_count": len(missing),
                "missing": missing[:10]  # First 10
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _generate_page_rasters(self) -> Dict:
        """Generate per-page PNG rasters at 200dpi."""
        try:
            # Get page count from pdfinfo
            page_count = int(
                self.report["checks"]["pdf_info"]["data"].get("Pages", 0)
            )

            if page_count == 0:
                return {"status": "error", "message": "Could not determine page count"}

            pages = []
            for page_num in range(1, page_count + 1):
                output_file = self.qa_dir / f"{self.package_name}-main-page-{page_num:03d}.png"

                # Skip if already exists (don't overwrite)
                if output_file.exists():
                    # Check if non-empty
                    non_empty = output_file.stat().st_size > 1000
                    pages.append({
                        "page": page_num,
                        "file": output_file.name,
                        "status": "existing",
                        "non_empty": non_empty
                    })
                    continue

                # Generate PNG with pdftoppm
                subprocess.run(
                    [
                        "pdftoppm",
                        "-png",
                        "-r", "200",
                        "-f", str(page_num),
                        "-l", str(page_num),
                        str(self.pdf_path),
                        str(output_file.with_suffix(""))
                    ],
                    check=True,
                    capture_output=True
                )

                # pdftoppm adds suffix, find and rename it
                # Single-page output: -1, multi-page varies
                for suffix in ["-1", f"-{page_num}"]:
                    generated_file = output_file.parent / f"{output_file.stem}{suffix}.png"
                    if generated_file.exists():
                        generated_file.rename(output_file)
                        break

                # Check if non-empty
                non_empty = output_file.stat().st_size > 1000

                pages.append({
                    "page": page_num,
                    "file": output_file.name,
                    "status": "generated",
                    "non_empty": non_empty
                })

            empty_pages = [p for p in pages if not p["non_empty"]]

            return {
                "status": "warning" if empty_pages else "ok",
                "total_pages": page_count,
                "generated": sum(1 for p in pages if p["status"] == "generated"),
                "existing": sum(1 for p in pages if p["status"] == "existing"),
                "empty_pages": [p["page"] for p in empty_pages],
                "pages": pages
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _generate_contact_sheet(self):
        """Generate contact sheet (grid overview) using ImageMagick montage."""
        try:
            # Find all generated page PNGs
            page_files = sorted(self.qa_dir.glob(f"{self.package_name}-main-page-*.png"))

            if not page_files:
                print("No page PNGs found for contact sheet")
                return

            output_file = self.qa_dir / "contact-sheet.png"

            # Use montage to create grid
            subprocess.run(
                [
                    "montage",
                    *[str(f) for f in page_files],
                    "-tile", "4x",
                    "-geometry", "200x200+2+2",
                    "-background", "white",
                    str(output_file)
                ],
                check=True,
                capture_output=True
            )

            print(f"Contact sheet: {output_file}")
        except FileNotFoundError:
            print("ImageMagick montage not available, skipping contact sheet")
        except Exception as e:
            print(f"Contact sheet generation failed: {e}")

    def _write_json_report(self):
        """Write JSON manifest."""
        output_file = self.qa_dir / "manifest.json"
        with open(output_file, "w") as f:
            json.dump(self.report, f, indent=2)
        print(f"JSON report: {output_file}")

    def _write_markdown_report(self):
        """Write human-readable Markdown manifest."""
        output_file = self.qa_dir / "manifest.md"

        lines = [
            f"# Visual QA Manifest: {self.package_name}",
            "",
            f"**PDF:** `{self.pdf_path.name}`",
            f"**Status:** {self.status or 'not specified'}",
            f"**Generated:** {self.report.get('timestamp', 'unknown')}",
            "",
            "## PDF Information",
            ""
        ]

        # PDF info
        pdf_info = self.report["checks"]["pdf_info"]
        if pdf_info["status"] == "ok":
            data = pdf_info["data"]
            lines.extend([
                f"- **Pages:** {data.get('Pages', 'unknown')}",
                f"- **Page size:** {data.get('Page size', 'unknown')}",
                f"- **SHA256:** `{data.get('sha256', 'unknown')[:16]}...`",
                ""
            ])

        # Fonts
        lines.append("## Font Embedding")
        lines.append("")
        fonts = self.report["checks"]["pdf_fonts"]
        if fonts["status"] == "ok":
            lines.append(f"- **Total fonts:** {fonts['total_fonts']}")
            lines.append(f"- **Embedded:** {fonts['embedded_count']}")
            lines.append(f"- **All embedded:** {'✓' if fonts['all_embedded'] else '✗'}")
            lines.append("")

        # Text scan
        lines.append("## Text Content Scan")
        lines.append("")
        text_scan = self.report["checks"]["text_scan"]
        if text_scan["status"] == "ok":
            lines.append("✓ No forbidden patterns detected")
        elif text_scan["hits"]:
            lines.append("⚠ **Forbidden patterns detected:**")
            for name, hit in text_scan["hits"].items():
                lines.append(f"- {name}: {hit['count']} hits")
                for sample in hit["samples"]:
                    lines.append(f"  - `{sample}`")
        lines.append("")

        # LaTeX
        lines.append("## LaTeX Build")
        lines.append("")
        latex = self.report["checks"]["latex"]
        if latex["status"] == "ok":
            lines.append(f"- **Log file:** `{latex['log_file']}`")
            lines.append(f"- **Errors:** {latex['errors']}")
            lines.append(f"- **Overfull:** {latex['overfull']}")
            lines.append(f"- **Undefined:** {latex['undefined']}")
        else:
            lines.append(f"Status: {latex['status']}")
        lines.append("")

        # Source freshness
        lines.append("## Source Freshness")
        lines.append("")
        freshness = self.report["checks"]["source_freshness"]
        if freshness["status"] == "ok":
            lines.append("✓ PDF is up-to-date with sources")
        elif freshness.get("newer_sources"):
            lines.append("⚠ **Sources newer than PDF:**")
            for source in freshness["newer_sources"]:
                lines.append(f"- `{source}`")
        lines.append("")

        # Graphics
        lines.append("## Graphics References")
        lines.append("")
        graphics = self.report["checks"]["graphics"]
        if graphics["status"] == "ok":
            lines.append("✓ All graphics found")
        elif graphics.get("missing"):
            lines.append(f"⚠ **Missing graphics:** {graphics['missing_count']}")
            for item in graphics["missing"]:
                lines.append(f"- `{item['file']}`: `{item['graphic']}`")
        lines.append("")

        # Pages
        lines.append("## Page Rasters")
        lines.append("")
        pages = self.report["checks"]["pages"]
        if pages["status"] == "ok" or pages["status"] == "warning":
            lines.append(f"- **Total pages:** {pages['total_pages']}")
            lines.append(f"- **Generated:** {pages['generated']}")
            lines.append(f"- **Existing:** {pages['existing']}")
            if pages.get("empty_pages"):
                lines.append(f"- ⚠ **Empty pages:** {pages['empty_pages']}")
        lines.append("")

        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"Markdown report: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified paper visual QA tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("package_dir", type=Path, help="Package directory")
    parser.add_argument("pdf_path", type=Path, help="PDF file path")
    parser.add_argument(
        "--status",
        choices=["candidate", "honest-draft", "incomplete"],
        help="Scientific status (optional)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.package_dir.is_dir():
        print(f"Error: Package directory not found: {args.package_dir}", file=sys.stderr)
        return 1

    if not args.pdf_path.is_file():
        print(f"Error: PDF file not found: {args.pdf_path}", file=sys.stderr)
        return 1

    # Run QA
    qa = PaperQA(args.package_dir, args.pdf_path, args.status)
    import datetime
    qa.report["timestamp"] = datetime.datetime.now().isoformat()

    try:
        qa.run()
        print("\n✓ Visual QA complete")
        return 0
    except Exception as e:
        print(f"\n✗ Visual QA failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
