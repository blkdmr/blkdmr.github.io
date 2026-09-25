import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def convert(md_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pandoc",
            str(md_path),
            "-f", "markdown+tex_math_dollars+pipe_tables+raw_html",
            "-o", str(pdf_path),
            "--pdf-engine=xelatex",
            "--standalone",
            # Resolve relative image paths against the Markdown file's folder.
            f"--resource-path={md_path.parent}",
            "-V", "papersize:a4",
            "-V", "geometry:margin=1.5cm",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a Markdown file to PDF.")
    parser.add_argument("input", type=Path, help="source .md file")
    parser.add_argument(
        "output",
        nargs="?",
        help="destination .pdf file (default: input with .pdf suffix)",
    )
    parser.add_argument("-f", "--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args()

    md_path: Path = args.input
    pdf_path = Path(args.output) if args.output else md_path.with_suffix(".pdf")
    # A trailing slash means "put it in this folder", even if it doesn't exist yet.
    if pdf_path.is_dir() or (args.output or "").endswith(("/", os.sep)):
        pdf_path = pdf_path / md_path.with_suffix(".pdf").name

    if not md_path.is_file():
        sys.exit(f"error: {md_path} not found")
    if pdf_path.exists() and not args.force:
        sys.exit(f"error: {pdf_path} already exists (use -f to overwrite)")
    for tool in ("pandoc", "xelatex"):
        if shutil.which(tool) is None:
            sys.exit(f"error: '{tool}' not found on PATH")

    print(f"Converting {md_path} -> {pdf_path}")
    try:
        convert(md_path, pdf_path)
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: pandoc failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()
