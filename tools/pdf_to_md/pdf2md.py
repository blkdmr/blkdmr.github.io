import argparse
import os
import sys
from pathlib import Path


def images_dir_for(md_path: Path) -> Path:
    # No spaces: they would break the relative links in the Markdown.
    return md_path.with_name(md_path.stem.replace(" ", "_") + "_images")


def convert(pdf_path: Path, md_path: Path, extract_images: bool = True) -> None:
    # Imported lazily: marker loads heavy ML dependencies.
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"disable_image_extraction": not extract_images},
    )
    rendered = converter(str(pdf_path))
    markdown, _, images = text_from_rendered(rendered)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    if images:
        # Keep each document's images in their own folder, so converting
        # several PDFs into one directory doesn't mix up (or overwrite) them.
        images_dir = images_dir_for(md_path)
        images_dir.mkdir(exist_ok=True)
        for name, image in images.items():
            if Path(name).suffix.lower() in (".jpg", ".jpeg") and image.mode != "RGB":
                image = image.convert("RGB")
            image.save(images_dir / name)
            markdown = markdown.replace(f"]({name})", f"]({images_dir.name}/{name})")
        print(f"Saved {len(images)} image(s) to {images_dir}")

    md_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown.")
    parser.add_argument("input", type=Path, help="source .pdf file")
    parser.add_argument(
        "output",
        nargs="?",
        help="destination .md file (default: input with .md suffix)",
    )
    parser.add_argument("-f", "--force", action="store_true", help="overwrite existing output")
    parser.add_argument(
        "--no-images", action="store_true", help="don't extract images from the PDF"
    )
    args = parser.parse_args()

    pdf_path: Path = args.input
    md_path = Path(args.output) if args.output else pdf_path.with_suffix(".md")
    # A trailing slash means "put it in this folder", even if it doesn't exist yet.
    if md_path.is_dir() or (args.output or "").endswith(("/", os.sep)):
        md_path = md_path / pdf_path.with_suffix(".md").name

    if not pdf_path.is_file():
        sys.exit(f"error: {pdf_path} not found")
    if md_path.exists() and not args.force:
        sys.exit(f"error: {md_path} already exists (use -f to overwrite)")

    print(f"Converting {pdf_path} -> {md_path}")
    convert(pdf_path, md_path, extract_images=not args.no_images)


if __name__ == "__main__":
    main()
