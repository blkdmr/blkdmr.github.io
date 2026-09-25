# info.db tools

Command-line tools for info.db, installed together as one package.

## Setup

From the `info.db` root:

```bash
conda create -n tools python=3.12
conda activate tools
pip install -e tools
```

Re-run `pip install -e tools` after adding a new command or dependency. Code
changes are picked up without reinstalling.

### System dependencies

`md-to-pdf` needs `pandoc` and `xelatex` on your `PATH`:

```bash
sudo apt install pandoc texlive-xetex
```

`pdf-to-md` needs nothing outside pip. On first run, marker downloads its model
weights (a few GB) from Hugging Face. After that it works offline.

## Commands

### `pdf-to-md`

```bash
pdf-to-md paper.pdf                  # -> paper.md
pdf-to-md paper.pdf notes/paper.md
pdf-to-md paper.pdf notes/           # -> notes/paper.md
```

Images are saved to a `<name>_images/` folder next to the Markdown file and
linked from it.

| Option | Effect |
|---|---|
| `-f`, `--force` | Overwrite the output if it already exists |
| `--no-images` | Skip image extraction |

marker is pinned below 2.0: newer versions need Docker or a separate
`llama-server` binary to run their models.

### `md-to-pdf`

```bash
md-to-pdf notes/paper.md             # -> notes/paper.pdf
md-to-pdf notes/paper.md build/
```

A4 paper, 1.5 cm margins, with `$...$` math, pipe tables and raw HTML. Images
are looked up relative to the Markdown file.

| Option | Effect |
|---|---|
| `-f`, `--force` | Overwrite the output if it already exists |

## Adding a tool

1. Create a package next to `pyproject.toml`, e.g. `my_tool/`, with an
   `__init__.py` and a module that has a `main()` function. Use underscores in
   the folder name, not hyphens.
2. Register the command under `[project.scripts]` in `pyproject.toml`:
   `my-tool = "my_tool.cli:main"`. Add any new libraries to `dependencies`.
3. Run `pip install -e tools` again.
