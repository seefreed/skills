# Repository Guidelines

## Project Structure & Module Organization
- `en-zh-bilingual-json/`, `en-to-zh-translator/`, `transcript-reflow/`, and `youtube-smart-edit/` each define a skill. The primary documentation for each skill is its `SKILL.md` file.
- `youtube-smart-edit/scripts/` contains executable Python utilities (e.g., `smart_edit.py`).
- `youtube-smart-edit/references/` holds command templates and reference notes.
- `test/` contains sample input/output files for translation and bilingual JSON workflows.
- `README.md` is minimal; treat `SKILL.md` files as the source of truth.

## Build, Test, and Development Commands
This repo is documentation-first with a small Python utility. There is no build system.
- Create a venv for the YouTube workflow: `python3 -m venv .venv` and `source .venv/bin/activate`.
- View script usage: `python youtube-smart-edit/scripts/smart_edit.py --help`.
- Optional smoke check: `python -m py_compile youtube-smart-edit/scripts/smart_edit.py`.

## Coding Style & Naming Conventions
- Python follows standard 4-space indentation and PEP 8-friendly naming (`snake_case` functions, `CamelCase` classes).
- Markdown uses clear section headings and numbered workflows; keep line length readable.
- File naming: skill folders are kebab-case; outputs in the skills often use suffixes like `_zh` or numbered chapter prefixes (e.g., `01_title.mp4`).

## Testing Guidelines
- There is no automated test framework configured.
- Use the `test/` fixtures as manual validation inputs/outputs when changing translation or JSON workflows.
- For `smart_edit.py`, prefer running a dry help check and a small local sample if available.

## Commit & Pull Request Guidelines
- Git history uses terse numeric commit messages (e.g., `13`, `12`). If maintaining the existing style, keep messages short; otherwise, adopt descriptive messages consistently and update history going forward.
- PRs should include: summary of the skill change, affected directories, and any example input/output filenames (e.g., files in `test/`).

## Configuration & Dependencies
- `youtube-smart-edit` relies on external tools (`yt-dlp`, `ffmpeg`). Document any new dependencies in its `SKILL.md` and include basic install notes.
