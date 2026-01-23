---
name: youtube-smart-edit
description: Create a YouTube smart clipping workflow with yt-dlp and ffmpeg. Use when asked to download YouTube videos/subtitles, generate fine-grained chapters, cut precise clips, or generate per-chapter English SRTs.
---

# YouTube Smart Edit

## Overview

Generate 2-4 minute chapter clips from a YouTube video by downloading MP4 + English subtitles, segmenting content, cutting precise clips, and producing per-chapter English SRTs.

## Workflow

### 0) Use the automation script to reduce tokens

- Prefer `scripts/smart_edit.py` for end-to-end runs (download, chaptering, clip cut, subtitle slicing).
- The script uses heuristic chaptering to avoid AI token usage.
- Create and use a local venv (no external packages required):
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python scripts/smart_edit.py --help`

### 1) Confirm inputs and environment

- Ask for the YouTube URL and whether English subtitles are available (manual preferred; auto as fallback).
- Check tools: `yt-dlp` and `ffmpeg`. If missing, install before proceeding.
- Use command templates in `references/commands.md`.

### 2) Download source video and subtitles

- Check current directory for existing source files before downloading:
  - If `<id>.mp4` and `<id>.en.vtt` already exist, skip yt-dlp download.
- Download highest 1080p MP4 and English VTT. Save in current directory with ID-based names:
  - `<id>.mp4`
  - `<id>.en.vtt` (or `<id>.en.auto.vtt` if manual subs absent)
- Also capture video metadata (id, title, duration, uploader) for reporting.
  - The script handles this when `--url` is provided.

### 3) Prepare output directory

- Create output directory using the original video title:
  - Replace spaces with underscores.
  - Remove/replace filesystem-unsafe characters.
- Place all chapter clips and subtitle files into this directory.

### 4) Generate fine-grained chapters (2-4 minutes)

- Parse the English VTT and draft chapter boundaries based on topic changes and sentence boundaries.
- Target 2-4 minutes per chapter; avoid cutting mid-sentence.
- Prefer semantic breaks (new concept, example, recap) over strict timing.
- Produce a chapter list with:
  - `title`, `start`, `end`, `reason`
  - The script uses sentence-boundary heuristics with `--min-seconds`, `--target-seconds`, `--max-seconds`.

### 5) Cut precise clips

- Use ffmpeg with accurate trimming and stable outputs. Always re-encode:
  - Place `-ss` after `-i` for accurate seeking.
  - Use `libx264` + `aac`, `-movflags +faststart`, and `-pix_fmt yuv420p` to maximize player compatibility.
  - Use a fast preset (e.g., `-preset veryfast`) to avoid long encodes and timeouts.
- Run clips serially and avoid external timeouts that kill ffmpeg mid-write.
- After each clip, validate with `ffprobe`; retry once if validation fails.
- Name each clip with an ordered prefix: `<nn>_<chapter_title>.mp4` using safe filenames:
  - Use a 2-digit index starting at 01.
  - Replace spaces with underscores.
  - Remove filesystem-unsafe characters.

### 6) Extract and convert subtitles per chapter

- Extract VTT segment for each chapter by time range.
- Convert each segment to SRT:
  - `<nn>_<chapter_title>.en.srt`
  - The script deletes per-chapter VTT unless `--keep-vtt` is set.

### 7) Report outputs

- Print output directory path, chapter list, and generated files.

## Output Rules

- Source files stay in current directory (`<id>.mp4`, `<id>.en.vtt`).
- All chapter clips and subtitle files are placed in the per-video directory named after the sanitized title.
- Use consistent time formats (`HH:MM:SS.mmm`).

## References

- Command templates and copy/paste examples: `references/commands.md`
- Automation: `scripts/smart_edit.py`
