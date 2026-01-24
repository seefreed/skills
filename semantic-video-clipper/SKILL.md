---
name: semantic-video-clipper
description: Clip long videos into 25-60 second segments using subtitle semantic analysis (not punctuation-only cuts) while preserving complete sentences, and generate matching subtitle clips. Use when asked to segment videos (e.g., .mp4) based on .vtt/.srt subtitles with output filenames as source_basename_index in the same directory.
---

# Semantic Video Clipper

## Overview

Segment a long video into 25-60 second clips by scoring subtitle semantic shifts while enforcing sentence-complete boundaries. Output clips and matching subtitle snippets with filenames `source_basename_<index>.*` in the same directory as the source video.

## Workflow

1. Parse subtitles from .vtt or .srt and compute lightweight semantic vectors.
2. Choose boundary points that align with full sentence endings while maximizing semantic topic shifts.
3. Cut the video and shift subtitle times so each clip starts at 00:00.
4. Save output files as `basename_1.mp4` + `basename_1.vtt` (or .srt), `basename_2.*`, etc.

## Scripted execution

Use `scripts/semantic_clip.py` for deterministic clipping.

```bash
python3 skills/semantic-video-clipper/scripts/semantic_clip.py /path/video.mp4 /path/subtitles.vtt
```

Optional flags:

```bash
--min-seconds 25 --max-seconds 60 --dry-run
```

Notes:
- Sentence completeness uses punctuation only as a guardrail; clip boundaries are chosen by semantic similarity scoring (TF-IDF cosine) rather than punctuation-only slicing.
- Output files always land beside the source video, named with an underscore plus 1-based index.
