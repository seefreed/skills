# skills

Help me learn English skills — a small collection of AI skills for translation and media workflows. Each skill is self-contained and documented in its own `SKILL.md` file.

## Skills
- `en-zh-bilingual-json/`: Turn English `.txt` articles into sentence-level `{en, zh}` JSON pairs.
- `en-to-zh-translator/`: Translate English `.txt`/`.md` to Chinese with `_zh` output files.
- `transcript-reflow/`: Clean and reflow transcript text into readable paragraphs.
- `youtube-smart-edit/`: Create YouTube chapter clips and per-chapter subtitles with `yt-dlp` + `ffmpeg`.
- `semantic-video-clipper/`: Segment videos by analyzing subtitle semantics with AI and clipping via FFmpeg.

## Repo Layout
- `test/`: Sample inputs/outputs for translation and JSON generation.
- `AGENTS.md`: Contributor guidelines for this repository.

## Notes
- External dependencies are required for `youtube-smart-edit` (e.g., `yt-dlp`, `ffmpeg`) and `semantic-video-clipper` (e.g., `ffmpeg`).
