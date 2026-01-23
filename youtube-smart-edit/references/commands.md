# Command Recipes

Use these as templates and adapt per request.

## Dependency check and install

- Check:

```bash
command -v yt-dlp
command -v ffmpeg
```

- Install (macOS + Homebrew):

```bash
brew install yt-dlp ffmpeg
```

## Download video and subtitles

```bash
yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b" \
  --merge-output-format mp4 \
  --write-subs --sub-langs "en" --sub-format vtt \
  --write-auto-subs --convert-subs vtt \
  --output "%(id)s.%(ext)s" \
  "<VIDEO_URL>"
```

Expected outputs (current directory):
- `<id>.mp4`
- `<id>.en.vtt` (preferred) or `<id>.en.auto.vtt`

## Extract metadata

```bash
yt-dlp --print "%(id)s\n%(title)s\n%(duration)s\n%(uploader)s" "<VIDEO_URL>"
```

## Cut clips

```bash
ffmpeg -hide_banner -y -ss <START> -to <END> -i "<id>.mp4" -c copy "<chapter_title>.mp4"
```

## Convert VTT to SRT

```bash
ffmpeg -hide_banner -y -i "<chapter_title>.vtt" "<chapter_title>.en.srt"
```

## Extract subtitle segment from master VTT

```bash
ffmpeg -hide_banner -y -ss <START> -to <END> -i "<id>.en.vtt" "<chapter_title>.vtt"
```

## Merge bilingual subtitles

Use a line-paired SRT merge script or manual merge to create:
- `<chapter_title>.en-zh.srt`

Keep timecodes from the English SRT and insert Chinese as the next line per cue.
