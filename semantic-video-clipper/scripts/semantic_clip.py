#!/usr/bin/env python3
"""Clip a long video into 25-60s segments using subtitle semantics."""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class Cue:
    start: float
    end: float
    lines: List[str]

    @property
    def text(self) -> str:
        return " ".join(line.strip() for line in self.lines).strip()


def parse_timestamp(raw: str) -> float:
    raw = raw.strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        raise ValueError(f"Unrecognized timestamp: {raw}")
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float, style: str) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs_ms = total_ms % 60_000
    secs = secs_ms / 1000.0
    stamp = f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    if style == "srt":
        return stamp.replace(".", ",")
    return stamp


def parse_time_line(line: str) -> Tuple[str, str]:
    parts = line.split("-->")
    if len(parts) < 2:
        raise ValueError(f"Invalid time line: {line}")
    start = parts[0].strip()
    end = parts[1].strip().split()[0]
    return start, end


def parse_vtt(lines: List[str]) -> List[Cue]:
    cues: List[Cue] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip("\ufeff").strip()
        if not line:
            i += 1
            continue
        if line.startswith("WEBVTT"):
            i += 1
            continue
        if line.startswith("NOTE") or line.startswith("STYLE"):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        if "-->" not in line:
            i += 1
            if i >= len(lines):
                break
            line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue
        start_raw, end_raw = parse_time_line(line)
        start = parse_timestamp(start_raw)
        end = parse_timestamp(end_raw)
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].rstrip("\n"))
            i += 1
        cues.append(Cue(start=start, end=end, lines=text_lines))
        while i < len(lines) and not lines[i].strip():
            i += 1
    return cues


def parse_srt(lines: List[str]) -> List[Cue]:
    cues: List[Cue] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip("\ufeff").strip()
        if not line:
            i += 1
            continue
        if line.isdigit():
            i += 1
            if i >= len(lines):
                break
            line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue
        start_raw, end_raw = parse_time_line(line)
        start = parse_timestamp(start_raw)
        end = parse_timestamp(end_raw)
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].rstrip("\n"))
            i += 1
        cues.append(Cue(start=start, end=end, lines=text_lines))
        while i < len(lines) and not lines[i].strip():
            i += 1
    return cues


def load_subtitles(path: str) -> Tuple[List[Cue], str]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if ext == ".vtt":
        return parse_vtt(lines), "vtt"
    if ext == ".srt":
        return parse_srt(lines), "srt"
    raise ValueError(f"Unsupported subtitle format: {ext}")


def write_vtt(cues: Iterable[Cue], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("WEBVTT\n\n")
        for cue in cues:
            handle.write(
                f"{format_timestamp(cue.start, 'vtt')} --> {format_timestamp(cue.end, 'vtt')}\n"
            )
            for line in cue.lines:
                handle.write(f"{line}\n")
            handle.write("\n")


def write_srt(cues: Iterable[Cue], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for idx, cue in enumerate(cues, start=1):
            handle.write(f"{idx}\n")
            handle.write(
                f"{format_timestamp(cue.start, 'srt')} --> {format_timestamp(cue.end, 'srt')}\n"
            )
            for line in cue.lines:
                handle.write(f"{line}\n")
            handle.write("\n")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def build_tfidf_vectors(texts: List[str]) -> List[dict]:
    token_lists = [tokenize(text) for text in texts]
    doc_count = len(token_lists)
    df = {}
    for tokens in token_lists:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    idf = {token: math.log((doc_count + 1) / (freq + 1)) + 1.0 for token, freq in df.items()}
    vectors: List[dict] = []
    for tokens in token_lists:
        if not tokens:
            vectors.append({})
            continue
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total = float(len(tokens))
        vec = {token: (count / total) * idf[token] for token, count in tf.items()}
        vectors.append(vec)
    return vectors


def average_vectors(vectors: List[dict]) -> dict:
    if not vectors:
        return {}
    sums = {}
    for vec in vectors:
        for token, value in vec.items():
            sums[token] = sums.get(token, 0.0) + value
    count = float(len(vectors))
    return {token: value / count for token, value in sums.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    for token, value in vec_a.items():
        dot += value * vec_b.get(token, 0.0)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_boundary_costs(vectors: List[dict], window: int = 3) -> dict:
    costs = {0: 0.0, len(vectors): 0.0}
    for pos in range(1, len(vectors)):
        left_vecs = vectors[max(0, pos - window) : pos]
        right_vecs = vectors[pos : min(len(vectors), pos + window)]
        left_avg = average_vectors(left_vecs)
        right_avg = average_vectors(right_vecs)
        costs[pos] = cosine_similarity(left_avg, right_avg)
    return costs


def is_sentence_end(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(re.search(r"[.!?。！？…][\"']?$", stripped))


def segment_cues(
    cues: List[Cue],
    min_seconds: float,
    max_seconds: float,
) -> List[Tuple[int, int]]:
    if not cues:
        return []
    vectors = build_tfidf_vectors([cue.text for cue in cues])
    boundary_costs = compute_boundary_costs(vectors)
    sentence_end_positions = {i + 1 for i, cue in enumerate(cues) if is_sentence_end(cue.text)}
    sentence_end_positions.add(len(cues))
    positions = [0] + sorted(sentence_end_positions)
    inf = 1e12
    dp = [inf] * len(positions)
    prev = [-1] * len(positions)
    dp[0] = 0.0
    for i, start_pos in enumerate(positions):
        if dp[i] >= inf:
            continue
        start_time = cues[start_pos].start if start_pos < len(cues) else cues[-1].end
        for j in range(i + 1, len(positions)):
            end_pos = positions[j]
            end_time = cues[end_pos - 1].end
            duration = end_time - start_time
            if duration < min_seconds:
                continue
            if duration > max_seconds:
                break
            boundary_cost = 0.0 if end_pos == len(cues) else boundary_costs.get(end_pos, 0.0)
            cost = dp[i] + boundary_cost
            if cost < dp[j]:
                dp[j] = cost
                prev[j] = i
    if dp[-1] >= inf:
        raise RuntimeError("Unable to segment subtitles within duration constraints.")
    segments: List[Tuple[int, int]] = []
    idx = len(positions) - 1
    while idx > 0:
        prev_idx = prev[idx]
        if prev_idx < 0:
            break
        segments.append((positions[prev_idx], positions[idx]))
        idx = prev_idx
    segments.reverse()
    return segments


def shift_cues(cues: List[Cue], offset: float) -> List[Cue]:
    shifted = []
    for cue in cues:
        shifted.append(Cue(start=cue.start - offset, end=cue.end - offset, lines=cue.lines))
    return shifted


def clip_video(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
) -> None:
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def derive_output_paths(video_path: str, index: int, subtitle_ext: str) -> Tuple[str, str]:
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out = os.path.join(base_dir, f"{base_name}_{index}.mp4")
    subtitle_out = os.path.join(base_dir, f"{base_name}_{index}.{subtitle_ext}")
    return video_out, subtitle_out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clip a long video into 25-60s segments using subtitle semantics."
    )
    parser.add_argument("video", help="Path to the source video file.")
    parser.add_argument("subtitles", help="Path to the subtitle file (.vtt or .srt).")
    parser.add_argument("--min-seconds", type=float, default=25.0)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="Only print segment plan.")
    args = parser.parse_args()

    cues, subtitle_ext = load_subtitles(args.subtitles)
    segments = segment_cues(cues, args.min_seconds, args.max_seconds)

    if args.dry_run:
        for idx, (start_pos, end_pos) in enumerate(segments, start=1):
            start_time = cues[start_pos].start
            end_time = cues[end_pos - 1].end
            print(f"{idx}: {start_time:.3f}s -> {end_time:.3f}s ({end_time - start_time:.2f}s)")
        return 0

    for idx, (start_pos, end_pos) in enumerate(segments, start=1):
        seg_start = cues[start_pos].start
        seg_end = cues[end_pos - 1].end
        seg_cues = shift_cues(cues[start_pos:end_pos], seg_start)
        video_out, subtitle_out = derive_output_paths(args.video, idx, subtitle_ext)
        clip_video(args.video, seg_start, seg_end, video_out)
        if subtitle_ext == "vtt":
            write_vtt(seg_cues, subtitle_out)
        else:
            write_srt(seg_cues, subtitle_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
