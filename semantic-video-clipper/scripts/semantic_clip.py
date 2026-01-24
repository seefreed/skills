#!/usr/bin/env python3
"""Clip a long video into 25-60s segments using subtitle semantics."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Tuple

try:
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def check_ffmpeg() -> None:
    """Check if ffmpeg is available in the system PATH."""
    if shutil.which("ffmpeg") is None:
        print(
            "Error: ffmpeg not found. Please install ffmpeg:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: Download from https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)


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


def _parse_cues_core(lines: List[str], skip_line_num: bool = False) -> List[Cue]:
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
        if skip_line_num and line.isdigit():
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


def parse_vtt(lines: List[str]) -> List[Cue]:
    return _parse_cues_core(lines, skip_line_num=False)


def parse_srt(lines: List[str]) -> List[Cue]:
    return _parse_cues_core(lines, skip_line_num=True)


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


def _cosine_similarity_sparse(vec_a: csr_matrix, vec_b: csr_matrix) -> float:
    dot = vec_a.dot(vec_b.T)[0, 0]
    norm_a = math.sqrt(vec_a.power(2).sum())
    norm_b = math.sqrt(vec_b.power(2).sum())
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _build_tfidf_vectors_fast(texts: List[str]) -> Tuple[csr_matrix, List[str]]:
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer.get_feature_names_out().tolist()


def _average_vectors_fast(tfidf_matrix: csr_matrix, indices: List[int]) -> csr_matrix:
    if not indices:
        return csr_matrix((1, tfidf_matrix.shape[1]))
    selected = tfidf_matrix[indices]
    avg = selected.mean(axis=0)
    return csr_matrix(avg)


def _cosine_similarity_dict(vec_a: dict, vec_b: dict) -> float:
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


def build_tfidf_vectors(texts: List[str]):
    if HAS_SKLEARN:
        return _build_tfidf_vectors_fast(texts)
    token_lists = [tokenize(text) for text in texts]
    doc_count = len(token_lists)
    df = {}
    for tokens in token_lists:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    idf = {
        token: math.log((doc_count + 1) / (freq + 1)) + 1.0
        for token, freq in df.items()
    }
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


def average_vectors(vectors, indices: List[int] | None = None):
    if HAS_SKLEARN and isinstance(vectors, csr_matrix):
        if indices is None:
            indices = list(range(vectors.shape[0]))
        return _average_vectors_fast(vectors, indices)
    if not vectors:
        return {}
    sums = {}
    for vec in vectors:
        for token, value in vec.items():
            sums[token] = sums.get(token, 0.0) + value
    count = float(len(vectors))
    return {token: value / count for token, value in sums.items()}


def cosine_similarity(vec_a, vec_b) -> float:
    if HAS_SKLEARN and isinstance(vec_a, csr_matrix):
        return _cosine_similarity_sparse(vec_a, vec_b)
    return _cosine_similarity_dict(vec_a, vec_b)


def compute_boundary_costs(vectors, window: int = 3) -> dict:
    if HAS_SKLEARN and isinstance(vectors, tuple):
        tfidf_matrix, _ = vectors
        n_docs = tfidf_matrix.shape[0]
    else:
        n_docs = len(vectors)
    costs = {0: 0.0, n_docs: 0.0}
    for pos in range(1, n_docs):
        if HAS_SKLEARN and isinstance(vectors, tuple):
            left_indices = list(range(max(0, pos - window), pos))
            right_indices = list(range(pos, min(n_docs, pos + window)))
            left_avg = average_vectors(vectors[0], left_indices)
            right_avg = average_vectors(vectors[0], right_indices)
        else:
            left_vecs = vectors[max(0, pos - window) : pos]
            right_vecs = vectors[pos : min(n_docs, pos + window)]
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
    sentence_end_positions = {
        i + 1 for i, cue in enumerate(cues) if is_sentence_end(cue.text)
    }
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
            boundary_cost = (
                0.0 if end_pos == len(cues) else boundary_costs.get(end_pos, 0.0)
            )
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
        shifted.append(
            Cue(start=cue.start - offset, end=cue.end - offset, lines=cue.lines)
        )
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
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"FFmpeg timed out while clipping {video_path} from {start:.3f}s to {end:.3f}s"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg failed to clip {video_path} from {start:.3f}s to {end:.3f}s:\n"
            f"stderr: {e.stderr}\n"
            f"stdout: {e.stdout}"
        )


def derive_output_paths(
    video_path: str, index: int, subtitle_ext: str
) -> Tuple[str, str]:
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out = os.path.join(base_dir, f"{base_name}_{index}.mp4")
    subtitle_out = os.path.join(base_dir, f"{base_name}_{index}.{subtitle_ext}")
    return video_out, subtitle_out


def _process_segment(args: Tuple[str, List[Cue], int, int, int, str, str]) -> None:
    video_path, cues, start_pos, end_pos, idx, subtitle_ext, base_name = args
    seg_start = cues[start_pos].start
    seg_end = cues[end_pos - 1].end
    seg_cues = shift_cues(cues[start_pos:end_pos], seg_start)
    base_dir = os.path.dirname(video_path)
    video_out = os.path.join(base_dir, f"{base_name}_{idx}.mp4")
    subtitle_out = os.path.join(base_dir, f"{base_name}_{idx}.{subtitle_ext}")
    clip_video(video_path, seg_start, seg_end, video_out)
    if subtitle_ext == "vtt":
        write_vtt(seg_cues, subtitle_out)
    else:
        write_srt(seg_cues, subtitle_out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clip a long video into 25-60s segments using subtitle semantics."
    )
    parser.add_argument("video", help="Path to the source video file.")
    parser.add_argument("subtitles", help="Path to the subtitle file (.vtt or .srt).")
    parser.add_argument("--min-seconds", type=float, default=25.0)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print segment plan."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for video clipping.",
    )
    args = parser.parse_args()

    check_ffmpeg()
    cues, subtitle_ext = load_subtitles(args.subtitles)
    segments = segment_cues(cues, args.min_seconds, args.max_seconds)

    if args.dry_run:
        for idx, (start_pos, end_pos) in enumerate(segments, start=1):
            start_time = cues[start_pos].start
            end_time = cues[end_pos - 1].end
            print(
                f"{idx}: {start_time:.3f}s -> {end_time:.3f}s ({end_time - start_time:.2f}s)"
            )
        return 0

    base_name = os.path.splitext(os.path.basename(args.video))[0]
    segment_args = [
        (args.video, cues, start_pos, end_pos, idx, subtitle_ext, base_name)
        for idx, (start_pos, end_pos) in enumerate(segments, start=1)
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process_segment, arg): arg[3] for arg in segment_args
        }
        if HAS_TQDM:
            with tqdm(total=len(futures), desc="Processing segments") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                        pbar.set_postfix_str(f"segment {idx}")
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing segment {idx}: {e}", file=sys.stderr)
                        raise
        else:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                    print(f"Completed segment {idx}")
                except Exception as e:
                    print(f"Error processing segment {idx}: {e}", file=sys.stderr)
                    raise

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
