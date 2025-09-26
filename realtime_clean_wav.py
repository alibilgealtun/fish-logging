from __future__ import annotations

import argparse
import math
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from noise.noise_controller import NoiseController


def resample_to_16k(pcm16: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return pcm16
    x = pcm16.astype(np.float32) / 32767.0
    g = math.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    y = resample_poly(x, up, down)
    y = np.clip(y, -1.0, 1.0)
    return np.asarray(y * 32767.0, dtype=np.int16)


def clean_wav_like_realtime(input_wav: str = "test.WAV", output_wav: Optional[str] = "test_clean.wav",
                             sample_rate: int = 16000,
                             vad_mode: int = 2,
                             min_speech_s: float = 0.4,
                             max_segment_s: float = 3.0,
                             padding_ms: int = 600,
                             chunk_s: float = 0.5) -> str:
    # Read input
    audio, sr = sf.read(input_wav, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample to 16 kHz to match realtime recognizer
    audio = resample_to_16k(audio, sr, sample_rate)

    # Initialize controller with same parameters as realtime path
    controller = NoiseController(
        sample_rate=sample_rate,
        vad_mode=vad_mode,
        min_speech_s=min_speech_s,
        max_segment_s=max_segment_s,
    )

    gen = controller.collect_segments(padding_ms=padding_ms)

    # Stream audio in CHUNK_S windows like realtime
    chunk = int(sample_rate * chunk_s)
    pos = 0
    while pos < len(audio):
        controller.push_audio(audio[pos:pos + chunk])
        pos += chunk

    # Append 1s of silence to force final VAD cut, then stop
    controller.push_audio(np.zeros(sample_rate, dtype=np.int16))
    controller.stop()

    # Drain segments
    segments: list[np.ndarray] = []
    for seg in gen:
        if seg is not None and seg.size > 0:
            segments.append(seg)

    # Concatenate and write
    cleaned = np.concatenate(segments) if segments else np.array([], dtype=np.int16)
    out_path = output_wav or (input_wav.rsplit(".", 1)[0] + "_clean.wav")
    sf.write(out_path, cleaned, sample_rate, subtype="PCM_16")
    print(f"Wrote {out_path} | segments={len(segments)} | duration={len(cleaned)/sample_rate:.2f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean a WAV using the same realtime NoiseController pipeline.")
    ap.add_argument("wav", nargs="?", default="test.WAV", help="Input WAV (default: test.WAV)")
    ap.add_argument("-o", "--output", default="test_clean.wav", help="Output WAV (default: test_clean.wav)")
    args = ap.parse_args()
    clean_wav_like_realtime(args.wav, args.output)


if __name__ == "__main__":
    main()

