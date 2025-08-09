from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from jiwer import wer

SAMPLES_DIR = Path("samples")
REFS_FILE = SAMPLES_DIR / "refs.txt"

MODEL_NAME = "base.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
FISH_PROMPT = (
    "This is a continuous conversation about fish species and their lengths, given in centimeters. "
    "Always transcribe fish names accurately, even if they are pronounced incorrectly or partially. "
    "Correct misheard words to the nearest valid fish name or number."
    "Example fish names: salmon, trout, tuna, sardine, mackerel, anchovy, bass, snapper, cod, haddock, flounder, sea bass. "
    "Lengths will usually be given in centimeters (e.g., 10cm, 20cm... 100cm). "
    "If 'cm' or 'centimeters' is pronounced incorrectly, still write it as 'cm'."
)


def load_references() -> List[Tuple[Path, str]]:
    pairs: List[Tuple[Path, str]] = []
    if not REFS_FILE.exists():
        print(f"Reference file not found: {REFS_FILE}. Create lines as 'filename.wav|reference text'")
        return pairs
    for line in REFS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            continue
        fname, ref = line.split("|", 1)
        wav_path = SAMPLES_DIR / fname.strip()
        if wav_path.exists():
            pairs.append((wav_path, ref.strip()))
        else:
            print(f"Missing wav file: {wav_path}")
    return pairs


def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise to achieve target SNR in dB."""
    if audio.size == 0:
        return audio
    sig_power = np.mean(audio**2)
    if sig_power <= 1e-12:
        return audio
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(scale=np.sqrt(noise_power), size=audio.shape).astype(audio.dtype)
    return audio + noise


def transcribe_file(model: WhisperModel, audio_path: Path) -> str:
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="en",
        condition_on_previous_text=True,
        initial_prompt=FISH_PROMPT,
    )
    text_parts: List[str] = []
    for s in segments:
        text_parts.append(s.text)
    return " ".join(text_parts).strip()


def main() -> None:
    refs = load_references()
    if not refs:
        return
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    snr_levels = [40.0, 30.0, 20.0, 10.0, 5.0, 0.0]
    print(f"Evaluating SNR levels: {snr_levels} dB")

    for snr in snr_levels:
        hyps: List[str] = []
        gts: List[str] = []
        print(f"\n--- SNR = {snr:.1f} dB ---")
        for path, ref in refs:
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            noisy = add_white_noise(audio, snr)
            tmp_path = path.with_suffix("")
            tmp_noisy = tmp_path.with_name(tmp_path.name + f"_snr{int(snr)}.wav")
            sf.write(str(tmp_noisy), noisy, sr)
            hyp = transcribe_file(model, tmp_noisy)
            hyps.append(hyp)
            gts.append(ref)
            print(f"- {path.name}: REF='{ref}' | HYP='{hyp}'")
            try:
                tmp_noisy.unlink()
            except Exception:
                pass

        score = wer(gts, hyps)
        print(f"WER @ {snr:.1f} dB: {score:.3f}")


if __name__ == "__main__":
    main()
