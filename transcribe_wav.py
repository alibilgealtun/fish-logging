from __future__ import annotations

import argparse

from noise_removing import process_wav


def transcribe_with_whisper(clean_wav: str, model_name: str = "base.en", device: str = "cpu", compute_type: str = "int8") -> str:
    from faster_whisper import WhisperModel  # lazy import

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        clean_wav,
        beam_size=3,
        best_of=5,
        temperature=0.0,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=1.0,
        language="en",
        condition_on_previous_text=True,
        initial_prompt=(
            "This is a continuous conversation about fish species and numbers (their measurements). "
            "The user typically speaks fish specie or number. "
            "Always prioritize fish species vocabulary over similar-sounding common words. "
            "If a word sounds like a fish name, bias towards the fish name. "
            "Units are typically centimeters (cm) or millimeters (mm), 'cm' is preferred in the transcript. "
            "You might also hear 'cancel', 'wait' and 'start'."
        ),
        vad_filter=False,
        vad_parameters=None,
        without_timestamps=True,
        word_timestamps=False,
    )
    return " ".join(seg.text for seg in segments).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a WAV with adaptive denoising (same algo as realtime)")
    ap.add_argument("wav", nargs="?", default="test.WAV", help="Input WAV file path (default: test.WAV)")
    ap.add_argument("-o", "--output", help="Optional: write cleaned wav here (default: <input>_clean.wav)")
    ap.add_argument("-r", "--rate", type=int, default=16000, help="Target sample rate (default: 16000)")
    ap.add_argument("--model", default="base.en", help="Whisper model name (default: base.en)")
    args = ap.parse_args()

    # 1) Denoise using the same algorithm
    clean_path = process_wav(args.wav, args.output, sample_rate=args.rate)

    # 2) Transcribe cleaned file
    text = transcribe_with_whisper(clean_path, model_name=args.model)
    print(text)


if __name__ == "__main__":
    main()
