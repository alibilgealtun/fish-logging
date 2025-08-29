# eval_noise.py
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
from speech import SpeechRecognizer
from faster_whisper import WhisperModel

# SNR levels to test
snrs = [40, 30, 20, 10, 5, 0]
samples_dir = Path("samples")

# Init recognizer + model
recognizer = SpeechRecognizer()
recognizer._model = WhisperModel(
    recognizer.MODEL_NAME,
    device=recognizer.DEVICE,
    compute_type=recognizer.COMPUTE_TYPE
)

results = []

# Loop over wav files in samples/
for wav_path in samples_dir.glob("*.wav"):
    print(f"Processing {wav_path.name} ...")

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    for snr in snrs:
        # Add noise
        sig_power = np.mean(audio**2)
        noise_power = sig_power / (10 ** (snr / 10))
        noise = np.random.normal(scale=np.sqrt(noise_power), size=audio.shape).astype(audio.dtype)
        noisy = audio + noise

        # Save temp noisy wav
        tmp_path = wav_path.with_name(f"{wav_path.stem}_snr{snr}.wav")
        sf.write(tmp_path, noisy, sr)

        # Transcribe
        segments, _ = recognizer._model.transcribe(
            str(tmp_path),
            beam_size=5,
            language="en",
            condition_on_previous_text=True,
            initial_prompt=recognizer.FISH_PROMPT
        )
        raw_segments = [seg.text for seg in segments]
        hyp = " ".join(raw_segments).strip()

        results.append({
            "file": wav_path.name,
            "snr": snr,
            "hyp": hyp,
            "segments": " | ".join(raw_segments)
        })

        # Clean up noisy file
        tmp_path.unlink()

# Save all results
df = pd.DataFrame(results)
df.to_excel("noise_eval.xlsx", index=False)
print("✅ Transcriptions written to noise_eval.xlsx")
