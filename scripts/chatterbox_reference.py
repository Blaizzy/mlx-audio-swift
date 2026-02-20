#!/usr/bin/env python3
"""Generate reference audio with Chatterbox Turbo using the Python MLX Audio library.

Uses the standard load_model pattern from the mlx-audio README.
"""

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

def main():
    text = "Testing one, two, three."
    output_path = "/Users/eric/Development/personal/mlx-audio-swift/chatterbox_turbo_reference.wav"

    print("Loading Chatterbox Turbo model...")
    model = load_model("mlx-community/chatterbox-turbo-fp16")
    print("Model loaded!")

    print(f'Generating audio for: "{text}"')

    audio_data = None
    for result in model.generate(
        text=text,
        temperature=0.8,
        top_p=0.95,
        top_k=1000,
        repetition_penalty=1.2,
        max_tokens=800,
    ):
        print(f"  Audio duration: {result.audio_duration}")
        print(f"  Real-time factor: {result.real_time_factor}x")
        print(f"  Peak memory: {result.peak_memory_usage:.2f} GB")
        audio_data = result.audio

    if audio_data is not None:
        audio_np = np.array(audio_data).flatten()
        print(f"Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")
        print(f"Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")

        sf.write(output_path, audio_np, 24000)
        print(f"\nSaved reference audio to: {output_path}")
    else:
        print("ERROR: No audio generated!")


if __name__ == "__main__":
    main()
