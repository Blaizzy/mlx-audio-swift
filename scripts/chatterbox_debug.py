#!/usr/bin/env python3
"""Debug script: dump intermediate tensor values from Chatterbox Turbo S3Gen pipeline.

Uses the standard generate path to get speech tokens, then traces S3Gen step by step.
Saves intermediate stats + the speech tokens so we can replicate in Swift exactly.
"""

import json
import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


def dump_stats(name: str, arr, results: dict):
    """Dump shape, dtype, min, max, mean, first few values."""
    a = np.array(arr).flatten()
    info = {
        "shape": list(np.array(arr).shape),
        "dtype": str(a.dtype),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "first_10": [float(x) for x in a[:10]],
        "last_5": [float(x) for x in a[-5:]],
    }
    results[name] = info
    print(f"  {name}: shape={info['shape']}, range=[{info['min']:.6f}, {info['max']:.6f}], mean={info['mean']:.6f}")


def main():
    text = "Testing one, two, three."
    results = {}

    print("Loading model...")
    model = load_model("mlx-community/chatterbox-turbo-fp16")
    s3gen = model.s3gen
    conds = model._conds

    # Dump conditioning info
    print("\n=== Conditioning ===")
    dump_stats("conds.gen.prompt_token", conds.gen["prompt_token"], results)
    dump_stats("conds.gen.prompt_token_len", conds.gen["prompt_token_len"], results)
    dump_stats("conds.gen.prompt_feat", conds.gen["prompt_feat"], results)
    dump_stats("conds.gen.embedding", conds.gen["embedding"], results)

    # Generate speech tokens via the full pipeline
    print("\n=== Generating speech tokens via model.generate() ===")
    audio_result = None
    for result in model.generate(
        text=text, temperature=0.8, top_p=0.95,
        top_k=1000, repetition_penalty=1.2, max_tokens=800,
    ):
        audio_result = result

    # Now, we need the actual speech tokens that were generated.
    # The best approach: instrument the S3Gen pipeline by manually running it
    # with the SAME conditioning data but with DETERMINISTIC inputs.
    #
    # Since T3 tokens are random, let's create synthetic speech tokens that
    # we can use identically in Python and Swift.
    # Use first 30 values from prompt_token as "generated speech tokens"
    prompt_token = conds.gen["prompt_token"]
    # Use a fixed small set of tokens for testing
    test_speech_tokens = prompt_token[:, :30]  # First 30 prompt tokens as test
    dump_stats("test_speech_tokens", test_speech_tokens, results)
    # Save the actual int values
    results["test_speech_tokens_values"] = [int(x) for x in np.array(test_speech_tokens).flatten()]

    print("\n=== S3Gen Pipeline (with test tokens) ===")
    B = 1
    prompt_token_len = conds.gen["prompt_token_len"]
    prompt_feat = conds.gen["prompt_feat"]
    embedding = conds.gen["embedding"]

    # 1. Speaker embedding normalization + projection
    embedding_norm = embedding / (mx.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8)
    spk_emb = s3gen.spk_embed_affine_layer(embedding_norm)
    dump_stats("spk_emb_projected", spk_emb, results)

    # 2. Token concatenation
    token_len = mx.array([test_speech_tokens.shape[1]])
    combined_token = mx.concatenate([prompt_token, test_speech_tokens], axis=1)
    combined_token_len = prompt_token_len + token_len
    dump_stats("combined_token", combined_token, results)
    dump_stats("combined_token_len", combined_token_len, results)

    # 3. Mask + Embed
    max_len = combined_token.shape[1]
    mask = mx.arange(max_len)[None, :] < combined_token_len[:, None]
    mask = mask[:, :, None].astype(mx.float32)
    token_emb = s3gen.input_embedding(combined_token.astype(mx.int32)) * mask
    dump_stats("token_emb", token_emb, results)

    # 4. Encode
    h, h_masks = s3gen.encoder(token_emb, combined_token_len)
    dump_stats("encoder_out_h", h, results)

    # 5. Project
    h_lengths = mx.sum(h_masks[:, 0, :].astype(mx.int32), axis=-1)
    mel_len1 = prompt_feat.shape[1]
    mel_len2 = h.shape[1] - mel_len1
    results["mel_len1_prompt"] = int(np.array(mel_len1))
    results["mel_len2_gen"] = int(np.array(mel_len2))
    results["h_lengths"] = int(np.array(h_lengths).flatten()[0])
    print(f"  mel_len1 (prompt): {mel_len1}, mel_len2 (gen): {mel_len2}, h_lengths: {results['h_lengths']}")

    h_proj = s3gen.encoder_proj(h)
    dump_stats("encoder_proj_h", h_proj, results)

    # 6. Conditioning signal
    zeros_padding = mx.zeros((B, mel_len2, 80))
    conds_signal = mx.concatenate([prompt_feat, zeros_padding], axis=1)
    conds_signal_t = conds_signal.transpose(0, 2, 1)  # (B, 80, T)
    dump_stats("conds_signal", conds_signal_t, results)

    # 7. Decoder mask
    dec_mask = mx.arange(h_proj.shape[1])[None, :] < h_lengths[:, None]
    dec_mask = dec_mask[:, None, :].astype(mx.float32)
    dump_stats("dec_mask", dec_mask, results)

    # 8. mu
    mu = h_proj.transpose(0, 2, 1)  # (B, 80, T)
    dump_stats("mu", mu, results)

    # 9. Noised mels (using fixed seed for reproducibility)
    mx.random.seed(123)  # Fixed seed
    noised_mels = mx.random.normal((B, 80, test_speech_tokens.shape[1] * 2))
    dump_stats("noised_mels", noised_mels, results)

    # 10. Flow matching
    mx.random.seed(123)  # Reset seed
    feat, _ = s3gen.decoder(
        mu=mu, mask=dec_mask, n_timesteps=2,
        spks=spk_emb, cond=conds_signal_t,
        noised_mels=noised_mels, meanflow=True,
    )
    dump_stats("flow_matching_output", feat, results)

    # 11. Extract generated portion
    feat_gen = feat[:, :, mel_len1:]
    dump_stats("feat_gen_only", feat_gen, results)

    # 12. Vocoder
    print("\n=== HiFi-GAN Vocoder ===")
    mel_for_vocoder = feat_gen.transpose(0, 2, 1)  # (B, T, 80)
    dump_stats("mel_for_vocoder_BT80", mel_for_vocoder, results)

    mel_cf = feat_gen  # Already (B, 80, T) — keep channel-first for HiFiGAN
    f0 = s3gen.mel2wav.f0_predictor(mel_cf)
    dump_stats("f0_predicted", f0, results)

    f0_up = s3gen.mel2wav._upsample_f0(f0)
    dump_stats("f0_upsampled", f0_up, results)

    source_out, _, _ = s3gen.mel2wav.m_source(f0_up)
    source_t = source_out.transpose(0, 2, 1)  # (B, 1, T_audio)
    dump_stats("source_signal", source_t, results)

    audio = s3gen.mel2wav.decode(mel_cf, source_t)
    dump_stats("vocoder_output", audio, results)

    # Save audio
    audio_np = np.array(audio).flatten()
    sf.write("/Users/eric/Development/personal/mlx-audio-swift/chatterbox_debug_python.wav", audio_np, 24000)
    print(f"\nSaved debug audio (test tokens)")

    # Save results
    output_file = "/Users/eric/Development/personal/mlx-audio-swift/chatterbox_debug_python.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved debug info to: {output_file}")


if __name__ == "__main__":
    main()
