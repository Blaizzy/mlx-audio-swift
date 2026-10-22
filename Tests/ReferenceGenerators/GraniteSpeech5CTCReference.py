"""Generate numerical references from Python mlx-audio; never convert weights."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_audio.stt.models.granite_speech5_ctc.config import EncoderConfig, ModelConfig
from mlx_audio.stt.models.granite_speech5_ctc.granite_speech5 import Model, compute_features, ctc_collapse
from mlx_audio.stt.utils import load_model, load_audio


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    encoder = EncoderConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        num_mel_bins=4, head_dim=8, max_position_embeddings=16,
        context_size=8, conv_kernel_size=3, subsample_layers=[0, 1])
    config = ModelConfig(vocab_size=16, encoder_config=encoder)
    model = Model(config)
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        phase = (sum(key.encode()) % 31) / 10
        weight = mx.sin(mx.arange(value.size, dtype=mx.float32) * 0.13 + phase).reshape(value.shape) * 0.03
        if key.endswith("running_var") or (key.endswith("weight") and "norm" in key):
            weight = weight + 1
        weights[key] = weight
    model.load_weights(list(weights.items()), strict=True)
    waveform = mx.sin(mx.arange(4961, dtype=mx.float32) * 0.02) * 0.15
    features = compute_features(waveform, num_mel_bins=4)
    input_features = mx.sin(mx.arange(2 * 19 * 16, dtype=mx.float32) * 0.017).reshape(2,19,16)
    mask = mx.array([[True] * 19, [True] * 13 + [False] * 6])
    arrays = {"weights." + key: value for key, value in weights.items()}
    arrays.update({"reference.audio": waveform, "reference.features": features,
        "reference.input": input_features, "reference.mask": mask,
        "reference.logits": model(input_features),
        "reference.masked_logits": model(input_features, attention_mask=mask)})
    mx.eval(arrays)
    mx.save_safetensors(str(output / "granite_speech5_ctc_reference.safetensors"), arrays)
    (output / "granite_speech5_ctc_reference_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")


def checkpoint(model_path, audio_path, output):
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(Path(model_path), strict=True)
    audio = load_audio(str(audio_path), sr=16000)
    features = compute_features(audio)[None]
    logits = model(features)
    ids = ctc_collapse(mx.argmax(logits[0], axis=-1), blank_id=model.config.pad_token_id)
    mx.eval(features, logits, ids)
    text = model._tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()
    mx.save_safetensors(str(output / "reference.safetensors"),
        {"audio": audio, "features": features, "logits": logits, "tokens": ids})
    (output / "reference.json").write_text(json.dumps({"checkpoint": str(model_path),
        "audio": str(audio_path), "text": text, "tokens": ids.tolist()}, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "media")
    parser.add_argument("--model-path")
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    if args.model_path:
        if args.audio is None:
            parser.error("--audio is required with --model-path")
        checkpoint(args.model_path, args.audio, args.output)
    else:
        tiny(args.output)
