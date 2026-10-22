"""Generate tiny deterministic Phonon-2 parity fixtures; does not convert model weights.

Run with the sibling mlx-audio environment and PYTHONPATH pointing to that repo.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.stt.models.parakeet.parakeet import ParakeetTDT, ParakeetTDTArgs
from mlx_audio.stt.models.phonon2 import Model, ModelConfig
from mlx_audio.stt.models.phonon2.packed import ENCODER_SUFFIXES, PackedFiveValueLinear
from mlx_audio.stt.utils import load_audio
from mlx_audio.utils import from_dict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, help="Optional local real checkpoint; no downloads or conversion")
parser.add_argument("--audio", type=Path, default=Path(__file__).resolve().parents[1] / "Tests/media/conversational_a.wav")
parser.add_argument("--output", type=Path, help="Required with --checkpoint, outside committed fixture directory")
args = parser.parse_args()
if args.checkpoint:
    if not args.output:
        parser.error("--checkpoint requires --output")
    args.output.mkdir(parents=True, exist_ok=True)
    config = ModelConfig.from_dict(json.loads((args.checkpoint / "config.json").read_text()))
    model = Model(config)
    weights = mx.load(str(args.checkpoint / "model.safetensors"))
    weights = {
        k: v.astype(mx.float32) if v.dtype in (mx.float16, mx.bfloat16, mx.float32)
        and not k.endswith((".row_lo", ".row_hi")) else v for k, v in weights.items()
    }
    model.load_weights(list(weights.items()))
    model.eval()
    waveform = load_audio(str(args.audio), sr=16000, dtype=mx.float32)
    mel = model._log_mel_spectrogram(waveform)
    encoded, lengths = model.encoder(mel, mx.array([mel.shape[1]], mx.int32))
    mx.save_safetensors(str(args.output / "reference.safetensors"), dict(waveform=waveform, mel=mel, encoded=encoded, lengths=lengths))
    output = model.generate(waveform, dtype=mx.float32, chunk_duration=30)
    (args.output / "transcript.txt").write_text(output.text)
    print("Python Phonon-2 FP32:", output.text)
    del model, weights
    mx.clear_cache()
    model = Model(config)
    model.load_weights(str(args.checkpoint / "model.safetensors"))
    model.eval()
    output = model.generate(waveform, dtype=mx.bfloat16, chunk_duration=30)
    (args.output / "transcript_bf16.txt").write_text(output.text)
    print("Python Phonon-2 BF16:", output.text)
    raise SystemExit(0)

out = Path(__file__).resolve().parents[1] / "Tests/media/phonon2"
out.mkdir(parents=True, exist_ok=True)
expected = out / "expected"
expected.mkdir(exist_ok=True)
config = {
    "model_type": "phonon2",
    "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
    "model_defaults": {"tdt_durations": [0, 1, 2]},
    "preprocessor": dict(sample_rate=16000, normalize="per_feature", window_size=0.025,
                         window_stride=0.01, window="hann", features=8, n_fft=512, dither=0),
    "encoder": dict(feat_in=8, n_layers=1, d_model=128, n_heads=4, ff_expansion_factor=2,
                    subsampling_factor=2, self_attention_model="rel_pos", subsampling="dw_striding",
                    conv_kernel_size=3, subsampling_conv_channels=4, pos_emb_max_len=128, use_bias=False),
    "decoder": dict(blank_as_pad=True, vocab_size=8, prednet=dict(pred_hidden=16, pred_rnn_layers=1)),
    "joint": dict(num_classes=8, vocabulary=[f"▁word{i}" for i in range(8)], num_extra_outputs=3,
                  jointnet=dict(joint_hidden=16, activation="relu", encoder_hidden=128, pred_hidden=16)),
    "decoding": dict(model_type="tdt", durations=[0, 1, 2], greedy=dict(max_symbols=4)),
}


def pack(codes):
    codes = codes.astype(np.uint32).reshape(codes.shape[0], -1, 16)
    return np.bitwise_or.reduce(codes << (2 * np.arange(16, dtype=np.uint32)), axis=-1)


mx.random.seed(7)
dense = ParakeetTDT(from_dict(ParakeetTDTArgs, config))
weights, modules = {}, []
for name, weight in tree_flatten(dense.parameters()):
    suffix = name.removeprefix("encoder.layers.0.").removesuffix(".weight")
    if name.startswith("encoder.layers.0.") and name.endswith(".weight") and suffix in ENCODER_SUFFIXES:
        rows, cols = weight.shape[0], int(np.prod(weight.shape[1:]))
        sign = np.resize(np.array([-1, 0, 1], np.int8), (rows, cols))
        mask = (np.indices((rows, cols))[1] % 2 == 0) & (sign != 0)
        module = name.removesuffix(".weight")
        modules.append(dict(name=module, shape=[rows, cols]))
        weights[f"{module}.base_q"] = mx.array(pack(sign + 1))
        weights[f"{module}.residual_q"] = mx.array(pack(sign * mask + 1))
        weights[f"{module}.row_lo"] = mx.full((rows,), 0.015625, mx.float16)
        weights[f"{module}.row_hi"] = mx.full((rows,), 0.046875, mx.float16)
    else:
        if name == "joint.joint_net.2.weight":
            weight = mx.zeros_like(weight)
        elif name == "joint.joint_net.2.bias":
            weight = mx.zeros_like(weight)
            weight[8] = weight[10] = 20
        weights[name] = weight.astype(mx.float32)
config["phonon2_modules"] = modules
(out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
mx.save_safetensors(str(out / "model.safetensors"), weights)
model = Model(ModelConfig.from_dict(config))
model.load_weights(list(weights.items()))
model.eval()
waveform = (mx.sin(mx.arange(1600) * 0.031) * 0.3 + mx.cos(mx.arange(1600) * 0.013) * 0.1).astype(mx.float32)
mel = model._log_mel_spectrogram(waveform)
encoded, lengths = model.encoder(mel, mx.array([mel.shape[1]], mx.int32))
mx.save_safetensors(str(expected / "reference.safetensors"), {
    "waveform": waveform, "mel": mel, "encoded": encoded, "lengths": lengths,
    "short_mel": model._log_mel_spectrogram(waveform[:1]),
})

# Independent packed-layer reference uses non-exact fp16 palettes to expose rounding mistakes.
rows, cols = 3, 128
sign = np.resize(np.array([-1, 0, 1], np.int8), (rows, cols))
mask = (np.indices(sign.shape)[1] % 2 == 0) & (sign != 0)
layer = PackedFiveValueLinear(cols, rows, bias=True)
layer.base_q, layer.residual_q = mx.array(pack(sign + 1)), mx.array(pack(sign * mask + 1))
layer.row_lo = mx.array([0.1, 0.2, 0.3], mx.float16)
layer.row_hi = mx.array([0.4, 0.5, 0.6], mx.float16)
layer.bias = mx.array([1, 2, 3], mx.bfloat16)
layer.materialize()
x = mx.array(np.random.default_rng(8).normal(size=(2, 3, cols)).astype(np.float32))
packed = dict(tree_flatten(layer.parameters()))
packed.update(input=x, expected=layer(x), expected_bf16=layer(x.astype(mx.bfloat16)))
mx.save_safetensors(str(expected / "packed.safetensors"), packed)
print(f"Generated Python parity fixtures in {out}")
