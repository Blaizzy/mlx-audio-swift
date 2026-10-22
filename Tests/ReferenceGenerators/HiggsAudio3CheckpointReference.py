"""Export runtime references and transcripts from an existing checkpoint.
This is a parity helper, not a model converter. Requires sibling mlx-audio.
"""
import argparse
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_audio.stt import load
from mlx_audio.stt.utils import load_audio
from mlx_audio.lm.models.cache import KVCache

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--output', required=True, help='Output .safetensors reference path')
args = parser.parse_args()
media = Path(__file__).resolve().parents[1] / 'media'
model = load(args.model_path, strict=True)
# Compare inference independently of VAD backend boundary differences.
model.config.vad_cut = False
wave = load_audio(str(media / 'intention.wav'), sr=16000)
chunks = model._chunk_waveform(np.array(wave))
longest = max(map(len, chunks))
first = mx.array(np.pad(chunks[0], (0, longest - len(chunks[0]))))
mel = model._extract_features(first)
encoded = model.audio_tower(mel)
projected = model.get_audio_features(mel)
ids, embeds, _ = model.get_input_embeddings(wave)
cache = [KVCache() for _ in model.layers]
hidden = model.model(None, cache=cache, input_embeddings=embeds)
logits = model.lm_head(hidden[:, -1:, :])
mx.eval(mel, encoded, projected, embeds, logits)
mx.save_safetensors(args.output, {'audio': wave, 'mel': mel, 'encoded': encoded,
    'projected': projected, 'embeddings': embeds, 'logits': logits, 'ids': mx.array(ids)})
results = {}
for filename in ('intention.wav', 'conversational_a.wav'):
    audio = load_audio(str(media / filename), sr=16000)
    result = model.generate(audio, max_tokens=256)
    results[filename] = {'text': result.text, 'prompt_tokens': result.prompt_tokens,
        'generation_tokens': result.generation_tokens, 'total_time': result.total_time}
    print(filename, results[filename], flush=True)
Path(args.output).with_suffix('.json').write_text(json.dumps(results, indent=2) + '\n')
