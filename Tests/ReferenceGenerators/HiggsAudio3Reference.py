"""Regenerate the tiny numerical fixture with the sibling Python mlx-audio checkout.
Run with that checkout's venv and PYTHONPATH pointing to it; no model conversion.
"""
import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_audio.stt.models.higgs_audio_3.config import ModelConfig
from mlx_audio.stt.models.higgs_audio_3.higgs_audio_3 import Model
from mlx_audio.stt.models.higgs_audio_3.audio import AudioFeatureExtractor
from mlx_audio.lm.models.cache import KVCache

root = Path(__file__).resolve().parents[1] / 'media'
config = {
    'model_type':'higgs_audio_3',
    'audio_encoder_config':{'num_mel_bins':4,'encoder_layers':1,
        'encoder_attention_heads':2,'encoder_ffn_dim':16,'d_model':8,'max_source_positions':32},
    'text_config':{'vocab_size':32,'hidden_size':8,'intermediate_size':16,
        'num_hidden_layers':1,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':4,
        'max_position_embeddings':128,'tie_word_embeddings':False},
    'vad_cut':False,'chunk_size_seconds':0.1,
}
mx.random.seed(73)
model = Model(ModelConfig.from_dict(config))
model.eval()
wave = mx.sin(mx.arange(1600, dtype=mx.float32) * 0.013) * 0.4 + mx.cos(mx.arange(1600, dtype=mx.float32) * 0.031) * 0.2
mel = mx.cos(mx.arange(48,dtype=mx.float32) * 0.1).reshape(1,4,12)
encoded = model.audio_tower(mel)
projected = model.audio_encoder_proj(encoded)
ids = mx.array([[3,5,8]], dtype=mx.int32)
cache = [KVCache()]
logits = model(ids, cache=cache)
cached = model(mx.array([[9]],dtype=mx.int32), cache=cache)
fixture = {'weights.'+k:v for k,v in tree_flatten(model.parameters())}
fixture.update({'reference.wave':wave,'reference.mel':mel,
    'reference.frontend':AudioFeatureExtractor(num_mel_bins=4)(wave),
    'reference.encoder':encoded,'reference.projector':projected,
    'reference.logits':logits,'reference.cached_logits':cached})
mx.eval(fixture)
root.mkdir(exist_ok=True)
mx.save_safetensors(str(root/'higgs_audio3_reference.safetensors'),fixture)
(root/'higgs_audio3_reference_config.json').write_text(json.dumps(config,indent=2)+'\n')
print('Generated',root/'higgs_audio3_reference.safetensors')
for key,value in fixture.items():
    if key.startswith('reference.'): print(key,value.shape)
