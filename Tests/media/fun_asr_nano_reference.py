"""Regenerate the tiny numerical fixture with the sibling Python mlx-audio. No downloads."""
import json
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_audio.stt.models.fun_asr_nano.config import FunASRNanoConfig
from mlx_audio.stt.models.fun_asr_nano.fun_asr_nano import FunASRNano
from mlx_audio.stt.models.fun_asr_nano.audio import prepare_audio, compute_fbank
config={
 'input_size':56,'frontend_conf':{'n_mels':8},
 'audio_encoder_conf':{'output_size':8,'attention_heads':2,'linear_units':16,'num_blocks':2,'tp_blocks':1,'kernel_size':3},
 'audio_adaptor_conf':{'downsample_rate':2,'ffn_dim':16,'llm_dim':8,'encoder_dim':8,'n_layer':1,'attention_heads':2},
 'text_config':{'hidden_size':8,'intermediate_size':16,'num_hidden_layers':1,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':4,'vocab_size':32,'tie_word_embeddings':True},
}
model=FunASRNano(FunASRNanoConfig.from_dict(config))
weights=[]
for key,value in tree_flatten(model.parameters()):
 phase=(sum(key.encode())%31)/10
 a=mx.sin(mx.arange(value.size,dtype=mx.float32)*0.13+phase).reshape(value.shape)*0.05
 if key.endswith('weight') and ('norm' in key): a=a+1
 weights.append((key,a))
model.load_weights(weights,strict=True)
t=mx.arange(1600,dtype=mx.float32)
audio=0.12*mx.sin(t*0.13)+0.04*mx.cos(t*0.037)
features,lengths,fake=prepare_audio(audio,model.config.frontend_conf)
encoded,_=model.audio_encoder(features,lengths)
adapted,_=model.audio_adaptor(encoded,lengths)
ids=mx.array([[1,4,6]],mx.int32)
logits=model(ids)
fixture={'config':config,'audio':np.array(audio).tolist(),'fake_token_length':fake,'weights':{k:list(v.shape) for k,v in weights}}
for key,value in [('fbank',compute_fbank(audio,model.config.frontend_conf)),('features',features),('encoded',encoded),('adapted',adapted),('logits',logits)]:
 fixture[key]={'shape':list(value.shape),'values':np.array(value).reshape(-1).tolist()}
destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_suffix('.json')
with destination.open('w') as f: json.dump(fixture,f)
print('Fixture generated',[(k,fixture[k]['shape']) for k in ('fbank','features','encoded','adapted','logits')])
