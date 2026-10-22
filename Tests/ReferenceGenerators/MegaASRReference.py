"""Generate small native-inference parity fixtures; performs no model conversion."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_audio.stt.models.mega_asr.router import AudioQualityRouter

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('Tests/media/mega_asr_reference.safetensors'))
args = parser.parse_args()
router = AudioQualityRouter(d_model=16, nhead=4, dim_feedforward=32, num_layers=1,
    frontend_hidden_dim=8, classifier_hidden_dim=8, max_len=64)
values={}
for key, value in tree_flatten(router.parameters()):
    if key.startswith('logmel.'): continue
    phase=sum(key.encode()) % 31 / 10
    tensor=mx.sin(mx.arange(value.size).reshape(value.shape)*0.13+phase)*0.05
    if key.endswith('weight') and ('.norm' in key or '.bn' in key): tensor=tensor+1
    if key.endswith('running_var'): tensor=mx.abs(tensor)+1
    values[key]=tensor
router.load_weights(list(values.items()),strict=False)
raw={}
for key,value in values.items():
    if '.self_attn.' in key and any('.'+p+'.' in key for p in ('q_proj','k_proj','v_proj')): continue
    name=key
    for old,new in [('frontend.conv1.','frontend.conv.0.'),('frontend.bn1.','frontend.conv.1.'),
                    ('frontend.conv2.','frontend.conv.4.'),('frontend.bn2.','frontend.conv.5.'),
                    ('classifier.fc1.','classifier.0.'),('classifier.fc2.','classifier.3.')]: name=name.replace(old,new)
    raw[name]=value
for parameter in ('weight','bias'):
    raw['transformer.layers.0.self_attn.in_proj_'+parameter]=mx.concatenate([
        values['transformer.layers.0.self_attn.'+part+'.'+parameter] for part in ('q_proj','k_proj','v_proj')],axis=0)
audio=mx.sin(mx.arange(3200,dtype=mx.float32)*0.031)*0.2+mx.cos(mx.arange(3200,dtype=mx.float32)*0.071)*0.03
base=(mx.arange(12,dtype=mx.float32).reshape(3,4)/31-0.2).astype(mx.bfloat16)
a=mx.array([[0.2,-0.1,0.3,0.4],[-0.3,0.2,0.1,0.6]],mx.float32)
b=mx.array([[0.1,-0.2],[0.3,0.4],[-0.2,0.1]],mx.float32)
reference={'audio':audio,'mel':router.logmel(audio),'logits':router.logits(audio),
    'lora_base':base,'lora_A':a,'lora_B':b,'lora_adapted':base+(b@a).astype(base.dtype)}
reference.update({'weights.'+key:value for key,value in raw.items()})
mx.eval(reference)
args.output.parent.mkdir(parents=True,exist_ok=True)
mx.save_safetensors(str(args.output),reference)
print(json.dumps({'output':str(args.output),'router_logits':reference['logits'].tolist(),'weights':len(raw)},indent=2))
