"""Reference inference only; model conversion and weight downloads are outside this tool."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
from mlx_audio.stt import load
from mlx_audio.stt.models.fun_asr_nano.audio import prepare_audio
from mlx_audio.stt.utils import load_audio
parser = argparse.ArgumentParser(description='Generate opt-in parity tensors with already-downloaded Fun-ASR-Nano weights.')
parser.add_argument('model_path')
parser.add_argument('output_directory')
parser.add_argument('--audio', help='Defaults to the checkpoint example/en.mp3')
args = parser.parse_args()
model_path=args.model_path
p=Path(args.output_directory); p.mkdir(parents=True, exist_ok=True)
model=load(model_path,strict=True)
audio_path=args.audio or str(Path(model_path)/'example/en.mp3')
audio=load_audio(audio_path,sr=16000)
features,lengths,_=prepare_audio(audio,model.config.frontend_conf)
encoded,_=model.audio_encoder(features,lengths)
adapted,_=model.audio_adaptor(encoded,lengths)
ids,embeds=model._build_inputs_embeds(audio,language='en',hotwords=None,itn=True)
logits=model(input_ids=ids,input_embeddings=embeds)[0,-1,:].astype(mx.float32)
refs={'audio':audio,'features':features,'encoder':encoded,'adaptor':adapted,'prompt_ids':ids,'embeddings':embeds,'logits':logits}
mx.eval(refs)
mx.save_safetensors(str(p/'reference.safetensors'),refs)
print({k:(list(v.shape),str(v.dtype)) for k,v in refs.items()},flush=True)
result=model.generate(audio,max_tokens=256,language='en')
metadata={'model_path':model_path,'audio_path':audio_path,'transcript':result.text,'duration':len(audio)/16000,'top_token':int(mx.argmax(logits))}
(p/'reference.json').write_text(json.dumps(metadata,indent=2))
print(metadata)
