"""Export native Python router/transcript references from already prepared weights."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
from mlx_audio.stt.utils import load_model, load_audio

parser=argparse.ArgumentParser()
parser.add_argument('--model-path',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--audio-directory',type=Path,default=Path(__file__).resolve().parents[1]/'media')
args=parser.parse_args()
model=load_model(args.model_path,strict=True)
for name in ('conversational_a','noisy_audio'):
    audio_path=args.audio_directory/(name+'.wav')
    audio=load_audio(str(audio_path),sr=16000)
    mel=model._router.logmel(audio)
    logits=model._router.logits(audio)
    mx.eval(mel,logits)
    route=model._router.route(audio)
    result=model.generate(audio,language='English',max_tokens=256,verbose=False)
    destination=args.output/('mega-reference-'+name)
    destination.mkdir(parents=True,exist_ok=True)
    mx.save_safetensors(str(destination/'reference.safetensors'),{'audio':audio,'router_mel':mel,'router_logits':logits})
    metadata={'checkpoint':str(args.model_path),'revision':args.model_path.name,'audio':str(audio_path),
        'samples':audio.size,'router_logits':logits.tolist(),'route':route,'text':result.text}
    (destination/'reference.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
