#!/usr/bin/env python3
"""Create the streaming CoreML/ANE encoder HF repo and upload the .mlpackage + model card.
  uvx --from huggingface_hub python hf_upload_stream.py
"""
from huggingface_hub import HfApi

REPO = "beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane-stream"
# Ship the 8-bit palettized model (2x smaller, ~28% faster, transcript-identical) under the
# production package name so the existing `--ane` auto-download keeps working unchanged.
MLPKG = "out/nemotron_35_stream_p8.mlpackage"
PKG_NAME = "nemotron_35_stream_func.mlpackage"

api = HfApi()
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
api.upload_file(path_or_fileobj="hf_model_card_nemotron_stream.md",
                path_in_repo="README.md", repo_id=REPO)
api.upload_folder(folder_path=MLPKG, path_in_repo=PKG_NAME, repo_id=REPO,
                  commit_message="ship 8-bit palettized encoder (2x smaller, ~28% faster, transcript-identical)")
print(f"done: https://huggingface.co/{REPO}")
