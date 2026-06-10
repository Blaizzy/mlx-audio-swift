#!/usr/bin/env python3
"""Re-upload the 8-bit palettized OFFLINE Nemotron 3.5 encoder (2x smaller, transcript-identical)
under the existing production package name so `--ane` keeps working unchanged.
  uvx --from huggingface_hub python hf_upload_offline.py
"""
from huggingface_hub import HfApi

REPO = "beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane"
MLPKG = "out/nemotron_35_enc_p8.mlpackage"
PKG_NAME = "nemotron_enc_3.5_0.6b.mlpackage"  # existing name in the repo

api = HfApi()
api.upload_file(path_or_fileobj="hf_model_card_nemotron.md", path_in_repo="README.md", repo_id=REPO)
api.upload_folder(folder_path=MLPKG, path_in_repo=PKG_NAME, repo_id=REPO,
                  commit_message="ship 8-bit palettized offline encoder (2x smaller, transcript-identical)")
print(f"done: https://huggingface.co/{REPO}")
