"""Download public data/models for the ImageNet and LLM extensions (ImageNetV2, HF models, WikiText-2)."""
import os, tarfile
from huggingface_hub import hf_hub_download, snapshot_download
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext_data")
os.makedirs(D, exist_ok=True)
tgt = os.path.join(D, "imagenetv2-matched-frequency-format-val")
if not os.path.isdir(tgt):
    p = hf_hub_download("vaishaal/ImageNetV2", "imagenetv2-matched-frequency.tar.gz", repo_type="dataset", local_dir=D)
    print("extracting", p, flush=True)
    with tarfile.open(p) as t:
        t.extractall(D)
    os.remove(p)
print("imagenetv2 ok:", len(os.listdir(tgt)), "class dirs", flush=True)
for m in ["google/mobilenet_v2_1.0_224", "microsoft/resnet-50", "Qwen/Qwen2.5-0.5B"]:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.*", "tokenizer*"])
    print("model ok:", m, flush=True)
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
print("wikitext ok:", {k: len(v) for k, v in ds.items()}, flush=True)
