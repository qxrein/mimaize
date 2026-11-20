#!/usr/bin/env python3
"""Download the ImageNet-1k (ILSVRC 2012) validation set from Hugging Face (gated: accept the terms at
https://huggingface.co/datasets/ILSVRC/imagenet-1k first; 14 Parquet shards) and build a 224x224 center-crop uint8 cache.

Output: ext_data/imagenet_val_224_x.npy (50000 x 224 x 224 x 3, memory-mappable) and
        ext_data/imagenet_val_224_y.npy (labels in the standard 0..999 class order).
Preprocessing matches the ImageNetV2 path: resize shorter side to 256 (bilinear), center crop 224.
"""

import io
import os

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ext_data")
XN, YN = os.path.join(DATA, "imagenet_val_224_x.npy"), os.path.join(DATA, "imagenet_val_224_y.npy")


def main():
    if os.path.exists(XN) and os.path.exists(YN):
        print("cache exists:", XN)
        return
    import pyarrow.parquet as pq
    os.makedirs(DATA, exist_ok=True)
    shards = [f"data/validation-{i:05d}-of-00014.parquet" for i in range(14)]
    x = np.lib.format.open_memmap(XN + ".tmp", mode="w+", dtype=np.uint8, shape=(50000, 224, 224, 3))
    ys = []
    k = 0
    for sh in shards:
        p = hf_hub_download("ILSVRC/imagenet-1k", sh, repo_type="dataset", local_dir=os.path.join(DATA, "in1k"))
        tab = pq.read_table(p, columns=["image", "label"]).to_pydict()
        for img, lab in zip(tab["image"], tab["label"]):
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            w, h = im.size
            s = 256 / min(w, h)
            im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
            w, h = im.size
            l, t = (w - 224) // 2, (h - 224) // 2
            x[k] = np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8)
            ys.append(int(lab))
            k += 1
        print(f"  {sh}: {k} images so far", flush=True)
        os.remove(p)  # keep disk use low
    assert k == 50000, k
    x.flush()
    del x
    os.replace(XN + ".tmp", XN)
    np.save(YN, np.array(ys, dtype=np.int64))
    print("done:", XN, "labels range", min(ys), max(ys), flush=True)


if __name__ == "__main__":
    main()
