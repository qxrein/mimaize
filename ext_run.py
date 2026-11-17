#!/usr/bin/env python3
"""ABC-Q extensions: ImageNet-pretrained CNNs and LLM weight-only quantisation (PyTorch).

    python ext_run.py --task imagenet --model mobilenetv2 --budgets 0.25 --seeds 0 1 2 3 4
    python ext_run.py --task imagenet --model resnet50    --budgets 0.25 --seeds 0 1 2 3 4
    python ext_run.py --task llm      --model qwen0.5b    --budgets 0.75 --seeds 0 1 2 3 4 --evals 800

ImageNet: ImageNetV2 (matched-frequency, 10k images) split into 512 calibration / 9,488 test images;
weights per-channel symmetric + layer inputs min-max at the layer's bit-width, bits {2,4,8}; cost =
MAC-weighted BOPs relative to W8A8 (same as the CIFAR study).
LLM: Qwen2.5-0.5B, weight-only group-wise (128) RTN quantisation of all 168 linear layers in the
transformer blocks, bits {2,3,4}; cost = bits per parameter relative to 4 bits (budget 0.75 = 3.0
bits/param); score = -NLL on 16 x 512 WikiText-2 *train* tokens; test = perplexity on WikiText-2 test.
The search algorithms are imported unchanged from the CIFAR study (``v2_search``).
"""

from __future__ import annotations

import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["imagenet", "llm"])
ap.add_argument("--quant", default="rtn", choices=["rtn", "gptq"],
                help="LLM layer quantizer used inside the search: round-to-nearest or GPTQ (cached per layer/bit)")
ap.add_argument("--model", required=True)
ap.add_argument("--budgets", type=float, nargs="+", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--methods", nargs="+", default=["random", "ga", "cmaes", "abc_v1", "abc_v2",
                                                  "probe_mckp", "fisher_mckp"])
ap.add_argument("--evals", type=int, default=300)
ap.add_argument("--calib", type=int, default=512)
ap.add_argument("--llm_calib_seqs", type=int, default=8)
ap.add_argument("--llm_test_blocks", type=int, default=128)
args = ap.parse_args()

import numpy as np
import torch
import torch.nn as nn

import v2_search as S
from ext_engine import TorchQuantEvaluator, group_asymmetric, per_channel_symmetric

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ext_data")
OUT = os.path.join(ROOT, "results_ext", args.task + ("_gptq" if args.quant == "gptq" else ""), args.model)
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
torch.manual_seed(0)

HF_IMAGENET = {"mobilenetv2": "google/mobilenet_v2_1.0_224", "resnet18": "microsoft/resnet-18",
               "resnet50": "microsoft/resnet-50"}
HF_LLM = {"opt125m": "facebook/opt-125m", "opt350m": "facebook/opt-350m",
          "qwen0.5b": "Qwen/Qwen2.5-0.5B", "tinyllama1.1b": "TinyLlama/TinyLlama_v1.1",
          "llama3.2-1b": "meta-llama/Llama-3.2-1B"}


# =============================================================================== ImageNet
def load_imagenetv2() -> tuple:
    xnpy, ynpy = os.path.join(DATA, "imagenetv2_224_x.npy"), os.path.join(DATA, "imagenetv2_224_y.npy")
    if os.path.exists(xnpy):  # memory-mapped: pages are read from disk on demand instead of living in RAM/swap
        return np.load(xnpy, mmap_mode="r"), np.load(ynpy)
    cache = os.path.join(DATA, "imagenetv2_224.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        np.save(xnpy, z["x"])
        np.save(ynpy, z["y"])
        del z
        return np.load(xnpy, mmap_mode="r"), np.load(ynpy)
    from PIL import Image
    root = os.path.join(DATA, "imagenetv2-matched-frequency-format-val")
    xs, ys = [], []
    for c in range(1000):
        d = os.path.join(root, str(c))
        for fn in sorted(os.listdir(d)):
            im = Image.open(os.path.join(d, fn)).convert("RGB")
            w, h = im.size
            s = 256 / min(w, h)
            im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
            w, h = im.size
            l, t = (w - 224) // 2, (h - 224) // 2
            xs.append(np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.uint8))
            ys.append(c)
    x, y = np.stack(xs), np.array(ys, dtype=np.int64)
    np.savez(cache, x=x, y=y)
    return x, y


def setup_imagenet():
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    name = HF_IMAGENET[args.model]
    proc = AutoImageProcessor.from_pretrained(name)
    mean = torch.tensor(getattr(proc, "image_mean", [0.485, 0.456, 0.406]), device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(getattr(proc, "image_std", [0.229, 0.224, 0.225]), device=DEV).view(1, 3, 1, 1)
    model = AutoModelForImageClassification.from_pretrained(name).to(DEV).eval()
    offset = 1 if model.config.num_labels == 1001 else 0  # TF-slim MobileNets have a background class
    x, y = load_imagenetv2()
    perm = np.random.RandomState(0).permutation(len(x))
    ci, ti = perm[: args.calib], perm[args.calib:]

    dt = {"v": torch.float32}

    def to_t(idx):
        t = torch.from_numpy(np.ascontiguousarray(x[idx])).to(DEV).permute(0, 3, 1, 2).float().div(255.0).sub(mean).div(std)
        return t.to(dt["v"]).contiguous()  # permute() leaves channels-last strides: ~3x slower on MPS

    xc, yc = to_t(ci), torch.from_numpy(y[ci] + offset).to(DEV)

    def logits(xb):
        return model(pixel_values=xb).logits

    @torch.no_grad()
    def calib_score():
        correct = 0
        for s in range(0, len(xc), 128):
            correct += int((logits(xc[s:s + 128]).argmax(1) == yc[s:s + 128]).sum())
        return correct / len(xc)

    @torch.no_grad()
    def test_score():
        correct = 0
        for s in range(0, len(ti), 256):
            idx = ti[s:s + 256]
            correct += int((logits(to_t(idx)).argmax(1).cpu().numpy() == (y[idx] + offset)).sum())
        return correct / len(ti)

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    # MACs per layer (per image) via a hook pass
    macs = {}
    hs = []
    for n, m in layers:
        def h(mod, inp, out, _n=n):
            if isinstance(mod, nn.Conv2d):
                k = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
                macs[_n] = float(out[0].numel() * k)
            else:
                macs[_n] = float(mod.in_features * mod.out_features)
        hs.append(m.register_forward_hook(h))
    with torch.no_grad():
        logits(xc[:1])
    for hh in hs:
        hh.remove()
    mac = np.array([macs[n] for n, _ in layers])
    bits = [2, 4, 8]
    # weights mixed in {2,4,8}, activations fixed at 8 bits (W{2,4,8}A8): BOPs = b_w * 8 * MACs vs W8A8
    cost = np.stack([b * 8 * mac for b in bits], 1) / (64.0 * mac.sum())

    def grad_stats():
        lossf = nn.CrossEntropyLoss()
        gp = np.zeros(len(layers))
        fi = np.zeros(len(layers))
        nb = 0
        for s in range(0, len(xc), 64):
            model.zero_grad()
            loss = lossf(logits(xc[s:s + 64]), yc[s:s + 64])
            loss.backward()
            for i, (_, m) in enumerate(layers):
                g = m.weight.grad
                gp[i] += float(g.abs().mean())
                fi[i] += float((g.float() ** 2).sum())
            nb += 1
        model.zero_grad()
        return gp / gp.sum(), fi / nb

    gp, fi = grad_stats()
    # forward passes in fp16 from here on (2x faster on Apple GPUs); statistics above were fp32
    model.half()
    dt["v"] = torch.float16
    xc = xc.half().contiguous()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    def act_calib():
        with torch.no_grad():
            for s in range(0, len(xc), 128):
                logits(xc[s:s + 128])

    ev = TorchQuantEvaluator(model, layers, bits, per_channel_symmetric, calib_score, act_quant=True,
                             act_calib_fn=act_calib, act_bits=8)
    return ev, bits, cost, gp, fi, test_score, "top-1 accuracy"


# =============================================================================== LLM
def build_gptq_cache(model, layers, bits, tr, nsamples=128, seqlen=512):
    """GPTQ-quantize every (layer, bit-width) once, from Hessians of the *float* model's layer inputs
    (layers are quantized independently so any allocation can be assembled from the cache), and store
    them on disk. Returns qweight(i, b) that loads a cached layer onto the device."""
    from ext_gptq import gptq_layer
    cdir = os.path.join(DATA, "gptq_cache", args.model)
    os.makedirs(cdir, exist_ok=True)
    fn = lambda i, b: os.path.join(cdir, f"{i}_{b}.pt")
    if not all(os.path.exists(fn(i, b)) for i in range(len(layers)) for b in bits):
        rng = np.random.RandomState(1)
        starts = rng.randint(0, len(tr) - seqlen - 1, size=nsamples)
        calib = torch.stack([tr[s:s + seqlen] for s in starts])
        linalg = torch.device("cuda") if DEV.type == "cuda" else torch.device("cpu")
        inner = model.model
        blocks = inner.decoder.layers if hasattr(inner, "decoder") else inner.layers
        blk_of = []
        for i, (n, _) in enumerate(layers):
            blk_of.append(int(n.split(".")[1]))

        class _Stop(Exception):
            pass
        for b_i, blk in enumerate(blocks):
            idx = [i for i in range(len(layers)) if blk_of[i] == b_i]
            if all(os.path.exists(fn(i, b)) for i in idx for b in bits):
                continue
            H = {i: torch.zeros(layers[i][1].in_features, layers[i][1].in_features, device=DEV) for i in idx}
            cnt = [0]
            hs = []
            for i in idx:
                def hk(mod, inp, _i=i):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    H[_i] += x.t() @ x
                hs.append(layers[i][1].register_forward_pre_hook(hk))

            def stop(mod, inp, out):
                raise _Stop
            hstop = blk.register_forward_hook(stop)
            with torch.no_grad():
                for s in range(0, nsamples, 4):
                    x = calib[s:s + 4].to(DEV)
                    try:
                        model(input_ids=x)
                    except _Stop:
                        pass
                    cnt[0] += x.numel()
            hstop.remove()
            for h in hs:
                h.remove()
            for i in idx:
                Hi = (2.0 / cnt[0]) * H[i].to(linalg)
                W = layers[i][1].weight.data.to(linalg)
                for b in bits:
                    q = gptq_layer(W, Hi, b, 128, 0.01).to(torch.float16).cpu()
                    torch.save(q, fn(i, b))
            del H
            if DEV.type == "mps":
                torch.mps.empty_cache()
            print(f"[gptq-cache] block {b_i + 1}/{len(blocks)}", flush=True)
    dt = layers[0][1].weight.dtype

    def qweight(i, b):
        return torch.load(fn(i, int(b)), map_location="cpu", mmap=True).to(DEV, dtype=dt)
    return qweight


def setup_llm():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = HF_LLM[args.model]
    seq = 512
    tok_cache = os.path.join(DATA, f"wikitext2_{args.model}.npz")
    if os.path.exists(tok_cache):
        z = np.load(tok_cache)
        tr, te = torch.from_numpy(z["tr"]), torch.from_numpy(z["te"])
    else:
        tok = AutoTokenizer.from_pretrained(name)
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        tr = tok("\n\n".join(ds["train"]["text"]), return_tensors="pt").input_ids[0]
        te = tok("\n\n".join(ds["test"]["text"]), return_tensors="pt").input_ids[0]
        np.savez(tok_cache, tr=tr.numpy(), te=te.numpy())
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
    rng = np.random.RandomState(0)
    starts = rng.randint(0, len(tr) - seq - 1, size=args.llm_calib_seqs)
    calib = torch.stack([tr[s:s + seq] for s in starts]).to(DEV)
    nt = min(len(te) // seq, args.llm_test_blocks)  # first 128 x 512 = 65,536 test tokens
    test = te[: nt * seq].reshape(nt, seq).to(DEV)

    bs_eval = 2 if model.config.vocab_size > 60000 else 8  # large-vocab logits (Qwen/Llama) are GBs in fp32

    def nll(batch, bs=None):
        bs = bs or bs_eval
        tot, cnt = 0.0, 0
        for s in range(0, len(batch), bs):
            ids = batch[s:s + bs]
            logits = model(input_ids=ids).logits[:, :-1]
            tgt = ids[:, 1:]
            for k in range(logits.shape[0]):  # loss in fp32, one sequence at a time
                tot += float(nn.functional.cross_entropy(logits[k].float(), tgt[k], reduction="sum"))
            cnt += tgt.numel()
            del logits
        return tot / cnt

    # decoder blocks: model.model.layers (Llama/Qwen) or model.model.decoder.layers (OPT)
    inner = model.model
    blocks = inner.decoder.layers if hasattr(inner, "decoder") else inner.layers
    layers = [(f"layers.{n}", m) for n, m in blocks.named_modules() if isinstance(m, nn.Linear)]
    # gradient prior / empirical-Fisher trace (fp16 forward/backward with loss scaling)
    gp = np.zeros(len(layers))
    fi = np.zeros(len(layers))
    scale = 1024.0
    gstep = 1 if model.config.vocab_size > 60000 else 2
    for p_ in model.parameters():  # gradients only for the quantised linear weights (saves GBs on 1B models)
        p_.requires_grad_(False)
    for _, m in layers:
        m.weight.requires_grad_(True)
    for s in range(0, len(calib), gstep):
        model.zero_grad(set_to_none=True)
        ids = calib[s:s + gstep]
        logits = model(input_ids=ids).logits[:, :-1].float()
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        (loss * scale).backward()
        for i, (_, m) in enumerate(layers):
            g = m.weight.grad.float() / scale
            gp[i] += float(g.abs().mean())
            fi[i] += float((g ** 2).sum())
    model.zero_grad(set_to_none=True)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    gp, fi = gp / gp.sum(), fi / (len(calib) // gstep)

    @torch.no_grad()
    def calib_score():
        return -nll(calib)

    @torch.no_grad()
    def test_score():
        return float(np.exp(nll(test)))  # perplexity

    bits = [2, 3, 4]
    npar = np.array([m.weight.numel() for _, m in layers], dtype=np.float64)
    cost = np.stack([b * npar for b in bits], 1) / (4.0 * npar.sum())
    if args.quant == "gptq":
        qfn = build_gptq_cache(model, layers, bits, tr)
    else:
        qfn = None
    ev = TorchQuantEvaluator(model, layers, bits, group_asymmetric, calib_score, act_quant=False, cache=False)
    if qfn is not None:
        ev.qweight = qfn  # GPTQ-quantized layers loaded from the on-disk cache
    return ev, bits, cost, gp, fi, test_score, "WikiText-2 perplexity"


# =============================================================================== driver
def main():
    t0 = time.time()
    ev, bits, cost, gp, fi, test_score, test_name = setup_imagenet() if args.task == "imagenet" else setup_llm()
    S.BITS = np.array(bits, dtype=np.int64)
    S.BIT_IDX = {b: j for j, b in enumerate(bits)}
    top = bits[-1]
    L = ev.L
    stats_p = os.path.join(OUT, "model_stats.json")
    if os.path.exists(stats_p):
        st = json.load(open(stats_p))
    else:
        st = {"task": args.task, "model": args.model, "L": L, "layer_names": ev.names, "bits": bits,
              "n_params": ev.n_params.tolist(), "cost": cost.tolist(), "test_metric": test_name,
              "float_calib": ev.score(None), "float_test": test_score()}
        for b in bits:
            c = np.full(L, b)
            st[f"uniform{b}"] = {"calib": ev.score(c), "test": test_score(),
                                 "cost": float(cost[np.arange(L), bits.index(b)].sum())}
        t1 = time.time()
        s0, delta = ev.probe(top)
        st.update({"score_top": s0, "delta": delta.tolist(), "probe_wall_s": time.time() - t1,
                   "grad_prior": gp.tolist(), "fisher_trace": fi.tolist(), "weight_qerr": ev.weight_qerr().tolist()})
        json.dump(st, open(stats_p, "w"), indent=1)
    print(f"[stats] L={L} float calib={st['float_calib']:.4f} test={st['float_test']:.4f} "
          + " ".join(f"u{b}: {st[f'uniform{b}']['calib']:.4f}/{st[f'uniform{b}']['test']:.4f}" for b in bits)
          + f" setup {time.time() - t0:.0f}s", flush=True)
    delta = np.array(st["delta"])
    s_top = float(st["score_top"])
    runs_p = os.path.join(OUT, "runs.jsonl")
    done = set()
    if os.path.exists(runs_p):
        for line in open(runs_p):
            r = json.loads(line)
            done.add((r["method"], round(r["budget"], 4), r["seed"]))
    for B in args.budgets:
        for method in args.methods:
            for seed in ([0] if method.endswith("mckp") else args.seeds):
                if (method, round(B, 4), seed) in done:
                    continue
                rng = np.random.RandomState(1000 * seed + 7)
                np.random.seed(1000 * seed + 7)
                oracle = S.Oracle(ev.score, cost, B, args.evals)
                t1 = time.time()
                if method in ("abc_v2", "probe_mckp"):
                    base = np.full(L, top, dtype=np.int64)
                    items = [(base, s_top)]
                    for i in range(L):
                        for j, b in enumerate(bits[:-1]):
                            c = base.copy()
                            c[i] = b
                            items.append((c, s_top - delta[i, j]))
                    for c, a in items:
                        oracle.cache[c.tobytes()] = a
                        oracle.history.append((c, a))
                        oracle.n += 1
                        oracle.trace.append(oracle.best_acc)
                if method == "random":
                    S.run_random(oracle, L, rng)
                elif method == "ga":
                    S.run_ga(oracle, L, rng)
                elif method == "cmaes":
                    S.run_cmaes(oracle, L, rng, seed)
                elif method == "abc_v1":
                    S.run_abc_v1(oracle, L, rng, np.array(st["grad_prior"]))
                elif method == "abc_v2":
                    S.run_abc_v2(oracle, L, rng, s_top, delta)
                elif method == "probe_mckp":
                    oracle(S.mckp(delta, cost, B))
                elif method == "fisher_mckp":
                    om = np.array(st["fisher_trace"])[:, None] * np.array(st["weight_qerr"])
                    oracle(S.mckp(om - om[:, -1:], cost, B))
                best = oracle.best_cfg
                ev.set_config(best)
                rec = {"method": method, "budget": B, "seed": seed, "evals_used": oracle.n,
                       "wall_s": time.time() - t1, "calib_score": oracle.best_acc, "test": test_score(),
                       "cost": oracle.bops(best), "config": best.tolist(),
                       "trace": [float(oracle.trace[k]) for k in range(0, len(oracle.trace), 5)]}
                with open(runs_p, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[run] B={B} {method:12s} seed={seed} calib={rec['calib_score']:.4f} "
                      f"test={rec['test']:.4f} cost={rec['cost']:.3f} wall={rec['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
