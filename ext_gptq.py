#!/usr/bin/env python3
"""Mixed-precision GPTQ for the bit allocations found by ABC-Q and the baselines.

GPTQ (Frantar et al., 2023) quantizes the linear layers of each transformer block one column at a time,
compensating the rounding error of every column with the inverse Hessian H = 2 X^T X of the layer
inputs. Here every layer is quantized at *its own* bit-width taken from an allocation (uniform 3/4-bit,
Probe-MCKP, ABC-Q, ...), with group-wise (128) asymmetric min-max scales -- the same quantization grid
as the round-to-nearest (RTN) study, so RTN and GPTQ results are directly comparable.

Blocks are processed sequentially: when block k is quantized, its Hessians are collected from a forward
pass in which blocks 0..k-1 are already quantized (standard GPTQ).

Evaluation: WikiText-2 test perplexity with the standard protocol of the GPTQ paper (whole test set,
non-overlapping windows of 2048 tokens), plus the 65,536-token / 512-context protocol used during search.

    python ext_gptq.py --model opt125m
    python ext_gptq.py --model llama3.2-1b --allocs rtn3 rtn4 gptq3 gptq4 probe_mckp abc_v2

Runs on CUDA (e.g. RTX 3050 6 GB), Apple MPS, or CPU. Results: results_ext/gptq/<model>.json (resumable).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--allocs", nargs="+",
                default=["fp16", "rtn3", "rtn4", "gptq3", "gptq4", "probe_mckp", "abc_v2"])
ap.add_argument("--nsamples", type=int, default=128, help="GPTQ calibration sequences")
ap.add_argument("--seqlen", type=int, default=512, help="GPTQ calibration sequence length")
ap.add_argument("--eval_ctx", type=int, default=2048)
ap.add_argument("--group", type=int, default=128)
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--alloc_src", default="llm_gptq", choices=["llm", "llm_gptq"],
                help="where searched allocations come from: RTN search (llm) or GPTQ-aware search (llm_gptq)")
ap.add_argument("--tag", default="")
ap.add_argument("--extra", action="store_true", help="also C4 perplexity + 6 zero-shot tasks (-> <model>_extra.json)")
args, _unknown = ap.parse_known_args()  # tolerate foreign args when imported by ext_run.py

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "ext_data")
OUT = os.path.join(ROOT, "results_ext", "gptq")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
# Cholesky / triangular solves are not implemented on MPS: do the GPTQ math on CPU there.
LINALG_DEV = DEV if DEV.type == "cuda" else torch.device("cpu")
HF = {"opt125m": "facebook/opt-125m", "opt350m": "facebook/opt-350m", "qwen0.5b": "Qwen/Qwen2.5-0.5B",
      "llama3.2-1b": "meta-llama/Llama-3.2-1B"}


# ------------------------------------------------------------------------------------------ data
def tokens(tok):
    p = os.path.join(DATA, f"wikitext2_{args.model}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return torch.from_numpy(z["tr"]), torch.from_numpy(z["te"])
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    tr = tok("\n\n".join(ds["train"]["text"]), return_tensors="pt").input_ids[0]
    te = tok("\n\n".join(ds["test"]["text"]), return_tensors="pt").input_ids[0]
    np.savez(p, tr=tr.numpy(), te=te.numpy())
    return tr, te


def blocks_of(model):
    inner = model.model
    return inner.decoder.layers if hasattr(inner, "decoder") else inner.layers


def linears(model):
    out = []
    for bi, blk in enumerate(blocks_of(model)):
        for n, m in blk.named_modules():
            if isinstance(m, nn.Linear):
                out.append((bi, f"layers.{bi}.{n}", m))
    return out


# ------------------------------------------------------------------------------------------ quantizers
def rtn_group(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    o, i = w.shape
    wf = w.float().reshape(o, i // group, group)
    lo, hi = wf.amin(2, keepdim=True), wf.amax(2, keepdim=True)
    nl = 2 ** bits - 1
    s = ((hi - lo) / nl).clamp_min(1e-12)
    return (torch.clamp(torch.round((wf - lo) / s), 0, nl) * s + lo).reshape(o, i).to(w.dtype)


@torch.no_grad()
def gptq_layer(W: torch.Tensor, H: torch.Tensor, bits: int, group: int, damp: float) -> torch.Tensor:
    """GPTQ with group-wise asymmetric min-max scales (group == block size, aligned)."""
    W = W.float().clone()
    H = H.clone()
    cols = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    nl = 2 ** bits - 1
    for i1 in range(0, cols, group):
        i2 = min(i1 + group, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        lo = W1.amin(1, keepdim=True)
        hi = W1.amax(1, keepdim=True)
        s = ((hi - lo) / nl).clamp_min(1e-12)
        for j in range(i2 - i1):
            w = W1[:, j:j + 1]
            q = torch.clamp(torch.round((w - lo) / s), 0, nl) * s + lo
            Q1[:, j:j + 1] = q
            err = (w - q) / Hi[j, j]
            W1[:, j:] -= err @ Hi[j:j + 1, j:]
            E1[:, j:j + 1] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


class _StopForward(Exception):
    pass


# ------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def perplexity(model, ids: torch.Tensor, ctx: int, max_tokens: int = 0) -> float:
    if max_tokens:
        ids = ids[:max_tokens]
    n = len(ids) // ctx
    nll, cnt = 0.0, 0
    for k in range(n):
        x = ids[k * ctx:(k + 1) * ctx].unsqueeze(0).to(DEV)
        logits = model(input_ids=x).logits[0, :-1]
        nll += float(nn.functional.cross_entropy(logits.float(), x[0, 1:], reduction="sum"))
        cnt += ctx - 1
        del logits
    return math.exp(nll / cnt)


# ------------------------------------------------------------------------------------------ main
def load_model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(HF[args.model], dtype=torch.float16).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def allocation(name: str, L: int) -> np.ndarray:
    if name in ("rtn3", "gptq3"):
        return np.full(L, 3)
    if name in ("rtn4", "gptq4"):
        return np.full(L, 4)
    runs = [json.loads(l) for l in open(os.path.join(ROOT, "results_ext", args.alloc_src, args.model, "runs.jsonl"))]
    runs = [r for r in runs if r["method"] == name and abs(r["budget"] - 0.75) < 1e-6]
    best = max(runs, key=lambda r: r["calib_score"])  # selected on calibration data only
    return np.array(best["config"])


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF[args.model])
    tr, te = tokens(tok)
    rng = np.random.RandomState(1)
    starts = rng.randint(0, len(tr) - args.seqlen - 1, size=args.nsamples)
    calib = torch.stack([tr[s:s + args.seqlen] for s in starts])
    out_p = os.path.join(OUT, f"{args.model}{args.tag}{'_extra' if args.extra else ''}.json")
    tasks = c4 = None
    if args.extra:
        from ext_llm_eval import c4_tokens, load_tasks
        tasks = load_tasks()
        c4 = c4_tokens(tok, DATA)
        print("[extra] tasks:", {k: len(v[0]) for k, v in tasks.items()}, "c4 tokens:", len(c4), flush=True)
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    print(f"[gptq] model={args.model} device={DEV} linalg={LINALG_DEV}", flush=True)

    for name in args.allocs:
        if name in res:
            continue
        t0 = time.time()
        model = load_model()
        lin = linears(model)
        L = len(lin)
        if name == "fp16":
            bits = None
        else:
            bits = allocation(name.replace("gptq_", ""), L)
        use_gptq = name not in ("fp16", "rtn3", "rtn4")
        if bits is not None and not use_gptq:
            for k, (_, _, m) in enumerate(lin):
                m.weight.data = rtn_group(m.weight.data, int(bits[k]), args.group)
        elif bits is not None:
            nblocks = len(blocks_of(model))
            for b in range(nblocks):
                idx = [k for k, (bi, _, _) in enumerate(lin) if bi == b]
                Hs = {k: torch.zeros(lin[k][2].in_features, lin[k][2].in_features, device=DEV,
                                     dtype=torch.float32) for k in idx}
                cnt = {"n": 0}
                hooks = []
                for k in idx:
                    def hk(mod, inp, _k=k):
                        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                        Hs[_k] += x.t() @ x
                    hooks.append(lin[k][2].register_forward_pre_hook(hk))
                with torch.no_grad():
                    bs = 4 if DEV.type != "cpu" else 2
                    for s in range(0, len(calib), bs):
                        x = calib[s:s + bs].to(DEV)
                        # run only up to (and including) block b: stop afterwards to save time

                        def stopper(mod, inp, out):
                            raise _StopForward
                        hstop = blocks_of(model)[b].register_forward_hook(stopper)
                        try:
                            model(input_ids=x)
                        except _StopForward:
                            pass
                        hstop.remove()
                        cnt["n"] += x.numel()
                for h in hooks:
                    h.remove()
                for k in idx:
                    m = lin[k][2]
                    H = (2.0 / cnt["n"]) * Hs[k].to(LINALG_DEV)
                    Q = gptq_layer(m.weight.data.to(LINALG_DEV), H, int(bits[k]), args.group, args.damp)
                    m.weight.data = Q.to(DEV, dtype=m.weight.dtype)
                del Hs
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                elif DEV.type == "mps":
                    torch.mps.empty_cache()
        npar = np.array([m.weight.numel() for _, _, m in lin], dtype=np.float64)
        avg_bits = float(np.sum(bits * npar) / npar.sum()) if bits is not None else 16.0
        ppl_std = perplexity(model, te, args.eval_ctx)
        ppl_search = perplexity(model, te, 512, max_tokens=128 * 512)
        extra = {}
        if args.extra:
            from ext_llm_eval import eval_zero_shot, perplexity as ppl_fn
            extra["ppl_c4_ctx2048"] = ppl_fn(model, c4, DEV)
            extra["zero_shot"] = eval_zero_shot(model, tok, DEV, tasks)
            print(f"[extra] {name} c4={extra['ppl_c4_ctx2048']:.2f} zs={extra['zero_shot']}", flush=True)
        res[name] = {**extra, "avg_bits": avg_bits, "rounding": "gptq" if use_gptq else ("none" if bits is None else "rtn"),
                     "ppl_wikitext2_ctx2048_full": ppl_std, "ppl_wikitext2_ctx512_65k": ppl_search,
                     "wall_s": time.time() - t0, "nsamples": args.nsamples, "seqlen": args.seqlen}
        print(f"[gptq] {name:12s} bits={avg_bits:.2f} ppl(2048,full)={ppl_std:.2f} "
              f"ppl(512,65k)={ppl_search:.2f} {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out_p, "w"), indent=2)
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
