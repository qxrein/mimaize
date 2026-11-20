"""Downstream evaluation for quantized LLMs: C4 perplexity and six zero-shot tasks.

Zero-shot scoring follows the log-likelihood protocol of lm-evaluation-harness:
  * ARC-Easy (acc), ARC-Challenge (acc_norm), HellaSwag (acc_norm), PIQA (acc), BoolQ (acc):
    pick the answer whose continuation has the highest (length-normalised for acc_norm) log-likelihood;
  * WinoGrande (acc): partial scoring -- each option is substituted for the blank and the log-likelihood
    of the remaining sentence is compared.
Full validation/test splits are used. C4: first 256 x 2048 tokens of the en validation shard 0 (GPTQ protocol).
"""

from __future__ import annotations

import math
import os
import re
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def _loglik(model, tok, dev, pairs: List[Tuple[str, str]], max_len: int = 1024) -> List[Tuple[float, int]]:
    """Sum log p(continuation | context) and continuation length in chars, batched per question."""
    seqs, cont_lens = [], []
    for ctx, cont in pairs:
        c_ids = tok(ctx, add_special_tokens=False).input_ids
        x_ids = tok(cont, add_special_tokens=False).input_ids
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        ids = (bos + c_ids + x_ids)[-max_len:]
        seqs.append(ids)
        cont_lens.append(len(x_ids))
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    inp = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    logits = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits.float()
    lp = torch.log_softmax(logits, -1)
    out = []
    for i, s in enumerate(seqs):
        n = cont_lens[i]
        tgt = torch.tensor(s[len(s) - n:], device=dev)
        pos = torch.arange(len(s) - n - 1, len(s) - 1, device=dev)
        out.append((float(lp[i, pos, tgt].sum()), len(pairs[i][1])))
    return out


def _mc_accuracy(model, tok, dev, items, norm: bool) -> float:
    correct = 0
    for ctx, choices, gold in items:
        ll = _loglik(model, tok, dev, [(ctx, c) for c in choices])
        scores = [l / max(n, 1) if norm else l for l, n in ll]
        correct += int(max(range(len(scores)), key=lambda k: scores[k]) == gold)
    return correct / len(items)


def _hs_pre(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_tasks() -> Dict[str, Tuple[List, bool]]:
    """Returns task -> (items, use_acc_norm). Items are (context, [continuations], gold_index)."""
    from datasets import load_dataset
    tasks = {}
    for cfg, name, norm in [("ARC-Easy", "arc_easy", False), ("ARC-Challenge", "arc_challenge", True)]:
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        items = []
        for r in ds:
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            items.append((f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                          labels.index(r["answerKey"])))
        tasks[name] = (items, norm)
    ds = load_dataset("Rowan/hellaswag", split="validation")
    tasks["hellaswag"] = ([(_hs_pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()),
                            [" " + _hs_pre(e) for e in r["endings"]], int(r["label"])) for r in ds], True)
    piqa = None
    for repo in ("baber/piqa", "ybisk/piqa"):
        try:
            piqa = load_dataset(repo, split="validation")
            break
        except Exception:
            continue
    if piqa is not None:
        tasks["piqa"] = ([(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"]))
                          for r in piqa], False)
    ds = load_dataset("google/boolq", split="validation")
    tasks["boolq"] = ([(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", [" no", " yes"], int(r["answer"]))
                       for r in ds], False)
    wg = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    wg_items = []
    for r in wg:
        s = r["sentence"]
        k = s.index("_")
        cont = " " + s[k + 1:].strip()
        wg_items.append(([s[:k] + r["option1"], s[:k] + r["option2"]], cont, int(r["answer"]) - 1))
    tasks["winogrande"] = (wg_items, False)
    return tasks


def eval_zero_shot(model, tok, dev, tasks) -> Dict[str, float]:
    res = {}
    for name, (items, norm) in tasks.items():
        if name == "winogrande":
            correct = 0
            for ctxs, cont, gold in items:
                ll = _loglik(model, tok, dev, [(c, cont) for c in ctxs])
                correct += int(max(range(2), key=lambda k: ll[k][0]) == gold)
            res[name] = correct / len(items)
        else:
            res[name] = _mc_accuracy(model, tok, dev, items, norm)
    res["average"] = sum(res.values()) / len(res)
    return res


def c4_tokens(tok, data_dir: str, n_tokens: int = 256 * 2048) -> torch.Tensor:
    import gzip
    import json
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset",
                        local_dir=os.path.join(data_dir, "c4"))
    texts = []
    with gzip.open(p, "rt") as f:
        for k, line in enumerate(f):
            if k >= 1100:
                break
            texts.append(json.loads(line)["text"])
    ids = tok(" ".join(texts), return_tensors="pt").input_ids[0]
    return ids[:n_tokens]


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, dev, ctx: int = 2048) -> float:
    n = len(ids) // ctx
    nll, cnt = 0.0, 0
    for k in range(n):
        x = ids[k * ctx:(k + 1) * ctx].unsqueeze(0).to(dev)
        lg = model(input_ids=x).logits[0, :-1].float()
        nll += float(nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum"))
        cnt += ctx - 1
    return math.exp(nll / cnt)
