"""ABC-Q on a tiny LM: same gradient-free discrete search as CNN CIFAR code, scaled for sequences.

Why a small RNN-LM first (not a 7B Transformer here):
- Your repo is TensorFlow/Keras; full LLM ABC search needs enormous calibration forward passes.
- This file proves the *method* carries over: per-layer b_i in {2,4,8}, uniform min–max weight snap,
  f(b)=Acc - λ*BOPs - μ*Mem, sensitivity-biased mutations, employed/onlooker/scout ABC.
- Mapping to real LLMs / ViTs:
  * LLM: quantize each ``Dense`` / linear projection (and optionally ``Embedding``); treat each
    *layer object* as one search dimension (sub-weights share one b_i). True LLMs use PyTorch +
    blocked KV-cache latencies — swap ``evaluate_bit_config`` for perplexity on a text stream.
  * ViT: add ``Conv2D`` patch stem + transformer ``Dense`` blocks; same ABC loop as
    ``abc_q_cifar10_full`` with a ViT forward (or share utils via quantizable-layer list).

Sensitivity: one backward w.r.t. LM loss (optional gradient-free proxy: Fisher diagonal or weight
norm). Pipeline wording for papers: "gradient-free *bit allocation*"; pretrain/fine-tune still use
gradients on weights unless you adopt a fully grad-free weight update elsewhere.

Run: python abc_q_llm_tiny.py

Data: character-level LM on real text (Shakespeare by default). Random i.i.d. tokens are
unlearnable (~1/V chance accuracy) — not suitable for papers.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from scipy.special import softmax
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.optimizers import Adam

import abc_q_cifar10_full as abcq


QUICK_DEMO = True
SEED = 42
SEQ_LEN = 64
EMBED_DIM = 128
GRU_UNITS = 256
GRU_LAYERS = 2
CALIB_BATCH = 64
# Softer hardware penalty on this toy task so ABC does not collapse to all-2-bit when noise exists.
LAM_LM = 0.08
MU_LM = 0.03

_SHAKESPEARE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
# Offline / firewall fallback (~6k chars): still structured natural language.
_FALLBACK_CORPUS = """From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decease,
His tender heir might bear his memory:
But thou, contracted to thine own bright eyes,
Feed'st thy light's flame with self-substantial fuel,
Making a famine where abundance lies,
Thyself thy foe, to thy sweet self too cruel.
Thou that art now the world's fresh ornament,
And only herald to the gaudy spring,
Within thine own bud buriest thy content,
And, tender churl, mak'st waste in niggarding.
Pity the world, or else this glutton be,
To eat the world's due, by the grave and thee.
When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
""" * 8


@dataclass
class SearchResult:
    """Best bit config stats (token accuracy on calib for LM)."""

    bit_config: np.ndarray
    fitness: float
    accuracy: float
    bops_ratio: float
    mem_ratio: float


def set_seed(seed: int) -> None:
    """Fix RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_quantizable_layers_lm(model: Model) -> List[int]:
    """Quantize Embedding, GRU/LSTM, and Dense weights (one b_i per Keras layer)."""
    quant_types = (
        tf.keras.layers.Embedding,
        tf.keras.layers.Dense,
        tf.keras.layers.GRU,
        tf.keras.layers.LSTM,
    )
    return [i for i, ly in enumerate(model.layers) if isinstance(ly, quant_types) and ly.get_weights()]


def load_corpus_text() -> str:
    """Load Shakespeare text; use embedded excerpt if download fails."""
    try:
        path = tf.keras.utils.get_file(origin=_SHAKESPEARE_URL, cache_subdir="mimaize_lm")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded corpus from file ({len(text)} chars).")
        return text
    except Exception as e:
        print(f"Corpus download failed ({e}); using embedded fallback.")
        return _FALLBACK_CORPUS


def build_char_lm_arrays(
    text: str,
    seq_len: int,
    stride: int,
    max_chars: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Overlapping char sequences: inputs (N, seq_len-1), targets (N, seq_len-1)."""
    text = text[:max_chars]
    chars = sorted(set(text))
    n_vocab = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    enc = np.array([c2i[c] for c in text], dtype=np.int32)
    span = seq_len
    starts = list(range(0, len(enc) - span, stride))
    rng.shuffle(starts)
    xs, ys = [], []
    for i in starts:
        w = enc[i : i + span]
        xs.append(w[:-1])
        ys.append(w[1:])
    x_arr = np.stack(xs).astype(np.int32)
    y_arr = np.stack(ys).astype(np.int32)
    return x_arr, y_arr, chars, n_vocab


def train_calib_split(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random split of sequence batches (same distribution)."""
    n = len(x)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    tr = idx[:n_train]
    ca = idx[n_train:]
    return x[tr], y[tr], x[ca], y[ca]


def build_tiny_rnn_lm(
    vocab_size: int,
    seq_in_len: int = SEQ_LEN - 1,
    embed_dim: int = EMBED_DIM,
    gru_units: int = GRU_UNITS,
    n_gru: int = GRU_LAYERS,
) -> Model:
    """Tiny next-token RNN LM: inputs (batch, T-1), targets (batch, T-1) int token ids."""
    inp = Input(shape=(seq_in_len,), dtype="int32")
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False)(inp)
    for _ in range(n_gru):
        x = GRU(gru_units, return_sequences=True)(x)
    out = Dense(vocab_size, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_lm_dataset(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> tf.data.Dataset:
    """tf.data for (x, y) int32 pairs."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def HARDWARE_COST_COMPONENTS(
    bit_config: np.ndarray,
    param_counts: np.ndarray,
) -> Tuple[float, float]:
    """Same BOPs/Mem ratio as CNN code (param-weighted bit sum vs all-8-bit)."""
    numerator = float(np.sum(bit_config.astype(np.float64) * param_counts))
    denominator = float(np.sum(8.0 * param_counts))
    r = numerator / denominator if denominator else 1.0
    return r, r


def evaluate_lm_bit_config(
    model: Model,
    quantizable_layers: Sequence[int],
    bit_config: np.ndarray,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    original_weights: Dict[int, List[np.ndarray]],
) -> Tuple[float, float, float]:
    """Apply quantization, token accuracy on calib, restore float weights."""
    abcq.apply_bit_config(model, quantizable_layers, bit_config, original_weights)
    ds = make_lm_dataset(x_calib, y_calib, CALIB_BATCH, shuffle=False)
    _, acc = model.evaluate(ds, verbose=0)
    abcq.restore_original_weights(model, quantizable_layers, original_weights)
    bop, mem = HARDWARE_COST_COMPONENTS(bit_config, param_counts)
    return float(acc), float(bop), float(mem)


def compute_fitness_lm(acc: float, bops_ratio: float, mem_ratio: float) -> float:
    """f(b) = Acc - λ BOPs - μ Mem (LM-specific λ,μ so search does not chase only compression)."""
    return acc - LAM_LM * bops_ratio - MU_LM * mem_ratio


def eval_lm_loss_acc(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Mean NLL (loss) and token accuracy on (x,y)."""
    ds = make_lm_dataset(x, y, CALIB_BATCH, shuffle=False)
    loss, acc = model.evaluate(ds, verbose=0)
    return float(loss), float(acc)


def compute_lm_sensitivity(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
) -> np.ndarray:
    """Normalized gradient magnitude per quantizable layer (one-time prior)."""
    x_b = x_calib[:32]
    y_b = y_calib[:32]
    with tf.GradientTape() as tape:
        preds = model(x_b, training=False)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_b, preds),
        )
    train_vars = model.trainable_variables
    grads = tape.gradient(loss, train_vars)
    grad_map = {id(v): g for v, g in zip(train_vars, grads)}

    sensitivities = []
    for idx in quantizable_layers:
        layer = model.layers[idx]
        mags = []
        for var in layer.trainable_weights:
            g = grad_map.get(id(var))
            if g is not None:
                mags.append(float(tf.reduce_mean(tf.abs(g)).numpy()))
        sensitivities.append(float(np.mean(mags)) if mags else 0.0)

    sens = np.array(sensitivities, dtype=np.float64)
    if np.allclose(sens.sum(), 0.0):
        sens = np.ones_like(sens) / len(sens)
    else:
        sens = sens / sens.sum()
    return sens


def run_abc_q_lm(
    model: Model,
    quantizable_layers: Sequence[int],
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    param_counts: np.ndarray,
    sensitivity: np.ndarray,
    num_bees: int,
    cycles: int,
    scout_limit: int,
) -> Tuple[SearchResult, int]:
    """ABC loop for LM (same phases as CNN; fitness uses token accuracy)."""
    original_weights = {i: copy.deepcopy(model.layers[i].get_weights()) for i in quantizable_layers}
    num_layers = len(quantizable_layers)
    foods = abcq.initialize_food_sources(num_bees, num_layers, sensitivity)
    trials = np.zeros(num_bees, dtype=np.int32)
    scores = np.zeros(num_bees, dtype=np.float64)
    accs = np.zeros(num_bees, dtype=np.float64)
    bops = np.zeros(num_bees, dtype=np.float64)
    mems = np.zeros(num_bees, dtype=np.float64)
    eval_count = 0

    for i in range(num_bees):
        acc, bop, mem = evaluate_lm_bit_config(
            model, quantizable_layers, foods[i], x_calib, y_calib, param_counts, original_weights
        )
        scores[i] = compute_fitness_lm(acc, bop, mem)
        accs[i], bops[i], mems[i] = acc, bop, mem
        eval_count += 1

    best_idx = int(np.argmax(scores))
    best = SearchResult(
        bit_config=foods[best_idx].copy(),
        fitness=float(scores[best_idx]),
        accuracy=float(accs[best_idx]),
        bops_ratio=float(bops[best_idx]),
        mem_ratio=float(mems[best_idx]),
    )

    for cycle in range(1, cycles + 1):
        scouts_triggered = 0
        for i in range(num_bees):
            cand = abcq.mutate_config(foods[i], sensitivity)
            acc, bop, mem = evaluate_lm_bit_config(
                model, quantizable_layers, cand, x_calib, y_calib, param_counts, original_weights
            )
            fit = compute_fitness_lm(acc, bop, mem)
            eval_count += 1
            if fit > scores[i]:
                foods[i], scores[i] = cand, fit
                accs[i], bops[i], mems[i] = acc, bop, mem
                trials[i] = 0
            else:
                trials[i] += 1

        probs = softmax(scores)
        for _ in range(num_bees):
            i = int(np.random.choice(num_bees, p=probs))
            cand = abcq.mutate_config(foods[i], sensitivity)
            acc, bop, mem = evaluate_lm_bit_config(
                model, quantizable_layers, cand, x_calib, y_calib, param_counts, original_weights
            )
            fit = compute_fitness_lm(acc, bop, mem)
            eval_count += 1
            if fit > scores[i]:
                foods[i], scores[i] = cand, fit
                accs[i], bops[i], mems[i] = acc, bop, mem
                trials[i] = 0
            else:
                trials[i] += 1

        thresh = np.quantile(sensitivity, 0.75)
        sens_mask = sensitivity >= thresh
        for i in range(num_bees):
            if trials[i] >= scout_limit:
                print(f"[Scout] Resetting bee {i} after {int(trials[i])} trials.")
                reset = np.random.choice(abcq.BITS, size=num_layers, replace=True).astype(np.int32)
                for d in np.where(sens_mask)[0]:
                    if reset[d] < 4:
                        reset[d] = np.random.choice([4, 8])
                acc, bop, mem = evaluate_lm_bit_config(
                    model, quantizable_layers, reset, x_calib, y_calib, param_counts, original_weights
                )
                fit = compute_fitness_lm(acc, bop, mem)
                foods[i], scores[i] = reset, fit
                accs[i], bops[i], mems[i] = acc, bop, mem
                trials[i] = 0
                eval_count += 1
                scouts_triggered += 1

        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best.fitness:
            best = SearchResult(
                bit_config=foods[best_idx].copy(),
                fitness=float(scores[best_idx]),
                accuracy=float(accs[best_idx]),
                bops_ratio=float(bops[best_idx]),
                mem_ratio=float(mems[best_idx]),
            )
        print(
            f"[LM ABC cycle {cycle}] best acc={best.accuracy:.4f} fitness={best.fitness:.4f} "
            f"BOPs={best.bops_ratio:.4f} | scouts={scouts_triggered}"
        )

    abcq.restore_original_weights(model, quantizable_layers, original_weights)
    return best, eval_count


def main_q() -> None:
    set_seed(SEED)
    rng = np.random.RandomState(SEED)

    if QUICK_DEMO:
        max_chars = 250_000
        stride = 2
        pretrain_epochs = 22
        train_frac = 0.88
        bees, cycles, scout = 8, 6, 4
    else:
        max_chars = 900_000
        stride = 1
        pretrain_epochs = 45
        train_frac = 0.9
        bees, cycles, scout = 10, 12, 5

    text = load_corpus_text()
    x_all, y_all, char_list, vocab = build_char_lm_arrays(
        text, SEQ_LEN, stride=stride, max_chars=max_chars, rng=rng
    )
    x_train, y_train, x_calib, y_calib = train_calib_split(x_all, y_all, train_frac, rng)
    chance = 1.0 / vocab

    print(
        f"Char LM: vocab={vocab} | train_seq={len(x_train)} calib_seq={len(x_calib)} | "
        f"random-guess acc≈{chance:.4f}"
    )

    model = build_tiny_rnn_lm(vocab_size=vocab, seq_in_len=SEQ_LEN - 1)
    train_ds = make_lm_dataset(x_train, y_train, 64, shuffle=True)
    cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )
    model.fit(train_ds, epochs=pretrain_epochs, verbose=2, callbacks=[cb])

    loss_f, acc_f = eval_lm_loss_acc(model, x_calib, y_calib)
    perp_f = float(np.exp(loss_f))
    print(f"Float model — calib loss={loss_f:.4f} perplexity={perp_f:.2f} token_acc={acc_f:.4f}")

    qidx = get_quantizable_layers_lm(model)
    pcounts = abcq.layer_param_counts(model, qidx)
    print(f"Quantizable LM layers: {len(qidx)} (Embedding + GRU(s) + Dense)")

    sens = compute_lm_sensitivity(model, qidx, x_calib, y_calib)
    best, _budget = run_abc_q_lm(
        model,
        qidx,
        x_calib,
        y_calib,
        pcounts,
        sens,
        num_bees=bees,
        cycles=cycles,
        scout_limit=scout,
    )
    print(f"ABC-Q best — calib token_acc={best.accuracy:.4f} bits={abcq.format_bits(best.bit_config)}")
    for name, cfg in [("uniform 8", np.full(len(qidx), 8, np.int32)), ("uniform 4", np.full(len(qidx), 4, np.int32))]:
        ow = {i: copy.deepcopy(model.layers[i].get_weights()) for i in qidx}
        oa, ob, _om = evaluate_lm_bit_config(model, qidx, cfg, x_calib, y_calib, pcounts, ow)
        print(f"  Baseline {name}: calib acc={oa:.4f} BOPs={ob:.4f}")

    abcq.apply_best_config_permanently(model, qidx, best.bit_config)
    loss_q, acc_q = eval_lm_loss_acc(model, x_calib, y_calib)
    print(f"After ABC quant (baked in): calib loss={loss_q:.4f} perplexity={np.exp(loss_q):.2f} token_acc={acc_q:.4f}")

    out_path = "tiny_lm_abc_quantized.keras"
    model.save(out_path)
    print(f"Saved mixed-precision tiny LM to {out_path}")
    print(
        "\nScale-up checklist for real LLMs/ViTs: "
        "(1) replace toy data + pretrain; "
        "(2) calib = held-out text windows; "
        "(3) fitness Acc -> -val_perplexity or BLEU slice; "
        "(4) extend get_quantizable_layers_lm for attention Linear layers; "
        "(5) expect huge ABC eval cost — prune layer groups or use proxy models."
    )


if __name__ == "__main__":
    main_q()
