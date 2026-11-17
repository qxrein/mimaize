"""ABC-Q v2 — budget-constrained mixed-precision search algorithms.

Problem:  maximise  Acc_calib(b)  subject to  BOPs(b) <= B * BOPs(W8A8),  b in {2,4,8}^L.

Every method receives the same number of *distinct* fitness evaluations ``E`` (calibration forward
passes). Re-visiting an already evaluated configuration is served from a cache and costs nothing,
for every method. Methods that use single-layer probes pay for them out of the same budget.

Methods
-------
* ``random``   uniform random configurations + random feasibility repair.
* ``ga``       generational GA (tournament, uniform crossover, per-gene mutation, elitism) + repair.
* ``cmaes``    CMA-ES on a continuous relaxation in [0, 2]^L, rounded to {2,4,8} and repaired.
* ``abc_v1``   the original ABC-Q operators (gradient prior, 4-bit floor on top-25% layers,
               single-dimension mutation with p(d) ∝ 1 - s̃_d, softmax onlookers, random scouts)
               plus the same random repair, so that it can be run under a hard budget.
* ``fisher_mckp``  one-shot HAWQ-V2-style allocation: Ω_i(b) = Tr(F_i)·||Q_b(W_i) - W_i||², solved
               exactly as a multiple-choice knapsack (no search; 1 evaluation).
* ``probe_mckp``   one-shot allocation from measured single-layer drops Δ_i(b), solved exactly as a
               multiple-choice knapsack (2L+1 probe evaluations + 1).
* ``abc_v2``   proposed: probe-initialised, surrogate-assisted discrete ABC (see ``run_abc_v2``).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

BITS = np.array([2, 4, 8], dtype=np.int64)
BIT_IDX = {2: 0, 4: 1, 8: 2}


# ============================================================================ budgeted oracle
class BudgetExhausted(Exception):
    pass


class Oracle:
    """Wraps the evaluator: caching, budget accounting, best-feasible trace."""

    def __init__(self, acc_fn: Callable[[np.ndarray], float], cost: np.ndarray, budget_B: float,
                 max_evals: int):
        self.acc_fn = acc_fn
        self.cost = cost  # (L, 3) fraction of W8A8 BOPs
        self.B = budget_B
        self.max_evals = int(max_evals)
        self.cache: Dict[bytes, float] = {}
        self.n = 0
        self.best_acc = -np.inf
        self.best_cfg: Optional[np.ndarray] = None
        self.trace: List[float] = []  # best feasible acc after each distinct evaluation
        self.history: List[Tuple[np.ndarray, float]] = []

    def bops(self, cfg: np.ndarray) -> float:
        return float(self.cost[np.arange(len(cfg)), [BIT_IDX[int(b)] for b in cfg]].sum())

    def feasible(self, cfg: np.ndarray) -> bool:
        return self.bops(cfg) <= self.B + 1e-12

    def remaining(self) -> int:
        return self.max_evals - self.n

    def __call__(self, cfg: np.ndarray, record: bool = True) -> float:
        cfg = np.asarray(cfg, dtype=np.int64)
        key = cfg.tobytes()
        if key in self.cache:
            return self.cache[key]
        if self.n >= self.max_evals:
            raise BudgetExhausted
        acc = float(self.acc_fn(cfg))
        self.n += 1
        self.cache[key] = acc
        self.history.append((cfg.copy(), acc))
        if record and self.feasible(cfg) and acc > self.best_acc:
            self.best_acc, self.best_cfg = acc, cfg.copy()
        self.trace.append(self.best_acc)
        return acc


# ============================================================================ repair operators
def random_repair(cfg: np.ndarray, oracle: Oracle, rng: np.random.RandomState) -> np.ndarray:
    """Lower a random layer (probability ∝ BOPs saved) by one level until feasible."""
    c = cfg.copy()
    while not oracle.feasible(c):
        idx = np.where(c > 2)[0]
        if len(idx) == 0:
            break
        save = np.array([oracle.cost[i, BIT_IDX[int(c[i])]] - oracle.cost[i, BIT_IDX[int(c[i])] - 1] for i in idx])
        p = save / save.sum()
        i = idx[rng.choice(len(idx), p=p)]
        c[i] = BITS[BIT_IDX[int(c[i])] - 1]
    return c


def surrogate_repair(cfg: np.ndarray, oracle: Oracle, w: np.ndarray, frozen: Optional[int] = None) -> np.ndarray:
    """Lower the layer with the smallest predicted accuracy loss per BOP saved until feasible."""
    c = cfg.copy()
    while not oracle.feasible(c):
        best, bi = np.inf, -1
        for i in range(len(c)):
            j = BIT_IDX[int(c[i])]
            if j == 0 or i == frozen:
                continue
            dloss = w[i, j - 1] - w[i, j]
            dsave = oracle.cost[i, j] - oracle.cost[i, j - 1]
            r = dloss / max(dsave, 1e-12)
            if r < best:
                best, bi = r, i
        if bi < 0:
            if frozen is not None:
                frozen = None
                continue
            break
        c[bi] = BITS[BIT_IDX[int(c[bi])] - 1]
    return c


def surrogate_fill(cfg: np.ndarray, oracle: Oracle, w: np.ndarray, frozen: Optional[int] = None) -> np.ndarray:
    """Spend leftover budget: raise the layer with best predicted gain per BOP while feasible."""
    c = cfg.copy()
    slack = oracle.B - oracle.bops(c)
    while True:
        best, bi = 0.0, -1
        for i in range(len(c)):
            j = BIT_IDX[int(c[i])]
            if j == 2 or i == frozen:
                continue
            dcost = oracle.cost[i, j + 1] - oracle.cost[i, j]
            if dcost > slack:
                continue
            gain = w[i, j] - w[i, j + 1]
            r = gain / max(dcost, 1e-12)
            if r > best:
                best, bi = r, i
        if bi < 0:
            break
        j = BIT_IDX[int(c[bi])]
        slack -= oracle.cost[bi, j + 1] - oracle.cost[bi, j]
        c[bi] = BITS[j + 1]
    return c


# ============================================================================ knapsack
def mckp(w: np.ndarray, cost: np.ndarray, B: float, grid: int = 4000) -> np.ndarray:
    """Exact multiple-choice knapsack (DP on a discretised cost grid): minimise Σ w[i, j_i]
    subject to Σ cost[i, j_i] <= B. Costs are rounded *up*, so the result is always feasible."""
    L = w.shape[0]
    unit = B / grid
    ic = np.ceil(cost / unit - 1e-9).astype(np.int64)
    INF = 1e18
    dp = np.full(grid + 1, INF)
    dp[0] = 0.0
    choice = np.zeros((L, grid + 1), dtype=np.int8)
    for i in range(L):
        nd = np.full(grid + 1, INF)
        nc = np.zeros(grid + 1, dtype=np.int8)
        for j in range(cost.shape[1]):
            c = ic[i, j]
            if c > grid:
                continue
            cand = np.full(grid + 1, INF)
            cand[c:] = dp[: grid + 1 - c] + w[i, j]
            better = cand < nd
            nd[better] = cand[better]
            nc[better] = j
        dp = nd
        choice[i] = nc
    k = int(np.argmin(dp))
    if dp[k] >= INF:
        return np.full(L, 2, dtype=np.int64)
    sel = np.zeros(L, dtype=np.int64)
    for i in range(L - 1, -1, -1):
        j = int(choice[i, k])
        sel[i] = j
        k -= ic[i, j]
    return BITS[sel]


# ============================================================================ surrogate
class AdditiveSurrogate:
    """Acc(b) ≈ a0 - Σ_i w[i, b_i]  with w[i, 8] = 0.

    Prior weights come from single-layer probes; after each real evaluation the weights are refit
    by ridge regression *shrunk towards the prior* (MAP estimate under a Gaussian prior centred on
    the probe values), so interactions between layers are absorbed into the effective weights.
    """

    def __init__(self, a0: float, w0: np.ndarray, alpha: float = 1.0):
        self.a0 = a0
        self.w0 = w0.copy()
        self.w = w0.copy()
        self.alpha = alpha
        self.L = w0.shape[0]

    def features(self, cfgs: np.ndarray) -> np.ndarray:
        X = np.zeros((len(cfgs), 2 * self.L))
        for r, c in enumerate(cfgs):
            for i, b in enumerate(c):
                j = BIT_IDX[int(b)]
                if j < 2:
                    X[r, 2 * i + j] = 1.0
        return X

    def fit(self, cfgs: List[np.ndarray], accs: List[float]) -> None:
        if not cfgs:
            return
        X = self.features(np.array(cfgs))
        y = self.a0 - np.asarray(accs)
        w0 = self.w0[:, :2].reshape(-1)
        A = X.T @ X + self.alpha * np.eye(X.shape[1])
        rhs = X.T @ y + self.alpha * w0
        wf = np.linalg.solve(A, rhs)
        self.w = np.zeros_like(self.w0)
        self.w[:, :2] = wf.reshape(self.L, 2)

    def predict(self, cfg: np.ndarray) -> float:
        return self.a0 - float(sum(self.w[i, BIT_IDX[int(b)]] for i, b in enumerate(cfg)))


# ============================================================================ baselines
def run_random(oracle: Oracle, L: int, rng: np.random.RandomState) -> None:
    try:
        while True:
            c = rng.choice(BITS, size=L)
            oracle(random_repair(c, oracle, rng))
    except BudgetExhausted:
        pass


def run_ga(oracle: Oracle, L: int, rng: np.random.RandomState, pop: int = 20, pc: float = 0.9,
           tour: int = 3, elite: int = 2) -> None:
    pm = 1.0 / L
    try:
        P = [random_repair(rng.choice(BITS, size=L), oracle, rng) for _ in range(pop)]
        F = [oracle(c) for c in P]
        while True:
            order = np.argsort(F)[::-1]
            newP = [P[k].copy() for k in order[:elite]]
            while len(newP) < pop:
                def sel():
                    idx = rng.choice(pop, size=tour, replace=False)
                    return P[idx[np.argmax([F[k] for k in idx])]]
                a, b = sel(), sel()
                if rng.rand() < pc:
                    m = rng.rand(L) < 0.5
                    child = np.where(m, a, b)
                else:
                    child = a.copy()
                mut = rng.rand(L) < pm
                for d in np.where(mut)[0]:
                    child[d] = rng.choice([x for x in BITS if x != child[d]])
                newP.append(random_repair(child, oracle, rng))
            P = newP
            F = [oracle(c) for c in P]
    except BudgetExhausted:
        pass


def run_cmaes(oracle: Oracle, L: int, rng: np.random.RandomState, seed: int) -> None:
    import cma

    def to_cfg(z):
        idx = np.clip(np.round(np.clip(z, 0, 2)), 0, 2).astype(np.int64)
        return BITS[idx]

    es = cma.CMAEvolutionStrategy(np.ones(L), 0.6,
                                  {"bounds": [0, 2], "seed": int(seed) + 1, "verbose": -9, "verb_log": 0})
    try:
        while True:
            Z = es.ask()
            fits = []
            for z in Z:
                c = random_repair(to_cfg(np.asarray(z)), oracle, rng)
                fits.append(-oracle(c))
            es.tell(Z, fits)
    except BudgetExhausted:
        pass


def run_abc_v1(oracle: Oracle, L: int, rng: np.random.RandomState, grad_prior: np.ndarray,
               N: int = 16, limit: int = 8, q_tau: float = 0.75) -> None:
    """Original ABC-Q operators (paper v1) + random repair to respect the hard budget."""
    thr = np.quantile(grad_prior, q_tau)
    sens = grad_prior >= thr
    pmut = np.clip(1.0 - grad_prior, 1e-8, None)
    pmut = pmut / pmut.sum()

    def new_source():
        c = rng.choice(BITS, size=L)
        for d in np.where(sens)[0]:
            if c[d] == BITS[0]:  # sensitivity floor: never the lowest bit-width
                c[d] = rng.choice(BITS[1:])
        return random_repair(c, oracle, rng)

    def mutate(c):
        n = c.copy()
        d = rng.choice(L, p=pmut)
        n[d] = rng.choice([x for x in BITS if x != c[d]])
        return random_repair(n, oracle, rng)

    try:
        foods = [new_source() for _ in range(N)]
        fit = np.array([oracle(c) for c in foods])
        trials = np.zeros(N, dtype=int)
        while True:
            for i in range(N):
                cand = mutate(foods[i])
                f = oracle(cand)
                if f > fit[i]:
                    foods[i], fit[i], trials[i] = cand, f, 0
                else:
                    trials[i] += 1
            e = np.exp(fit - fit.max())
            probs = e / e.sum()  # v1 used plain softmax of fitness (nearly uniform)
            for _ in range(N):
                i = rng.choice(N, p=probs)
                cand = mutate(foods[i])
                f = oracle(cand)
                if f > fit[i]:
                    foods[i], fit[i], trials[i] = cand, f, 0
                else:
                    trials[i] += 1
            for i in range(N):
                if trials[i] >= limit:
                    foods[i] = new_source()
                    fit[i] = oracle(foods[i])
                    trials[i] = 0
    except BudgetExhausted:
        pass


# ============================================================================ proposed: ABC-Q v2
def run_abc_v2(oracle: Oracle, L: int, rng: np.random.RandomState, acc8: float, delta: np.ndarray,
               N: int = 12, limit: int = 6, K: int = 6, p_share: float = 0.5, tau: float = 0.01,
               init_noise: float = 0.5, alpha: float = 1.0, learn: bool = True, screen: bool = True,
               knapsack_init: bool = True, share: bool = True) -> Dict[str, float]:
    """Probe-initialised, surrogate-assisted discrete ABC under a hard BOPs budget.

    1. Probes (2L evaluations, paid from the budget): Δ_i(b) for b ∈ {2,4}.
    2. Surrogate: additive model, prior = probes, refit by ridge-to-prior after every cycle.
    3. Food sources initialised as exact knapsack (MCKP) optima of *randomly perturbed* surrogates
       (source 0 = unperturbed optimum) → diverse, feasible, near the budget boundary.
    4. Neighbour operator (discrete analogue of v = x + φ(x - x_k)):
         with prob p_share copy the bit of a random other source k on a dimension where they differ;
         otherwise move one layer one level up/down. The changed layer is then frozen while the
         configuration is projected back onto the budget boundary (surrogate repair + fill), giving a
         budget-neutral "trade" between layers.
    5. Surrogate screening: K neighbours are generated, only the one with the best predicted
       accuracy is truly evaluated.
    6. Onlookers pick sources with a temperature-scaled softmax (τ = 0.01 on accuracy).
    7. Scouts: sources stagnant for ``limit`` trials are re-seeded from a perturbed-surrogate MCKP.
    """
    info = {"probe_evals": 0}
    sur = AdditiveSurrogate(acc8, delta, alpha=alpha)
    cost = oracle.cost

    def perturbed_mckp(noise: float) -> np.ndarray:
        if not knapsack_init:
            return random_repair(rng.choice(BITS, size=L), oracle, rng)
        wn = sur.w * (np.exp(noise * rng.randn(*sur.w.shape)) if noise > 0 else 1.0)
        wn[:, 2] = 0.0
        return mckp(wn, cost, oracle.B)

    def neighbour(c: np.ndarray, foods: List[np.ndarray], i: int) -> np.ndarray:
        n = c.copy()
        d = None
        if share and rng.rand() < p_share and len(foods) > 1:
            k = rng.choice([x for x in range(len(foods)) if x != i])
            diff = np.where(foods[k] != c)[0]
            if len(diff):
                d = int(rng.choice(diff))
                n[d] = foods[k][d]
        if d is None:
            d = int(rng.randint(L))
            j = BIT_IDX[int(c[d])]
            if j == 0:
                j2 = 1
            elif j == 2:
                j2 = 1
            else:
                j2 = j + (1 if rng.rand() < 0.5 else -1)
            n[d] = BITS[j2]
        n = surrogate_repair(n, oracle, sur.w, frozen=d)
        n = surrogate_fill(n, oracle, sur.w, frozen=d)
        return n

    def propose(c: np.ndarray, foods: List[np.ndarray], i: int) -> np.ndarray:
        if not screen:
            return neighbour(c, foods, i)
        cands = []
        for _ in range(K * 3):
            n = neighbour(c, foods, i)
            if n.tobytes() not in oracle.cache:
                cands.append(n)
            if len(cands) >= K:
                break
        if not cands:
            return neighbour(c, foods, i)
        preds = [sur.predict(n) for n in cands]
        return cands[int(np.argmax(preds))]

    def refit():
        if learn:
            cf = [h[0] for h in oracle.history]
            ac = [h[1] for h in oracle.history]
            sur.fit(cf, ac)

    try:
        foods: List[np.ndarray] = [perturbed_mckp(0.0)]
        while len(foods) < N:
            foods.append(perturbed_mckp(init_noise))
        fit = np.array([oracle(c) for c in foods])
        trials = np.zeros(N, dtype=int)
        refit()
        while True:
            n_before = oracle.n
            for i in range(N):
                cand = propose(foods[i], foods, i)
                f = oracle(cand)
                if f > fit[i]:
                    foods[i], fit[i], trials[i] = cand, f, 0
                else:
                    trials[i] += 1
            e = np.exp((fit - fit.max()) / tau)
            probs = e / e.sum()
            for _ in range(N):
                i = int(rng.choice(N, p=probs))
                cand = propose(foods[i], foods, i)
                f = oracle(cand)
                if f > fit[i]:
                    foods[i], fit[i], trials[i] = cand, f, 0
                else:
                    trials[i] += 1
            refit()
            stalled = oracle.n == n_before  # whole neighbourhood already evaluated: re-seed everything
            for i in range(N):
                if stalled or (trials[i] >= limit and i != int(np.argmax(fit))):
                    foods[i] = perturbed_mckp(init_noise)
                    fit[i] = oracle(foods[i])
                    trials[i] = 0
    except BudgetExhausted:
        pass
    return info
