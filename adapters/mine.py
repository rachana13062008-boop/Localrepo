"""
adapters/mine.py — Autoclave Queue solver.
"""

from __future__ import annotations
import time
import math
import random

from adapter import Solver, simulate, cost_of

TIME_LIMIT = 4.85  # Hard wall with a slight buffer under 5.0s
SUBMIT_CHECKPOINTS = (0.05, 0.20, 0.50, 0.95, 1.00)

class _Rng:
    """Fast xorshift RNG."""
    __slots__ = ("s",)
    def __init__(self, seed: int = 0x9E3779B97F4A7C15):
        self.s = seed & 0xFFFFFFFFFFFFFFFF or 1

    def next(self) -> int:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = s
        return s

    def randint(self, n: int) -> int:
        return (self.next() >> 11) % n if n > 0 else 0


class MySolver(Solver):
    def __init__(self, instance):
        super().__init__(instance)
        self.n = instance.size
        self.proc = list(instance.proc)
        self.rel = list(instance.release)
        self.due = list(instance.due)
        self.w = list(instance.weight)
        self.fam = list(instance.fam)
        self.setup = instance.setup

        self.MSPAN_W = 4
        self.rng = _Rng()
        self.best_order = None
        self.best_cost = math.inf
        self._t0 = time.perf_counter()
        self._next_submit = 0
        self._submit_func = None

    def _submit(self, order):
        if order is None or self._submit_func is None:
            return
        try:
            self._submit_func({"order": list(order)})
        except Exception:
            pass

    def _cost_of(self, order):
        return float(cost_of(self.instance, order))

    def _atcs_order(self, kappa=2.0, alpha=None):
        n = self.n
        proc, rel, due, w, fam, setup = self.proc, self.rel, self.due, self.w, self.fam, self.setup
        if alpha is None:
            alpha = kappa

        pbar = (sum(proc) / n) or 1.0
        allset = [setup[f][g] for f in range(len(setup)) for g in range(len(setup))]
        sbar = (sum(allset) / len(allset)) or 1.0

        remaining = list(range(n))
        remaining.sort(key=lambda b: rel[b])

        order = []
        t = 0.0
        prev_f = -1
        k1 = kappa * pbar
        k2 = alpha * sbar
        wp = [(w[b] / proc[b]) if proc[b] > 0 else w[b] * 1e6 for b in range(n)]

        while remaining:
            if t < rel[remaining[0]]:
                t = rel[remaining[0]]

            best_b, best_val, best_i = -1, -1.0, -1
            for i, b in enumerate(remaining):
                if rel[b] > t:
                    break
                sf = math.exp(-setup[prev_f][fam[b]] / k2) if prev_f >= 0 else 1.0
                slack = max(0.0, due[b] - proc[b] - t)
                val = wp[b] * math.exp(-slack / k1) * sf
                if val > best_val:
                    best_val, best_b, best_i = val, b, i

            if best_b < 0:
                best_i, best_b = 0, remaining[0]

            order.append(best_b)
            remaining.pop(best_i)
            prev_f = fam[best_b]
            if len(order) == 1:
                s = rel[best_b]
            else:
                s = max(t + setup[fam[order[-2]]][fam[best_b]], rel[best_b])
            t = s + proc[best_b]

        return order

    def _family_blocked_order(self):
        n = self.n
        fam, due, rel, w = self.fam, self.due, self.rel, self.w
        fam_ids = sorted(set(fam))
        fam_key = {}
        for f in fam_ids:
            members = [b for b in range(n) if fam[b] == f]
            agg = sum(w[b] / (due[b] + 1.0) for b in members)
            fam_key[f] = -agg
        fam_order = sorted(fam_ids, key=lambda f: fam_key[f])

        order = []
        for f in fam_order:
            members = [b for b in range(n) if fam[b] == f]
            members.sort(key=lambda b: (due[b], rel[b]))
            order.extend(members)
        return order

    def _move_delta_insert(self, order, i, L, j):
        n = len(order)
        if i < 0 or i + L > n:
            return None
        j = max(0, min(j, n))
        if j == i or j == i + L:
            return None

        block = order[i:i + L]
        rest = order[:i] + order[i + L:]
        jj = j - L if j > i else j
        jj = max(0, min(jj, len(rest)))
        return rest[:jj] + block + rest[jj:]

    def _move_delta_swap(self, order, a, b):
        if a == b:
            return None
        cand = order[:]
        cand[a], cand[b] = cand[b], cand[a]
        return cand

    def _move_delta_reverse(self, order, a, b):
        if a >= b:
            return None
        cand = order[:]
        cand[a:b + 1] = reversed(cand[a:b + 1])
        return cand

    def _lahc(self, order, cost, budget_hist=64):
        n = self.n
        cur, cur_cost = list(order), cost
        history = [cur_cost] * budget_hist
        v = 0
        best_o, best_c = list(cur), cur_cost

        it = 0
        deadline = self._t0 + TIME_LIMIT
        while True:
            it += 1
            if (it & 63) == 0 and time.perf_counter() >= deadline:
                break

            roll = self.rng.randint(10)
            if roll < 5:
                L = 1 + self.rng.randint(3)
                i = self.rng.randint(n - L + 1) if n - L + 1 > 0 else 0
                j = self.rng.randint(n + 1)
                cand = self._move_delta_insert(cur, i, L, j)
            elif roll < 8:
                cand = self._move_delta_swap(cur, self.rng.randint(n), self.rng.randint(n))
            else:
                a, b = self.rng.randint(n), self.rng.randint(n)
                if a > b:
                    a, b = b, a
                cand = self._move_delta_reverse(cur, a, b)

            if cand is None:
                continue

            c = self._cost_of(cand)

            if c <= history[v] or c < cur_cost:
                cur, cur_cost = cand, c
                if c < best_c:
                    best_c, best_o = c, list(cand)
                    self._submit(best_o)

            history[v] = cur_cost
            v = (v + 1) % budget_hist

        return best_o, best_c

    def solve(self, instance, submit_candidate):
        self._t0 = time.perf_counter()
        self._submit_func = submit_candidate
        n = self.n

        if n <= 1:
            self._submit(list(range(n)))
            return {"order": list(range(n))}

        seeds = [
            self._family_blocked_order(),
            self._atcs_order(kappa=0.5),
            self._atcs_order(kappa=1.0),
            self._atcs_order(kappa=2.0),
            sorted(range(n), key=lambda b: self.due[b]),
            sorted(range(n), key=lambda b: self.rel[b])
        ]

        best_o, best_c = None, math.inf
        for s in seeds:
            c = self._cost_of(s)
            if c < best_c:
                best_c, best_o = c, list(s)

        self.best_order, self.best_cost = list(best_o), best_c
        self._submit(self.best_order)

        cur_o, cur_c = list(best_o), best_c
        deadline = self._t0 + TIME_LIMIT

        while time.perf_counter() < deadline:
            run_o, run_c = self._lahc(cur_o, cur_c, budget_hist=64)
            if run_c < self.best_cost:
                self.best_cost, self.best_order = run_c, list(run_o)
                self._submit(self.best_order)
            cur_o, cur_c = run_o, run_c

        self._submit(self.best_order)
        return {"order": self.best_order}
