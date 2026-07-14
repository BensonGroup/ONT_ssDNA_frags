# What duplicate reads actually measure

*The sampling logic behind UMI-tagged selection data. This document contains no measurements — every
claim in §2 is a theorem, and holds regardless of what any particular dataset looks like. Sections 3–8
give the method that follows from it.*

---

## 0. Notation — one symbol per concept

| symbol | meaning |
| --- | --- |
| `R` | the number of **lineages** in the pool (its *richness*) — unknown |
| `pᵢ` | **true** fraction of the pool made of lineage *i*, so `Σᵢ pᵢ = 1` — unknown |
| `N` | number of reads sequenced — **fixed by you**, not random |
| `nᵢ` | number of reads **observed** for lineage *i* — data, `Σᵢ nᵢ = N` |
| **`S`** | the **collision probability**, `S = Σᵢ pᵢ²` — the chance two random reads come from the same lineage. **The one quantity this method estimates.** |
| **`Ŝ`** | our **estimate of `S`** computed from the counts (the hat means "from data") |
| **`C_eff`** | `1 / S` — the *effective diversity*. **A lower bound on `R`, never a headcount** (§2.2). |

A **lineage** = one original molecule *plus every PCR copy it ever spawned*. They all carry the same UMI,
so they are one lineage, not many molecules.

**On `N²` vs `N(N−1)`:** exact results below always use `N(N−1)` (or `C(N,2)`). Where a formula is written
with `N²` it is an **approximation** for readability, valid because `N` is large. Every such place is
flagged.

---

## 1. Two different meanings of "duplicate"

| | what it means | does PCR make these? |
| --- | --- | --- |
| **duplicate molecule** | two physical DNA strands that are copies of the same original | **Yes — by the billion. That is PCR's job.** |
| **duplicate read** | two *sequencing reads* from the same original lineage (you can tell: they share a UMI) | **Only if the sequencer happens to pick up two copies of the same lineage.** |

PCR is **necessary** for duplicate reads: without amplification each molecule exists in one copy, so you
could never draw the same one twice. **PCR creates the copies that make repeated sampling possible.**

But the rate is not set by the cycle count:

> **If PCR amplifies every lineage *proportionally*, it multiplies every `pᵢ` by the same factor. Since the
> `pᵢ` are fractions, they are unchanged. So `S = Σpᵢ²` is unchanged, and so is the expected duplication
> rate.**

A lineage that was one-in-a-million before PCR is still one-in-a-million after — because every *other*
lineage was copied too. Amplification scales the pool up; **proportional** amplification does not
*concentrate* it.

Duplicate reads therefore arise from **PCR-generated copies + sampling**, at a rate set by the pool's
composition (`S`) and your depth (`N`) — **not** by how many cycles you ran.

**The caveat that matters:** amplification is not perfectly proportional. Unequal efficiency — especially
**early-cycle jackpotting** — genuinely changes the `pᵢ`, and can manufacture high-count UMIs. That risk is
**worst when few templates enter the reaction**, i.e. exactly in a bottlenecked post-selection sample, and
**least** in a high-complexity one. This asymmetry is why a control must be matched on template count (§7).

---

## 2. The mathematics

### 2.1 What we are measuring: `S`, the collision probability

Pick two reads at random. What is the chance they came from the **same** lineage?

They can only match on *one* lineage — but you don't know which, so list every way it could happen. These
are **mutually exclusive** (a pair cannot be "both from lineage 1" *and* "both from lineage 2"), so their
probabilities simply **add**:

```
    S  =  P(both from lineage 1)  +  P(both from lineage 2)  +  …

       =        p₁ · p₁           +        p₂ · p₂           +  …

       =  Σᵢ pᵢ²
```

> `Σᵢ` does **not** mean the pair is spread across all lineages. It means: *here are all the lineages the
> pair could have matched on — add up the chance of each.*

**The dice version — the identical formula.** Roll two fair dice; what's the chance they match?

```
    P(match) = P(both 1) + P(both 2) + … + P(both 6) = 6 × (1/6)² = 1/6
```

Two dice land on **one** number when they match, yet the probability is a **sum over all six numbers they
could have matched on**. Same logic.

### 2.2 `C_eff = 1/S`, and the theorem that says what it is

`S` is a tiny, awkward number, so we usually quote its reciprocal, `C_eff = 1/S`.

**THEOREM.** For any pool whatsoever, `C_eff ≤ R`, with equality **if and only if** every lineage is equally
abundant.

*Proof.* By Cauchy–Schwarz applied to the vectors `(p₁,…,p_R)` and `(1,…,1)`:

```
    1  =  ( Σᵢ pᵢ )²   ≤   R · Σᵢ pᵢ²   =   R · S        ⟹        C_eff  =  1/S  ≤  R
```

Cauchy–Schwarz holds with equality exactly when the two vectors are proportional — i.e. when all `pᵢ` are
equal. ∎

**This is the single most important fact about `C_eff`, and it is a theorem, not an observation:**

> **`C_eff` is a LOWER BOUND on the number of lineages, always. It equals that number only for a perfectly
> uniform pool. The moment the pool is skewed, `C_eff` falls strictly below `R` — and it can fall
> arbitrarily far below.** So a fall in `C_eff` may mean lineages were lost, or that the survivors became
> uneven, or both. **`C_eff` alone cannot tell you which.**

*The two extremes.* Take a uniform pool of `C` lineages: `pᵢ = 1/C`, so `S = C · (1/C)² = 1/C` and
`C_eff = C` exactly — the bound is tight. Now take a **loaded die**, six faces with
`p = (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)`:

```
    S = 0.5² + 5 × 0.1²  =  0.25 + 0.05  =  0.30        →        C_eff = 1/0.30 = 3.3
```

The loaded die **still has six faces** (`R = 6`), but it *collides* like a die with 3.3. The gap between
`C_eff = 3.3` and `R = 6` **is** the skew — exactly as the theorem says it must be.

### 2.3 What you observe: counting collisions (exact — no probability)

Lineage *i* contributed `nᵢ` of your `N` reads. How many **pairs** of reads came from the same lineage?
Partition the pairs by lineage: a colliding pair needs **both** reads from one lineage, and the reads of
lineage *i* form a set of size `nᵢ`, from which you can choose 2 in `C(nᵢ,2)` ways:

```
    colliding pairs  =  Σᵢ C(nᵢ, 2)  =  Σᵢ nᵢ(nᵢ − 1) / 2
```

This is **pure counting** — an exact identity about your dataset. Nothing is assumed.

*Micro-example.* Reads `A A A B B C` → `n = [3, 2, 1]`, `N = 6`:

```
     from A:  (A₁A₂)(A₁A₃)(A₂A₃)  = 3  = C(3,2)
     from B:  (B₁B₂)              = 1  = C(2,2)
     from C:  —                   = 0  = C(1,2)
     ────────────────────────────────────────────
     colliding pairs = 4     of  C(6,2) = 15  total pairs
```

### The weighting is quadratic — and that is the correct answer

A lineage seen `nᵢ` times contributes `C(nᵢ,2)` pairs, **not one**. This is not a quirk; it is forced by the
question. The method asks *"pick two reads at random — how often do they collide?"*, so a lineage's
contribution **must** scale with how many pairs it can form. A lineage seen `n` times is not `n−1` duplicate
*events*: every one of its `n` reads can pair with every other, giving `n(n−1)/2` colliding pairs.

**Consequence, and it is a trap worth naming:**

> **`N − distinct` is NOT the number of colliding pairs.** It counts "extra reads" — a **linear** weight
> (`n−1`) — where the theory needs a **quadratic** one (`n(n−1)/2`). The two agree only when no lineage
> exceeds 2 copies (there `n−1 = 1 = C(n,2)`). As soon as any lineage is abundant they diverge without
> limit, and substituting one for the other silently corrupts every downstream number.

### 2.4 What you should expect

**Every pair has the same collision probability `S`.** Look at `S = Σᵢ pᵢ²` — nothing in it mentions *which*
two reads you picked. Every read is the same kind of thing: an independent draw from the same pool. Read #1
and read #7 are statistically identical.

So the expected number of collisions is *(how many pairs) × (chance each collides)*:

```
    E[ collisions ]  =  C(N, 2)  ×  S
```

Exactly like: *100 coins, each heads with probability 0.5 → expect `100 × 0.5 = 50` heads.*

> **This holds even though the pairs are NOT independent.** They are wildly dependent: if reads a,b match
> and reads b,c match, then a,c *must* match. But the coin rule doesn't care — even if the 100 coins were
> glued together you would still expect 50 heads. **Dependence changes the spread, never the average.** That
> is the one property this entire method leans on. **Remember it — §2.7 is where it bites back.**

### 2.5 The estimator: match observed to expected

One equation, one unknown — set what you counted equal to what theory predicts, and solve for `Ŝ`:

```
    Σᵢ nᵢ(nᵢ−1)/2   =   [N(N−1)/2] · Ŝ            ⟹            Ŝ  =  Σᵢ nᵢ(nᵢ−1)  /  [N(N−1)]
      ↑ what you counted     ↑ what theory predicts
```

**The ½'s cancel**, which is why the final formula carries no 2 — and why the code equivalently counts
*ordered* pairs (`N(N−1)` and `Σnᵢ(nᵢ−1)`, each exactly twice the unordered count). Same ratio either way.
Check on the micro-example: `Σnᵢ(nᵢ−1) = 6+2+0 = 8` over `N(N−1) = 30` gives `Ŝ = 8/30 = 4/15 = 0.267` ✓ —
identical to the unordered `4/15`.

**`Ŝ` is unbiased, at any depth.** There are exactly `N(N−1)` ordered pairs of distinct reads, each colliding
with probability `S`, so by linearity of expectation `E[Σᵢ nᵢ(nᵢ−1)] = N(N−1)·S`. Since **`N` is fixed by
you, not random**, `N(N−1)` is a constant — and dividing by a constant cannot introduce bias:

```
    E[Ŝ]  =  N(N−1)·S / [N(N−1)]  =  S             — exactly, for every N ≥ 2
```

**One honest caveat:** "set observed = expected" does not mean the two *are* equal in your data. They agree
only **on average** over hypothetical repeats; any single sample scatters around `C(N,2)·S`. **Unbiased means
"right on average", not "right every time".**

### 2.6 Why `nᵢ(nᵢ−1)` and never `nᵢ²`

The tempting shortcut is to plug the observed frequencies straight in: `p̂ᵢ = nᵢ/N`, giving `Σᵢ (nᵢ/N)²`.
**This is biased**, and the reason is the `−1`.

`nᵢ²` counts every ordered pair `(a,b)` of reads from lineage *i* — **including `a = b`, a read paired with
itself.** A read is trivially identical to itself, but that is not a duplicate; it is one read. The number of
pairs of *genuinely distinct* reads is `nᵢ(nᵢ−1)`. The `−1` removes exactly the `nᵢ` self-pairs.

Make it precise. Since `nᵢ² = nᵢ(nᵢ−1) + nᵢ` and `Σᵢ nᵢ = N`:

```
    Σᵢ nᵢ²        =   Σᵢ nᵢ(nᵢ−1)  +  N

    E[ Σᵢ nᵢ² ]   =   N(N−1)·S     +  N

                          N(N−1)·S + N            N−1         1               1 − S
    E[ Σᵢ (nᵢ/N)² ]  =  ───────────────   =   S · ───  +  ───   =    S   +   ─────
                               N²                  N          N                 N
```

> **The naive estimator carries a bias of exactly `(1 − S)/N`.** Because `S ≪ 1` in any high-complexity pool
> this is numerically `≈ 1/N` — **but the exact term is `(1−S)/N`, not `1/N`.** Note that it vanishes as
> `N → ∞`: the naive estimator is *consistent*, merely biased. That is precisely what makes it dangerous —
> it behaves well in a well-sampled pool and fails silently in an undersampled one.

**Why the bias is fatal here, and only here.** The bias is `≈ 1/N`; the signal is `S ≈ 1/C_eff`. So

```
    bias / signal   ≈   C_eff / N
```

In a **well-sampled** pool (`N ≫ C_eff`) the bias is a rounding error and nobody notices. In a **badly
undersampled** pool (`N ≪ C_eff`) — the regime of a high-complexity selection library — the bias is **larger
than the quantity you are trying to measure**. In the limit `S → 0` the naive estimator returns `≈ 1/N`, so
`1/Ŝ → N`: **it hands your read count back to you wearing a disguise.** It measures nothing but how deeply
you sequenced.

### 2.7 Depth-robustness — and exactly where the guarantee stops

**The objection:** the collision count *also* grows with depth (sequence twice as deep, get roughly 4× as
many colliding pairs). How can anything built on it be depth-robust?

**The answer: the depth cancels.** `E[collisions] = C(N,2)·S`, so

```
        C(N,2)                       C(N,2)
     ─────────────   estimates    ─────────────   =   1/S   =   C_eff
      collisions                   C(N,2) · S
```

Depth appears in numerator *and* denominator and divides out. Rigorously: `Ŝ` is unbiased for `S` at **any**
`N` (§2.5), and `S` is a property of the *pool*, not of the *run*. Depth affects how *precisely* you can
estimate `S`, never *what it converges to*.

Contrast "how many distinct lineages did I see": its expectation genuinely depends on `N`, and there is no
depth-free quantity for it to estimate. **That is the difference.**

### ⚠️ `S` is well-behaved. `C_eff` is not.

|  | `S` (collision probability) | `C_eff = 1/S` |
| --- | --- | --- |
| what it is | the **primitive** quantity — a probability | a **reciprocal**, for readability |
| estimator | `Σnᵢ(nᵢ−1) / N(N−1)` | `N(N−1) / Σnᵢ(nᵢ−1)` |
| **unbiased at any depth?** | **Yes — exactly** | **No** |
| behaviour | well-behaved everywhere | *consistent*, but **biased upward** |

`1/x` is **convex** on `(0,∞)`, so by **Jensen's inequality** `E[1/Ŝ] ≥ 1/E[Ŝ] = 1/S`. **The reciprocal of an
unbiased estimate is not an unbiased estimate of the reciprocal.** `C_eff` overshoots, and the overshoot is
worst exactly when there are few collisions to work with.

> **Practical rule: `S` is what you estimate; `C_eff` is what you print. Never read `C_eff` without the
> collision count beside it.** An estimate resting on a handful of collisions must not be quoted bare.

### The zero-collision case — and why the *rule of three* does **not** apply

If you observe **no** collisions at all, then `Ŝ = 0` and `C_eff = 1/0 = ∞`. That is not an edge case needing
a hack; it is the reciprocal doing what reciprocals do. The only honest answer is a **lower bound** on
`C_eff` — equivalently an upper bound on `S`.

**The tempting shortcut is wrong.** The *rule of three* says: zero events in `n` **independent** Bernoulli
trials ⟹ `p < 3/n` at 95%. It is natural to set `n = C(N,2)` pairs and be done. **You cannot.** The `C(N,2)`
pairs are **not independent** — that is the very dependence §2.4 relies on being harmless *for the mean*.
Linearity of expectation survives dependence; **a binomial tail bound does not.** Invoking the rule of three
here would use the dependence when convenient and assume it away when inconvenient.

**The correct route gives the same constant.** Collisions are **rare** and only **weakly** dependent, so
their count is well approximated by a **Poisson** distribution (Chen–Stein) with mean

```
    λ  =  C(N,2) · S
```

Observing zero collisions has probability `e^(−λ)`. That observation stays consistent with the data at the
5% level as long as `e^(−λ) ≥ 0.05`, i.e. `λ ≤ 3` (since `−ln 0.05 = 2.996`). Hence

```
    C(N,2) · S  ≤  3       ⟹       S  ≤  3 / C(N,2)       ⟹       C_eff  ≥  C(N,2)/3  =  N(N−1)/6
```

**Same `3`, same bound — but resting on a Poisson approximation for a sum of rare, weakly dependent
indicators, not on an independence assumption that is plainly false.** This is the same Poisson model §7 uses
to ask whether an all-singleton control is plausible; the document uses one model throughout.

Report such a sample as **`C_eff > x`**, never as `C_eff = x`.

### 2.8 Rarefaction — putting samples on a common depth

Richness counts grow with depth, so to compare them across samples of unequal depth you must ask: *how many
distinct lineages would this sample have shown at `m` reads?*

**Exact (sampling without replacement).** Lineage *i* is missed iff none of its `nᵢ` reads is drawn:

```
    E[distinct]  =  Σᵢ [ 1  −  C(N − nᵢ, m) / C(N, m) ]
```

**Approximation (sampling with replacement / binomial).** Each read is retained independently with
probability `m/N`, so lineage *i* is missed with probability `(1 − m/N)^nᵢ`:

```
    E[distinct]  ≈  Σᵢ [ 1  −  (1 − m/N)^nᵢ ]
```

**The implementation uses the second form.** The two agree closely when `N` is large and `nᵢ ≪ N`, which is
the regime here — but they are **not the same formula**, and the one in use is the approximation.

> **Rule of thumb: if a number gets bigger when you sequence more, it is not a property of your library.**
> Rarefy any richness count to a common depth before comparing across samples. `S` needs no such correction
> (§2.7), which is exactly why it is the right instrument.

**But rarefaction has a blind spot, and it is severe.** If the pool is far larger than the depth (`R ≫ m`),
almost every read lands on a lineage not yet seen, and `E[distinct] → m` **for any sufficiently large pool**.
Rarefied richness then **saturates at the read count** and can no longer distinguish a pool of `10⁶` from one
of `10⁸`. In that regime *"richness barely changed"* is **not** evidence that the pool kept its members — it
is evidence that **this instrument is blind**. Only `S` can still see.

---

## 3. What `C_eff` measures: concentration, not headcount

Everything in this section follows from the theorem in §2.2. No data is required.

`C_eff ≤ R`, always. So a fall in `C_eff` between rounds tells you the pool became **more concentrated**, and
*nothing more*. Concentration has **two** possible causes:

1. **Lineages were lost** — `R` itself fell; or
2. **Survivors became uneven** — `R` held, but a few lineages grew abundant, widening the gap between `C_eff`
   and `R`.

**`C_eff` cannot distinguish them.** Both raise `S`; both are real forms of selection. Reporting a fall in
`C_eff` as *"the library collapsed to `C_eff` molecules"* asserts cause (1) **and** asserts the bound is
tight — two claims the statistic does not make.

### How to tell the two apart

| symptom | what it points to |
| --- | --- |
| `S` rises **and** the largest `nᵢ` rise sharply | **skew** — a few lineages amplified |
| `S` rises while all counts stay near 1 | **attrition** — the pool shrank, fairly evenly |
| a large share of `Σᵢ C(nᵢ,2)` comes from a handful of lineages | **skew**, and `C_eff` is hostage to them |
| rarefied richness falls at matched depth | **attrition** — *but only if the depth can see it (§2.8)* |

**Always report, alongside `C_eff`:** the collision count `Σᵢ C(nᵢ,2)`, the **maximum** `nᵢ`, and the share of
the collision sum carried by the top few lineages. Those three say *which* mechanism you are looking at.
`C_eff` on its own does not.

> Lost lineages do **not** create high-count UMIs. **PCR skew and clonal expansion do.** That is the
> discriminator.

### The mechanism and the cause are different axes — and the empty round only reads one

The two mechanisms above answer *what happened to the pool*. They say nothing about *why*. These are
**independent** questions, and the most common mistake in reading this statistic is to assume that a
control round settles both. It does not.

The mechanism axis is exhaustive — `S = Σᵢ pᵢ²` can only rise because `R` fell or because the `pᵢ` went
uneven; there is no third way. The cause axis has **three** entries, because a number can move without the
pool moving at all:

| | **attrition** — pool shrank (`R` fell) | **skew** — survivors uneven (`pᵢ` spread) |
| --- | --- | --- |
| **protocol** | aliquot transfer = random subsampling | PCR jackpotting |
| **biology** | lineages died / were washed out | clonal expansion of a real binder |
| **artifact** | over-aggressive UMI merging — chance `ed ≤ 1` neighbours collapsed into one (§7 Use 5) | a degenerate UMI absorbing many molecules — a low-complexity/garbage tag, or anchor-only reads sharing one |

Every cell raises `S`. `C_eff` sees only the sum. To place an observation in a cell you must read the two
axes with **different instruments**:

- **The column — attrition vs skew — is read from the count shape *within one sample*:** the maximum `nᵢ`
  and the share of `Σᵢ C(nᵢ,2)` carried by the top few lineages, exactly as the table above prescribes.
  **This needs no control of any kind.** One sample is enough.
- **The row — protocol vs biology — is read from the empty round (§7):** whatever concentration the empty
  round shows is the protocol's, and the excess in the real round is selection-associated.
- **The artifact row is read from QC and audit, and must be settled *first*** — before either of the other
  two readings is meaningful. Check that the observed collision count is consistent with `λ = C(N,2)·S`
  (§7 Use 2), subtract chance `ed ≤ 1` neighbours before calling merges "errors" (§7 Use 5), and confirm
  that the top-`nᵢ` lineages are QC-clean and actually carry payload. **An artifact in the top lineage
  corrupts `S` more than any biology can**, because the weighting is quadratic (§2.3).

> **An empty round reads the *cause* axis, never the *mechanism* axis.** "Is this attrition or skew?" is
> answered by the shape of your own counts. "Is this selection or the protocol?" is answered by the control.
> Neither question answers the other.

**Two caveats stop this being a clean decomposition:**

1. **The empty round systematically under-reads protocol skew.** Jackpotting worsens as the number of
   templates entering the PCR falls (§6), so a high-complexity empty round and a severely bottlenecked
   selected round sit in **different jackpotting regimes**. The empty round therefore yields a **lower
   bound** on the protocol skew present in the selected round, not a matched estimate. Only a
   **template-count-matched** control closes that gap (§7).
2. **An aliquot transfer produces attrition, not skew** — it is random subsampling, and random subsampling
   does not preferentially amplify anyone. So essentially *all* of an empty round's **skew** is PCR
   jackpotting. That is precisely what makes the empty round a usable jackpot baseline even though, by the
   rule above, it cannot settle the attrition question.

Finally, note that a jackpot check exists that needs **no empty round at all**: the two-axis fragment-vector
readout of §8.4 — *reads per vector* against *distinct lineages per vector*. A vector whose reads arrive
from many independent lineages cannot be a single amplified clone, whatever the protocol did.

---

## 4. Why almost every UMI is a singleton — and why samples barely overlap

### The photocopy picture

Imagine a pool of `R` lottery tickets, each with a unique number — your UMI-tagged lineages.

**Photocopy every ticket a thousand times.** That is PCR. You now have `1000·R` sheets of paper — but still
only `R` distinct numbers. Photocopying made more paper; it did not invent new numbers.

Now **draw `N` sheets at random.** That is sequencing. If `N ≪ R` you will almost never draw the same *number*
twice: for a near-uniform pool the expected collisions are `≈ N²/(2R)` *(the `N²` here is the approximation
for `N(N−1)`)*. Almost every sheet shows a number you have never seen.

> **All-singleton data is the signature of undersampling, not of a broken experiment.**

### Two traps that follow

**Trap 1 — richness counts are depth-dependent.** The number of distinct things you observe depends on how
hard you looked. Comparing raw distinct counts across samples of unequal depth is invalid. Rarefy first
(§2.8) — and remember that rarefaction goes blind once the pool exceeds the depth.

**Trap 2 — between-sample overlap cannot measure pool size.** Two independent samples of `N` reads from a
shared near-uniform pool of `R` lineages share only

```
    E[shared lineages]  ≈  R · (N/R)²  =  N² / R
```

so overlap **collapses toward zero for any sufficiently large pool** — *even if not a single lineage was
lost*. Low overlap is a statement about `N²/R`, not about attrition.

> **Measure pool concentration with *within-sample* collisions, never with *between-sample* overlap.**

| what you see inside one sample | what it means |
| --- | --- |
| **many collisions** (you keep re-drawing the same lineages) | the pool is **concentrated** |
| **near-all singletons** (every read a fresh lineage) | the pool is still **large relative to your depth** |

---

## 5. So what *does* create duplicate reads? Concentrating the pool.

Collisions rise when `N` grows or when `S` grows. At roughly constant depth the driver is **`S` rising** —
i.e. `C_eff` falling — which, per §3, means the pool concentrated **either** by losing lineages **or** by
becoming uneven.

Selection can do both. **So can the protocol:** every aliquot transfer is itself a bottleneck that
concentrates the pool with no biology involved at all. In this statistic, drift from transfers looks exactly
like selection. That is why the control rounds exist (§7).

---

## 6. Does PCR jackpotting create duplicates?

PCR *is* biased. But bias shows up in your reads only if it shifts a lineage's **relative** frequency enough
to be visible at your depth. A lineage at frequency `p` yields `N·p` reads on average, so it needs `N·p ≳ 1`
— i.e. `p ≳ 1/N` — to appear at all. In a pool where the typical lineage sits at `p ≈ 1/R` with `R ≫ N`, a
jackpot must lift a lineage by a factor of order `R/N` before it produces even a handful of reads.

> **Modest amplification bias is invisible at these depths; only a large jackpot registers.**

**But the crucial asymmetry:** jackpotting is worst when **few templates enter the PCR**. A high-complexity
sample is the *least* susceptible; a severely bottlenecked post-selection sample is the *most*. So clean
behaviour in a high-complexity library **does not** license the assumption that a bottlenecked sample's
high-count lineages are bias-free. That question needs a **template-count-matched** control (§7).

---

## 7. The empty / PCR-only control rounds

An empty round = the same aliquot transfers and PCR cycles, **no biological material**. It is the null for
everything that depends on the pool being concentrated.

### The question they exist to answer

> **How much of the observed concentration is selection, and how much is just the protocol?**

**That is the *cause* axis, and it is the only axis these rounds read.** Whether the concentration arose by
attrition or by skew is settled by the shape of the counts *within* a sample — the max `nᵢ` and the
top-lineage share of the collision sum — and needs no control at all. See §3, *"The mechanism and the cause
are different axes"*. Do not ask an empty round a question it cannot answer.

### Use 1 — the protocol-only baseline

Compute `S`, `C_eff`, the collision count, the max `nᵢ`, and the top-lineage share of the collision sum for
every empty round, then compare:

```
    total concentration     =  C_eff(initial)  / C_eff(selected round k)
    protocol-associated     =  C_eff(initial)  / C_eff(empty round k)
    selection-associated    =  C_eff(empty k)  / C_eff(selected round k)
```

Match the control to the real round with the **same number of transfers**, not to the initial library.

> **This is a descriptive comparison, not a causal decomposition.** `C_eff` is nonlinear, and protocol effects
> **interact** with selection — a bottlenecked sample is amplified from far fewer templates than a
> high-complexity control (§6). The ratio does not mathematically isolate a "selection factor". For
> attribution, compare the underlying distributions: `S`, the count distributions, the **top-lineage shares**,
> and vector enrichment against matched controls.

### ⚠️ Use 2 — "all singletons" is not automatically a clean result

Before interpreting an all-singleton control, **compute what you should have seen**. If the control were drawn
from a pool with collision probability `S`, then at depth `N`:

```
    λ  =  E[collisions]  =  C(N,2) · S              P(observe ZERO)  =  e^(−λ)
```

Take `S` from the initial library and `N` from the control's own depth. **If `λ` is large, a literally
zero-collision control is essentially impossible** — `e^(−λ)` is astronomically small — and its appearance
means something is *wrong*, not that something is clean.

| what the empty round shows | what to conclude |
| --- | --- |
| a collision count consistent with `λ = C(N,2)·S(initial)` | ✅ consistent — the pool is intact, the control is usable |
| **far fewer collisions than `λ` predicts** (especially literally zero) | 🚩 **audit before interpreting.** UMI filtering too aggressive? Sequencing error inflating apparent diversity? Sample definition? A bug in the counting? An **anomaly**, not a clean control. |
| **many more collisions**, with high-count UMIs | the pool is concentrated — by attrition, PCR skew, or both (§3) |

### What the control can honestly establish — and what it cannot

1. **Near-zero collisions gives only a LOWER BOUND** on the remaining `C_eff` (§2.7). It says *"the pool is
   still large relative to my depth"*. It does **not** prove that few lineages were lost: by §2.2 a pool can
   shed most of its lineages and still show no collisions, provided what remains is large relative to `N`.
2. **It does NOT control for PCR jackpotting in the selected rounds.** A high-complexity empty control and a
   severely bottlenecked cell-recovered sample sit in **different jackpotting regimes** (§6). "No high-count
   UMI in the empty control" **supports but does not prove** that a high-count lineage in a selected round is
   biological.

**To pin a high-count lineage down you would need** a *template-count-matched* no-cell control (input diluted
to the recovered template number), technical PCR replicates from the same recovered material, independent
biological selections, or a second UMI added *after* selection but *before* the final PCR.

### Use 3 — an empirical null for enrichment

For every fragment vector *i*, depth-normalise its count in each sample:

```
        fᵢ  =  nᵢ / N          (×10⁶ if you want "reads per million")
```

> **Do not scale by the sample's observed richness.** A published `count × total_diversity / total_reads` uses
> `total_diversity` = the **library size**, a design *constant* identical in every sample; it only sets where
> "enrichment = 1" falls. Substituting the *observed per-sample* richness is wrong twice: it differs per
> sample, and it is **itself depth-dependent** (§2.8) — so it re-injects the very confound you are trying to
> remove. **A constant factor rescales the axis; a per-sample factor corrupts the comparison.**

Then compute two fold-changes:

```
    log2FC_selection(i)  =  log2 ( fᵢ(selected) / fᵢ(initial) )      ← what you want to test
    log2FC_null(i)       =  log2 ( fᵢ(empty)    / fᵢ(initial) )      ← what "no selection" looks like
```

`log2FC_null` is the distribution of fold-changes produced by **nothing but** transfers, PCR, and sequencing.
It is a noise model **measured, not assumed**.

**The test:** call a vector enriched only if `log2FC_selection` exceeds (say) the 99th percentile of
`log2FC_null` **among vectors of comparable starting count**. That last clause is essential — low counts
fluctuate far more than high ones, so the threshold must be a *function of count*. The empty round gives you
that function for free.

**With more than one empty round you get something better:** compare *empty vs empty*. Two samples differing
**only** by PCR and sequencing give the pure technical-replicate variance — the cleanest noise floor, with no
initial-library count noise leaking in.

### Use 4 — what `min_reads` is actually for

**The count-stratified null above already handles low counts.** A 3-read vector is compared against the null
*of 3-read vectors*, which is very wide, so it must clear a far higher bar. **The test penalises low counts
automatically; it needs no cutoff to do so.**

So `min_reads` is **not** an independent noise criterion. What it legitimately remains is a **power /
multiple-testing pre-filter**: a vector with 2 reads cannot reach significance no matter what, and testing
every vector instead of only those that could possibly win inflates your multiple-testing correction at the
expense of the vectors you care about.

> **Set `min_reads` to "the smallest count at which a vector could *possibly* clear the null" — a POWER
> threshold — derived from the same null. Use it to decide *what to test*, never *what is significant*.**

**How to derive it.** In an empty round nothing was selected, so every `log2FC_null` is pure noise, ideally 0.
It won't be — it scatters. **"Spread" is the width of that scatter**, and it *shrinks as counts grow*:

1. **Bin** vectors by their initial-library count: `1, 2, 3–4, 5–9, 10–29, 30–99, 100+`.
2. **Within each bin**, take the `log2FC_null` values and measure their width — the **95th percentile of
   `|log2FC_null|`** reads directly as *"95% of vectors in this bin wobble by less than this, with no biology
   at all."*
3. **`min_reads` = the smallest bin whose spread falls below your tolerance** (a natural choice: ±1 on the
   log2 scale, i.e. a 2-fold wobble).

**What to expect.** If the only noise were Poisson counting, then for a ratio of two independent counts of
size `n` the delta method gives

```
    SD( log2 FC )  ≈  √(2/n) / ln 2
```

so at `n = 10`, SD ≈ **0.65**, a 95% band of ≈ **±1.3** — a **2.5-fold** wobble with no biology at all.
**That is what a `min_reads` of 10 implicitly buys you.** The empty round tells you whether reality is *worse*
(PCR adds overdispersion — usually) or better (in which case you can lower the threshold and **recover hits
currently discarded**).

### Use 5 — the UMI error rate, with the chance rate subtracted

In an empty round, ask how many UMIs sit within edit distance 1 of another. **Chance neighbours are not
negligible**, and you must subtract them before calling the remainder "errors".

For `D` random UMIs of length `L`, the expected number of **chance** pairs within edit distance 1 is

```
    E[chance pairs]  =  C(D,2) · (size of the distance-1 neighbourhood) / 4^L

        Hamming-1:      neighbourhood = 3L         (one of L positions × one of 3 other bases)
        Levenshtein-1:  neighbourhood ≈ 7L + 3     (substitutions + insertions + deletions; approximate)
```

**starcode clusters by Levenshtein distance, so the larger neighbourhood is the relevant one.** Note the
scaling: `C(D,2)` grows **quadratically** in the number of distinct UMIs while `4^L` is **fixed** — so the
chance rate can become a material fraction of all merges as `D` grows, even at a length that felt safe.

> **Compute `E[chance pairs]` for your own `D` and `L`, subtract it from the observed merge count, and only
> then call the remainder sequencing error.** Skipping that step inflates the error rate. Do **not** assume
> "every edit-distance-1 pair is an error".

Barcode correction should also not lean on edit distance alone — read quality, abundance, the associated
structure sequence, and sample context all carry information a pure `ed ≤ 1` rule discards.

### What the empty rounds cannot tell you

- **They do not measure index hopping.** They are drawn from the *same* DNA pool, so they share lineages with
  every other sample **by ancestry**. Shared UMIs are therefore expected, and are indistinguishable from
  barcode crosstalk. Measuring hopping needs a **distinguishable spike-in** on the same flowcell.
- **They are not biological replicates.** They are a technical null. Two independent selections would be
  needed for biological reproducibility.

---

## 8. What this means for the analysis

1. **Do not deduplicate.** When the typical molecule is seen about once there is almost nothing to remove —
   and collapsing would **delete the high-count lineages**, which are the signal, not noise. PCR-duplicate
   removal is right when amplification is a nuisance. Here amplification *is* the readout.

2. **When reads-per-UMI ≈ 1, counting distinct UMIs ≈ counting reads.** In that regime a "molecule count" is
   not measuring your library; it is measuring your sequencer.

3. **Count at the fragment-vector level instead.** Many *different* lineages carry the *same* ligand
   combination, so counts accumulate there and the statistics become usable — and the initial library works
   as a baseline, which it cannot do at the lineage level when almost every lineage is a singleton.

4. **Use the UMI for the one thing it is genuinely good at: counting independent clones.** A vector whose
   reads arrive from **many distinct lineages** is reproducible selection. The same read count arriving from
   **one** lineage is a single amplified clone masquerading as a hit. That check — *reads per vector* versus
   *distinct lineages per vector* — is the two-axis readout the UMI was installed for.

5. **Report `S`, the collision count, the max `nᵢ`, and the top-lineage share — never `C_eff` alone** (§3).

---

## 9. Summary

> **PCR generates copies of each UMI-tagged lineage. If amplification is proportional it preserves lineage
> frequencies, while cellular selection, transfer bottlenecks, and amplification bias may alter them.
> Within-sample UMI collisions estimate the pool's *concentration* through `S = Σpᵢ²`. A high collision rate
> shows the pool has become concentrated; it does not, by itself, say whether lineages were lost or merely
> became uneven, nor prove that the concentration is biological.**

**The six things to carry away:**

1. **`S` is the well-behaved quantity; `C_eff = 1/S` is only its reciprocal.** `S` is unbiased at *any* depth;
   `C_eff` is consistent but **biased upward** (Jensen), worst when collisions are scarce.
2. **`C_eff ≤ R`, always, with equality only for a uniform pool** (Cauchy–Schwarz, §2.2). It is a **lower
   bound on the number of lineages, not a headcount.** A fall in `C_eff` means the pool concentrated — by
   attrition, by skew, or both — and **`C_eff` alone cannot say which.**
3. **Never use `N − distinct` as the collision count.** It weights a lineage linearly (`n−1`); the theory
   requires quadratically (`n(n−1)/2`). They agree only when nothing exceeds 2 copies.
4. **Never use `Σ(nᵢ/N)²`.** Its bias is exactly `(1−S)/N` — self-pairs — which *exceeds the signal itself*
   whenever `N ≪ C_eff`, and makes it hand your read count back to you.
5. **Zero collisions gives only a lower bound**, justified by a **Poisson** argument — *not* by the rule of
   three, whose independence assumption these pairs violate. And zero is only *expected* when `λ = C(N,2)·S`
   is small; if `λ` is large, zero is an **anomaly to audit**, not a clean control.
6. **Never compare a raw richness count across depths; never measure pool size with overlap.** Rarefy first
   (§2.8) — and note that rarefaction **saturates**, going blind once the pool exceeds the depth. Overlap
   (`≈ N²/R`) collapses toward zero for *any* large pool, lost lineages or not.

**None of this changes the pipeline:** still don't deduplicate, still count at the fragment-vector level,
still use the UMI for incidence.
