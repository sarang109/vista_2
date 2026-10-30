# Control-typed attested evidence games: definitions, results and proofs

This note states every result the code relies on, gives a complete proof of each, and points to the
numerical checks in `e0_verification.csv` (experiment E0) and to the experiments that measure them.
Theorems 1–3 are new to v2. Theorems 4 and 5 carry over from v1's selection-leakage analysis and are
restated here so that the document is self-contained.

## Names used here and in the paper

The repository numbers its results independently of the manuscript ("Strategy-Proof Fraud Decisions over
Attested Evidence for Real-Time and Agentic Payments"). Equation numbers refer to the manuscript.

| this document, README and code | manuscript |
|---|---|
| Definition 1 (presentation order), Definition 2 (SP game) | Section III and Eq. (2), Section IV |
| Theorem 1 (characterization) | **Theorem 1** (Reachability characterization), Section IV |
| Corollary 1 (control types) | the three engineering constraints stated after Theorem 1 (unnumbered) |
| Theorem 2 (evaluation gap) | Eq. (3) and "By Theorem 1, FA⋆ ≥ FA0", Section IV (unnumbered) |
| Proposition 1 (safe repair), Algorithm 2b, box repair | **Proposition 1** (Conservative monotone repair) and Eq. (4), Section IV |
| Theorem 3 (splitting) | cap argument and Eq. (7), Section V (unnumbered) |
| Lemma 1 (leakage decomposition) | Eq. (9), Section VI |
| Theorem 4 (no free selection) | **Proposition 2** (Zero-leakage disclosure) and the obstruction after it, Section VI |
| Theorem 5 (repeated payments) | coverage bound Eq. (10) and sticky selection, Section VI (unnumbered) |
| Algorithms 1, 2a, 2b, 3 | Section VI-B (training and repair implementation) and Section V (cumulative claim) |

## 1. Model

**Parties.** A sender `S` (a bank, or an AI agent paying under a user's mandate), a receiver `R` that runs a
decision rule `d` (any trained model plus thresholds), and third parties `T` (trust registry, acquirer,
mandate ledger) that sign facts `S` cannot change. Presentation is holder-driven, as in SD-JWT and verifiable
credentials: the sending bank (issuer, honest) signs claims for the payer's wallet or agent (holder), and the
holder chooses which disclosures travel with the payment. A fraudster controlling the account, or a
prompt-injected agent, is therefore a strategic presenter.

**Inputs.** Each payment presents `z = (w, f, t)`:

| type | meaning | who can change it | mechanism |
|---|---|---|---|
| `W` withholdable attested | signed threshold claims on ordered domains | the holder may show any weaker `w' ≤ w`, never a stronger one | VSAC credential with salted digests, PID binding |
| `F` freely chosen | fields the holder picks or shapes | the holder, within a reachability relation | none: this is the attack surface |
| `T` third-party fixed | facts signed or observed by someone other than the holder | nobody on the sender side | registry, acquirer, ledger, receiver observation |

**Definition 1 (presentation order).** `z → z'` when `z'` can be presented by a holder whose true input is
`z`: `w' ≤ w` in the product of the claim orders, `f'` reachable from `f` under the reflexive–transitive
closure of the allowed manipulations (split, wait, relabel, rewrite text), and `t' = t`. Each coordinate
relation is reflexive and transitive and the product of preorders is a preorder, so `→` is a preorder. Using
the closure is deliberate: a scripted or prompt-injected agent can chain manipulations.

A `W` coordinate may be a chain (`0 < 1 < … < L`, with `0` meaning "nothing shown") or a poset with a bottom
`0` below mutually incomparable values. The second form models a match flag whose values *match* and
*mismatch* can both be hidden but neither can be turned into the other (`vsac.py`, `attack.py`).

**Realized decision.** Decisions are ordered (reject < step-up < accept, or hold < accept) and the
adversarial holder prefers higher ones. Whatever rule `R` deploys, what it does against a best-responding
holder is

    d*(z) = max_{z → z'} d(z').

**Definition 2 (strategy-proofness game).** Fix `d`. The adversary wins if it outputs `z` in the support and
`z'` with `z → z'` and `d(z') > d(z)`. `d` is *strategy-proof* (SP) if no such pair exists. The game is
information-theoretic; that the holder cannot produce a `W` or `T` value that was never issued is the binding
game of v1, which reduces to EUF-CMA security of Ed25519 and collision resistance of SHA-256 (the PID hash and
the disclosure digests). E7 exercises it empirically.

## 2. Theorem 1 (characterization)

For every rule `d`:

1. `d* ≥ d`, and `d*` is monotone: `z → z'` implies `d*(z) ≥ d*(z')`.
2. `d` is SP ⇔ `d = d*` ⇔ `d` is monotone in `→`.
3. Against a best-responding holder every deployed rule acts as its own `d*`. The outcome functions a receiver
   can implement are exactly the `→`-monotone functions, and restricting training to monotone rules loses
   nothing that was achievable.

*Proof.* (1) `z → z` by reflexivity, so `d*(z) ≥ d(z)`. If `z → z'` and `z' → z''`, then `z → z''` by
transitivity, so the set maximised in `d*(z)` contains the set maximised in `d*(z')`, and `d*(z) ≥ d*(z')`.

(2) `d` is SP iff `d(z') ≤ d(z)` for every `z → z'`, which is the definition of monotone in `→`. This holds
iff `max_{z→z'} d(z') ≤ d(z)` for every `z`, i.e. `d* ≤ d`, and with (1) iff `d* = d`.

(3) A holder with truth `z` presents a maximiser of `d` over `{z' : z → z'}` and obtains `d*(z)`. By (1)
`d*` is monotone, so by (2) it is SP and `(d*)* = d*`. Hence every realized outcome function is monotone, and
every monotone `g` is realized by deploying `g` itself (`g* = g`). Deploying `d*` instead of `d` gives the same
realized outcome and makes the evaluated outcome equal to it. ∎

Transitivity of `→` is Green and Laffont's nested-range condition (1986), under which their revelation
principle for evidence games applies; Theorem 1 is that principle stated for receiver decision rules over typed
payment inputs. Checks `T1.*` in E0 verify every item on random preorders (2,000–3,000 instances, brute force
for item 3's maximality on spaces of up to six inputs).

### The safe repair

For any `d`, define `d_(z) = min_{y → z} d(y)`: decide as if the truth were the least favourable input that
could have produced what is shown.

**Proposition 1.** `d_` is SP, `d_ ≤ d`, every SP rule `g ≤ d` satisfies `g ≤ d_`, and `d_ = d` iff `d` is SP.

*Proof.* If `z → z'` and `y → z`, then `y → z'`, so the minimum for `z'` is over a superset and
`d_(z') ≤ d_(z)`: `d_` is monotone, hence SP by Theorem 1. Reflexivity gives `d_ ≤ d`. If `g` is SP and
`g ≤ d`, then for every `y → z`, `g(z) ≤ g(y) ≤ d(y)`; minimising over `y` gives `g(z) ≤ d_(z)`. If `d` is SP,
then `d(y) ≥ d(z)` for every `y → z`, so the minimum is attained at `y = z`; conversely `d_ = d` makes `d` SP. ∎

On a risk score `s` with "accept iff `s ≤ τ`", `d_` is the rule "accept iff `s̄ ≤ τ`" with
`s̄(z) = max_{y→z} s(y)`. On a product order the set `{y : y → z}` is the product of per-coordinate up-sets:
levels `≥` the shown level on a `W` chain; every value if nothing is shown on a `W` poset, else the shown
value; every value on a completely reachable `F` coordinate; amounts `≥` the shown amount on a split-closed
amount ("as if the amount were unbounded", which is Theorem 3 inside the repair); the shown value on `T`.

**Algorithm 2b (lattice DP).** On a product of chains, `s̄[w] = max(s[w], max_i s̄[w + e_i])` evaluated in
decreasing order equals `max_{w' ≥ w} s[w']`, because the up-set of `w` is `{w} ∪ ⋃_i up(w + e_i)`.
`repair.lattice_dp` computes it as a composition of per-axis suffix maxima. Check `A2b.dp` compares both with
brute force.

**Box repair for tree ensembles (new in this artifact).** For a tree ensemble `s = σ(Σ_k h_k)`, let `B(z)` be
the bounding box of the up-set `U(z) = {y : y → z}` and `s_box(z) = σ(Σ_k max_{x ∈ B(z)} h_k(x))`, where each
per-tree maximum is the largest leaf whose region meets the box.

*Claim.* `s_box ≥ s̄ ≥ s`, and the rule "accept iff `s_box ≤ τ`" is SP.
*Proof.* `U(z) ⊆ B(z)` and `max Σ ≤ Σ max` give `s_box ≥ s̄`. If `z → z'`, every coordinate interval of
`B(z')` contains that of `B(z)`: a weaker `W` level lowers the lower end of `[w, L]`; hiding a poset value
replaces a point by the full range; complete `F` coordinates are always the full range; a smaller split amount
lowers the lower end of `[a, max]`; `T` coordinates do not move. A larger box can only meet more leaves, so each
per-tree maximum and hence `s_box` is non-decreasing along `→`; by Theorem 1 the rule is SP. ∎

The box repair is not the largest SP rule below `d`: it is looser when trees interact on moved coordinates,
and E13 shows the excess grows for presentations that show little. Checks `BOX.*` verify that it equals the
model when nothing moves, never falls below the exact repair, and never falls when claims are withheld.

## 3. Corollary 1 (control types)

Let `→` be the product of a chain or poset down-relation on each `W` coordinate, a reachability preorder on
each `F` coordinate and equality on each `T` coordinate. Then `d` is SP iff

* (a) `d` is non-decreasing in every `W` coordinate (more verified evidence never lowers the decision);
* (b) `d` never rewards a reachable change of an `F` coordinate, and when every value of an `F` coordinate is
  reachable from every other (the hour of payment, free text), `d` does not depend on it at all;
* (c) `T` coordinates are unconstrained.

In particular the absence of a claim may be used against the holder only through a `T` fact, such as a
registry-signed "this institution can issue claim i".

*Proof.* If `z → z'` in the product order, change the coordinates of `z` into those of `z'` one at a time; each
intermediate input is reachable from the previous one by a single-coordinate move (the other coordinates use
reflexivity). So `d` is SP iff no single-coordinate move increases `d`. On a `W` coordinate the moves are
weakenings, which gives (a). On an `F` coordinate they are the reachable changes, which gives the first part of
(b); if `f → f'` and `f' → f`, SP forces `d(…f…) ≥ d(…f'…)` and the reverse, hence equality, which gives the
second part. On a `T` coordinate the only move is the identity, which gives (c). ∎

Check `C1.enumeration` enumerates all 2^18 binary rules on `W chain(3) × F complete(3) × T(2)` and confirms
that SP holds for exactly the rules described by (a)–(c).

**Delegated authority.** An agent's authority chain `user → platform → agent → sub-agent` signs a level per
link; the chain level is `min` over links, which is non-decreasing in every link, so the chain is a `W`
coordinate and Theorems 1–2 apply unchanged (attenuation by minimum, as in SPKI, UCAN and macaroons;
reused, not claimed). Stripping a link drops the chain to "nothing shown", which an SP rule never rewards.

**Agent-written text.** A justification the agent writes is a completely reachable `F` coordinate, so by
(b) an SP rule cannot depend on it. *Remark (measured in E4).* A rule that lets text only lower acceptance,
`g = d ∧ [text not flagged]`, still violates (b). If any unflagged text exists, `g* = d*` at `d`'s threshold;
at a fixed legitimate-hold budget, the holds spent on flagged legitimate text loosen `d`'s threshold, so
`g`'s realized fraud acceptance exceeds that of the text-free rule calibrated to the same budget, while its
evaluated acceptance is lower.

## 4. Theorem 2 (evaluation gap)

For any rule `d` and any distribution of fraudulent inputs `Z`,

    P(d*(Z) ≥ accept) ≥ P(d(Z) ≥ accept),

with equality iff `P(d(Z) < accept ≤ d*(Z)) = 0`, i.e. iff no input in the support that is not accepted can
reach an accepted presentation (the induced accept/not-accept rule is monotone on the support).

*Proof.* `d* ≥ d` pointwise (Theorem 1), so `{d(Z) ≥ accept} ⊆ {d*(Z) ≥ accept}` and the difference of the two
probabilities is `P(d(Z) < accept ≤ d*(Z))`. ∎

Every standard fraud-model evaluation reports the right-hand side. If `d` is monotone on the support, equality
holds. With more than two decision levels that condition is sufficient but not necessary: what matters is the
accept boundary. (The blueprint's wording "equality iff `d` is monotone on the fraud support" is corrected to
this form.) Checks `T2.*` verify both directions on random instances with partial support. In the
experiments, `theorem2_violations` counts payments whose attacked risk exceeds their truthful risk. It is zero
everywhere, because the truth is always a candidate presentation.

## 5. Theorem 3 (splitting)

Let the per-payment amount be a split-closed `F` coordinate. A *stateless* rule decides from the current
payment's presentation alone; the non-amount coordinates of the parts of a split payment are equal.

1. **Impossibility.** If a stateless rule rejects some amount `a` and accepts some `a' < a` with the other
   inputs fixed, then for every `k` a holder moves `k·a'` in `k` accepted payments of `a'`, and `k·a' > a` once
   `k > a/a'`. So no stateless rule bounds a sender's total exposure while also rejecting large single
   payments.
2. **Sufficiency.** Add a mandatory issuer-signed cumulative-spend claim `c`: the spend under this mandate or
   by this payer (or to this beneficiary) within a window, including the current payment. If acceptance is
   non-increasing in `c`, every other input fixed, then the total accepted per window is at most `B`, the
   largest `c` the rule accepts, minus the spend already in the window.

*Proof.* (1) Each part presents the same input as a single payment of `a'`, which the rule accepts; the
parts' total is `k·a'`. (2) The issuer keeps a counter per (key, window) that it updates atomically, adding a
payment's amount when it signs the claim and subtracting it if that payment is not sent. Rejected parts are
therefore released, so at any time the counter is at least the prior spend plus the accepted parts; in-flight
reservations only make it larger. Each signed `c` is bound to one payment's PID and nonce, so a claim with a
smaller `c` cannot be reused for a later part (E7: `cum_replay`). Let `t*` be the last accepted part. Its
claim carries `c_{t*} ≥ prior + (accepted total)`, and acceptance requires `c_{t*} ≤ B`. ∎

Atomicity and binding are necessary, not cosmetic. Without the lock, two parallel requests read the same old
value. Check `A3.racy` and `e3_atomicity.csv` show the non-atomic counter accepting up to 3.2× the bound under
32 parallel parts, while the atomic counter never exceeds it. The counter itself is a standard spending
ledger (CAPE uses one for agent budgets). What is claimed here is the stateless impossibility and carrying
the counter to another institution as an issuer-signed, PID-bound claim.

## 6. Disclosure privacy (Theorems 4 and 5, carried over from v1)

**Setting.** A receiver policy has frontier branches `M = {v_1, …, v_m}` (an antichain). `π_S` is the prior on
claim profiles conditioned on the policy being satisfied. The *signature* of `x` is `A(x) = {v ∈ M : v ≤ x}`.
A selection kernel `q(v | x)` on `A(x)` chooses the branch to disclose. Write `up(v) = π_S(X ≥ v)`,
`L_S(v) = −log₂ up(v)`, and *selection leakage*
`Λ(q) = Σ_v P(v) · KL(P(X | V=v) ‖ π_S(X | X ≥ v))`, the information revealed beyond what the disclosed claims
imply.

**Lemma 1.** `Λ(q) = I(X;V) − E[L_S(V)]`.
*Proof.* For `x ≥ v`, `P(x|v)/π_S(x | X ≥ v) = q(v|x)·up(v)/P(v)`. Taking the expectation of `log₂` over
`(X, V)` gives `I(X;V) + Σ_v P(v) log₂ up(v)`. ∎

**Theorem 4 (no free selection).** (a) Some kernel has `Λ = 0` iff `Σ_{v∈a} w_v = 1` for every signature `a`
with positive mass has a solution `w ≥ 0`; the kernel `q(v|x) = w_v` then attains it. (b) If the singleton
signatures `{v_a}`, `{v_b}` and some signature containing both have positive mass (e.g. the prior puts mass on
`v_a`, `v_b` and `v_a ∨ v_b`), every branch-revealing kernel has `Λ > 0`.
*Proof.* (a) `Λ = 0` iff every KL term vanishes, i.e. by the ratio in Lemma 1 iff `q(v|x) = P(v)/up(v) =: w_v`
does not depend on `x`. Normalisation over `A(x)` is the system. Conversely, a solution defines a kernel with
`P(x|v) = π_S(x)/up(v)`. (b) The singletons force `w_a = w_b = 1`, and the joint signature then sums to at
least 2. ∎

**Theorem 5 (repeated payments).** A holder pays the same receiver `n` times with unchanged evidence. Let
`J_n` be the join of the branches shown and `Λ_n = I(X;V^n) − E[L_S(J_n)]`.
1. With fresh draws and `q(v|x) ≥ q_min > 0` on `A(x)`, every supported branch has been shown after
   `n ≥ ln(m/δ)/q_min` payments with probability at least `1 − δ`.
   *Proof:* `P(some v ∈ A(x) unseen) ≤ Σ_v (1 − q(v|x))^n ≤ m e^{−n q_min} ≤ δ`. ∎
2. `I(X;V^n) ≤ n·I(X;V)`: the draws are conditionally independent given `X`. ∎
3. Sticky selection repeats the stored branch while it stays supported (and otherwise redraws it with a keyed
   PRF), so `V^n = (V_1, …, V_1)` and `Λ_n = Λ_1`. ∎ The sticky key is (account, receiver policy, evidence
   epoch). The epoch changes on reset-type events, so a stored choice never outlives its evidence.

Checks `T4.*` and `T5.*` verify these on random signature spaces. E5 measures them on receiver policies over
agent authority chains with the empirical prior of the agentic workload.

**How the parts fit.** Theorems 1–3 say what the receiver's model may decide on; Theorems 4–5 say what the
holder's minimal disclosure still reveals. Security and privacy are analysed in one model.

## 7. What is not claimed as new

Evidence games and the nested-range condition (Green and Laffont 1986), and strategic and withheld-feature
classification (Hardt et al. 2016; Krishnaswamy et al. 2021), are prior art. So are monotone learning
(LightGBM and XGBoost monotone constraints; non-negative-weight monotone networks), Ed25519 and salted-digest
selective disclosure, the atomic budget ledger (CAPE), and authority attenuation by minimum (SPKI, UCAN,
macaroons). What is claimed:

* the control-typed formulation for cross-institution payments with AI-agent senders;
* the characterization applied to learned payment-risk models, with the evaluation gap measured;
* the stateless splitting impossibility and the PID-bound, issuer-signed carriage of the cumulative counter;
* the repairs (exact lattice repair, and the box repair for tree ensembles with its soundness argument),
  together with the measured accuracy cost of each defence.
