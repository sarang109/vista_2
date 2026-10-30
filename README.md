# VISTA v2: strategy-proof receiver decisions over attested evidence

**Research artifact for agentic and real-time payments.**

A receiving bank increasingly scores incoming payments with a machine-learning model over evidence that the
sender chooses to present. When the sender is a compromised account or a prompt-injected AI agent, that
choice is an attack surface. VISTA v2 asks which decision rules a receiving institution can safely run when
the sender picks which verified evidence to show, and answers it in one model.

* Every input is **control-typed** by who controls it: withholdable attested claims (`W`), freely chosen
  fields (`F`), or third-party facts (`T`).
* **Theorem 1:** a rule resists strategic presentation exactly when it is monotone in the sender's
  reachability order. Corollary 1 turns this into a training rule: monotone in `W`, blind to `F`, free in `T`.
* **Theorem 2:** every model that is not monotone has an *evaluation gap* between the fraud acceptance its
  evaluation reports and the acceptance it actually delivers.
* **Theorem 3:** no stateless rule bounds exposure under payment splitting; an issuer-signed, PID-bound
  cumulative-spend claim does.
* **Theorems 4–5** (carried over from v1): what the holder's minimal disclosure still reveals, per payment
  and across repeated agent payments, and how sticky selection stops the leak.

Statements and complete proofs are in [`docs/theory.md`](docs/theory.md). The feature typing of every
dataset is in [`docs/feature_typing.md`](docs/feature_typing.md).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[test,data]"   # macOS: brew install libomp (LightGBM)
.venv/bin/python -m pytest                                          # unit tests (~30 s)
.venv/bin/python -m vista2.run_all --config configs/smoke.json      # everything at reduced scale (~3 min)
./scripts/fetch_data.sh                                             # BAF, IEEE-CIS (Kaggle), PaySim cache
.venv/bin/python -m vista2.run_all --config configs/full.json       # full run, 5 seeds (8 cores)
```

Run single steps with `--only e3,e7`. Steps whose data is missing are skipped and logged. No API key or
paid service is needed beyond a free Kaggle account for the real datasets.

Outputs go to `results/<config>/`:

* `*.csv`: one file per experiment (per-seed rows in `*_runs.csv`, mean and standard deviation in the others);
* `figures/*.png`: regenerated from the CSVs;
* `SUMMARY.md`: headline tables and the two evidence gates, built only from the CSVs;
* `manifest.json`: config, library versions, hardware, per-step timings, SHA-256 of every output;
* `run.log`: the run's progress log.

## Layout

| module | role |
|---|---|
| `control.py` | `W`/`F`/`T` feature specs, typed tables, Algorithm 1 (ControlTypeAudit), model variants |
| `game.py` | finite evidence games: closure, realized decision `d*`, SP test, safe repair `d_`, splitting |
| `attack.py` | best-response presentation search against any scorer (withholding, splitting, timing, relabelling, text) |
| `repair.py` | Algorithm 2b (exact lattice repair for any scorer) and the box repair for tree ensembles |
| `learners.py` | LightGBM, XGBoost, MLP and a monotone-by-construction MLP |
| `cumulative.py` | Algorithm 3: atomic issuer counter, PID-bound signed cumulative claim, receiver verification |
| `vsac.py`, `trust.py`, `payment.py`, `crypto/` | holder-driven binding layer (VSAC v2) and trust registry, from v1 |
| `disclosure/` | selection leakage on the signature space, repeated payments, sticky selection (Theorems 4–5) |
| `datasets/` | synthetic real-time payments, agentic workload, BAF, IEEE-CIS, PaySim |
| `experiments/` | E0–E13 |
| `figures.py`, `report.py`, `run_all.py` | figures, summary, one-command runner |

## Experiments

| id | question | data |
|---|---|---|
| E0 | Do Theorems 1–5, Corollary 1, Algorithm 2b, Algorithm 3 and the box repair hold numerically? | random instances |
| E1 | How large is the evaluation gap for LightGBM, XGBoost and an MLP? | IEEE-CIS, BAF, synthetic |
| E2 | What does each defence cost (naive, registry flags, monotone, W-repair, full repair, control-typed)? | synthetic, IEEE-CIS, BAF |
| E3 | Does the cumulative claim bound exposure under splitting? Is atomicity needed? | PaySim |
| E4 | Do the four agentic adversaries behave as predicted (Gate B)? | agentic workload |
| E5 | How fast does fresh branch selection leak across repeated agent payments? | agentic prior |
| E6 | What does v2 cost in the payment path? | measured |
| E7 | Does the binding layer still reject every forgery, including on the cumulative claim? | v1 attack suite + new |
| **E8** | How sensitive are the real-data results to the typing? What would verifying self-reported fields buy? | BAF, IEEE-CIS |
| **E9** | How does the withholding attack depend on the rollout stage? | synthetic, 0–70% non-issuing banks |
| **E10** | Measured comparison: raw evidence, opaque score, policy token, SD-JWT, AP2-style caps, v1, v2 | agentic workload |
| **E11** | Is the attack search tight? Does the attack need white-box access (transfer attacks)? | synthetic |
| **E12** | Realized operating curves (fraud accepted vs legitimate held, truthful and attacked) | from E2 |
| **E13** | How does repair cost grow with the claim lattice (64 to 262,144 points)? | synthetic lattices |
| **E14** | Does the gap survive a bounded attacker (few fields, small value moves)? | IEEE-CIS, BAF, synthetic |

Bold rows are additions beyond the original plan.

## Results

See `results/full/SUMMARY.md` for every table. Headline numbers are filled in below from that file.

All numbers are from `results/full` (5 seeds unless stated; mean ± standard deviation over seeds; fraud
acceptance at the threshold that holds 5% of legitimate validation payments).

**Theory checks (E0).** 27 numerical checks on 50 to 3,000 random instances each (Theorems 1–5, Corollary 1 by
exhaustive enumeration of all 262,144 rules on a small typed space, Algorithm 2b, the box repair): no failures.
The non-atomic counter control breaks the splitting bound in 50 of 50 trials, as it should.

**The evaluation gap is real and large (E1, E2).**

| data | model | AUC | evaluated | withholding only | full attack |
|---|---|---|---|---|---|
| synthetic | naive LightGBM | 0.907 | 40.7% | 52.9% | 89.6% |
| synthetic | with registry flags | 0.907 | 40.5% | 49.9% | 87.7% |
| synthetic | monotone in W | 0.907 | 40.8% | 40.8% | 81.0% |
| synthetic | W-repaired (Alg. 2b) | 0.882 | 51.5% | 51.5% | 89.4% |
| synthetic | **control-typed** | **0.898** | **42.6%** | **42.6%** | **42.6%** |
| IEEE-CIS | naive LightGBM | 0.896 | 36.9% | 46.8% | 79.0% |
| IEEE-CIS | monotone in W | 0.894 | 37.0% | 37.0% | 76.5% |
| IEEE-CIS | W-repaired | 0.881 | 41.8% | 41.8% | 75.6% |
| IEEE-CIS | full repair (box) | 0.722 | 67.2% | 67.2% | 67.2% |
| IEEE-CIS | **control-typed** | **0.882** | **39.9%** | **39.9%** | **39.9%** |
| BAF | naive LightGBM | 0.892 | 42.1% | 44.3% | 100.0% |
| BAF | **control-typed** | **0.774** | **64.1%** | **64.1%** | **64.1%** |

The gap appears for LightGBM, XGBoost and an MLP alike (E1): on IEEE-CIS the full attacker raises fraud
acceptance by 41–47 points for every learner (to 77.4% for XGBoost and 92.5% for the MLP). Monotonicity in `W`
removes the withholding gain but not the `F` attacks, which are the larger effect on every dataset.

**Gate A passes on IEEE-CIS.** Control-typed training costs 0.014 AUC (0.896 → 0.882) and removes the gap
entirely: 39.9% of fraud accepted truthfully and under every attack. E8 moves each uncertain feature group
between types; the cost stays between 0.007 and 0.021 (only treating the anonymised numeric `id_` features as
attacker-controlled pushes it just past 0.02), so the result does not hinge on the typing.

**Gate A fails on BAF as typed today.** Control-typed training costs 0.118 AUC there, because self-reported
fields carry most of BAF's signal. E8 shows what would change that: if the sending bank attested the identity
fields the cost drops to 0.049, and with all self-reported fields attested to 0.029. On the synthetic workload
the cost is 0.009.

**The gap survives a bounded attacker (E14, 3 seeds).** The default attacker may move every freely chosen
field anywhere, so E14 limits it. On IEEE-CIS the conventional model goes from 37.1% fraud accepted to 59.8% when
the attacker may change a single field, 66.0% with two, and 55.0% when every field may only move within 5% of its
range of the true value (splitting into up to four parts and withholding stay allowed). On BAF a single field
takes it from 41.8% to 75.1%. The control-typed model stays at its truthful level under every budget (39.9% on
IEEE-CIS). The one place a weak attacker gains little is BAF with very small value moves (+4 points at a 2%
window), because its self-reported fields matter mainly at their extremes.

**Repairing a black box is not a substitute for control-typed training (E2, E13).** Repairing `W` only
(Algorithm 2b) costs 0.025 AUC and leaves the `F` attacks open. The full safe repair is SP by construction,
but on the synthetic model even the exact repair falls to AUC 0.772 (box relaxation: 0.444), and on IEEE-CIS
the box-repaired model falls to AUC 0.722. Exact
repair cost grows as `4^n` for a payment that shows nothing (270 ms at 262,144 lattice points); the box repair
stays near 1 ms but is loose exactly there.

**Splitting (E3, PaySim).** Every stateless rule passes 100% of fraud value once payments may be split into
16 parts. With the issuer-signed cumulative claim the worst accepted total per attacker window stays at the
bound (800,992 at a 3.5% legitimate hold) for every split count. Without an atomic counter, 32 parallel parts
pushed the accepted total to 3.2× the bound in 40 of 40 trials; with it, never.

**Agent payments (E4, Gate B passes).** Fraud accepted from prompt-injected agents:

| model | truthful | withhold + split | persuasive text | merchant retry | chain strip | all |
|---|---|---|---|---|---|---|
| naive | 32.0% | 57.4% | 61.6% | 51.2% | 51.5% | 82.7% |
| monotone in W | 32.6% | 40.2% | 61.8% | 50.5% | 32.6% | 79.7% |
| control-typed | 41.7% | 41.7% | 41.7% | 41.7% | 41.7% | 41.7% |
| control-typed + text hold | 37.7% | 37.7% | 43.8% | 37.7% | 37.7% | 43.8% |

A hold triggered by agent-written text ("text may only lower acceptance") looks better when evaluated and is
worse once the agent writes benign text: it spends hold budget on a signal the agent controls.

**Measured comparison (E10, agentic workload, full attacker).** Raw evidence sharing 83.6%, SD-JWT with an
unconstrained model 83.5%, opaque sender score 81.4%, AP2-style mandate caps plus model 72.5%, v1 policy plus
model 73.1%, v1 policy alone 98.6% (it is SP for withholding but a weak detector), **v2 47.0%** with no gap.
v2 sees the same claims as SD-JWT (5.45 bits); it does not reduce disclosure, it changes what the model may
do with it.

**Rollout stage (E9).** Contrary to the expectation that withholding is mainly a partial-rollout problem, the
gain from withholding is largest at full participation (+13.9 points) and shrinks as more banks issue no
claims (+3.5 at 70%): when claims are rare the model relies on them less.

**Attacker (E11).** Two rounds of coordinate search find what exhaustive enumeration of the claim lattice
finds. A transfer attacker with only a surrogate model trained on 30% of the data reaches 86% (vs 90%
white-box), so the attack does not need access to the receiver's model.

**Repeated agent payments (E5).** None of the three agent-chain policies admits zero selection leakage
(Theorem 4). With fresh randomised selection, leakage grows from 0.13 to 0.99 bits over ten payments
(three-branch policy) and a learned attacker's lift rises from 9% to 22% by the fourth payment; sticky
selection holds both at their one-payment values.

**Cost (E6, E7).** Issuing the credential and cumulative claim takes 0.15 ms at the sending bank. Verifying
both takes 0.26 ms p99, and scoring with the control-typed model brings the receiver's path to 0.82 ms p99.
The capsule is 1,635 bytes. 0 of 7,793 targeted forgeries (26 attack types, including six on the cumulative
claim) were accepted; withholding passes the binding layer by design and is handled by the decision rule.

## Honest limitations

* **Synthetic claims.** No public dataset contains interbank authentication evidence or agent mandates. The
  `W` claims of the synthetic and agentic workloads are generated, and so are their scenario labels; they are a
  controlled testbed, not evidence about real fraud rates.
* **Dataset semantics.** BAF is account opening, not payments. IEEE-CIS columns are anonymised. The `W`/`F`/`T`
  typing of both is a modelling assumption, published in `docs/feature_typing.md` and stress-tested by E8.
* **The default attacker may move every freely chosen field anywhere on its grid.** E14 bounds it (at most
  b fields, or values within a fraction of each field's range of the truth) to show how much of the gap a
  weaker attacker still gets.
* **The attacker is a search, not an oracle.** Coordinate search (two rounds) gives lower bounds on attack
  success; E11 shows that exhaustive enumeration of the claim lattice finds no more on the synthetic data.
* **Control-typed resistance holds by construction.** What the experiments measure is its accuracy cost and the
  size of the attacks it removes.
* **Honest-issuer assumption.** The sending bank signs truthfully and runs the counter atomically. The threat
  is the holder (account takeover or a prompt-injected agent), not the issuer.
* **PaySim payers rarely repeat**, so the cumulative bound costs nothing extra there; workloads with heavy repeat
  spending would pay for the bound in held legitimate payments (see `e3_bound_sweep.csv`).
* Repaired variants are slow to score, so E2 evaluates them on up to 20,000 legitimate and 300 fraudulent test
  payments per seed (all other variants use the full test split); their seed-to-seed spread is larger.
* CPU timings are measured on the machine that ran them; network latency is not modelled.

## Relation to v1 and to CAPE and MERIDIAN

The binding layer (VSAC, trust registry, payment encoding) is carried over from VISTA v1, adapted to
holder-driven presentation, and copied into this repository rather than imported. CAPE and MERIDIAN secure the
payer side of an agent payment; VISTA v2 studies the receiving institution's decision rule. The atomic
cumulative counter comes from CAPE's and MERIDIAN's budget ledgers, and the authority-chain minimum from
MERIDIAN's scope attenuation; both are reused and cited, not claimed.
