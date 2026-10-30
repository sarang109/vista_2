# VISTA v2 results: run `full`

Python 3.14.7, LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.1, 8 CPUs (arm64). Seeds: [1, 2, 3, 4, 5]. Fraud acceptance is measured at the threshold that holds 5% of legitimate validation payments; ± is the standard deviation over seeds.

| step | status | seconds |
|---|---|---|
| e0 theory checks (Theorems 1-5, Corollary 1, Algorithms 2b and 3) | ok | 4.1 |
| e3 splitting and the cumulative claim (PaySim) | ok | 48.6 |
| e4 agentic adversaries, Gate B | ok | 154.3 |
| e5 repeated agent payments | ok | 2.4 |
| e6 payment-path overhead | ok | 10.2 |
| e7 binding-layer attack suite | ok | 8.5 |
| e9 rollout-stage sweep | ok | 164.4 |
| e10 measured comparison of designs | ok | 294.0 |
| e11 attacker strength and transfer | ok | 34.3 |
| e13 repair cost against lattice size | ok | 7.7 |
| e1 evaluation gap on real data | ok | 500.5 |
| e2 cost of each defence, operating curves | ok | 3871.1 |
| e8 typing sensitivity | ok | 432.4 |

## E0 theory checks

27 checks, 0 failing. The non-atomic counter is expected to break the bound; it is the control for Algorithm 3's atomicity requirement.

| id | check | instances | failures |
|---|---|---|---|
| T1.dstar_ge_d | dstar_ge_d | 3000 | 0 |
| T1.dstar_monotone | dstar_monotone | 3000 | 0 |
| T1.sp_iff_fixed_point | sp_iff_fixed_point | 3000 | 0 |
| T1.sp_iff_monotone | sp_iff_monotone | 3000 | 0 |
| T1.repair_sp | repair_sp | 3000 | 0 |
| T1.repair_below_d | repair_below_d | 3000 | 0 |
| T1.repair_largest | repair_largest | 1865 | 0 |
| T1.repair_fixes_sp | repair_fixes_sp | 3000 | 0 |
| T1.closure_preorder | closure_preorder | 3000 | 0 |
| T1.dstar_idempotent | dstar_idempotent | 3000 | 0 |
| C1.enumeration | SP iff monotone in W, constant in complete F, free in T | 262144 | 0 |
| T2.gap_nonnegative | realized >= evaluated acceptance | 3000 | 0 |
| T2.equality | gap = 0 iff no supported non-accepted input reaches an accepted presentation | 3000 | 0 |
| T3.1.impossibility | k parts of an accepted amount move more than a rejected amount | 3000 | 0 |
| T3.2.sufficiency | accepted total per window <= B - prior spend | 3000 | 0 |
| A3.atomic | atomic counter: accepted total <= B under 16 parallel parts | 50 | 0 |
| A3.racy | racy counter: accepted total <= B under 16 parallel parts | 50 | 50 |
| A2b.dp | lattice DP = recursion = max over up-set | 300 | 0 |
| BOX.point | box repair with no moving coordinates equals the model | 300 | 0 |
| BOX.upper | box repair >= exact repair (W lattice) | 300 | 0 |
| BOX.exact_ge_model | exact repair >= model risk | 300 | 0 |
| BOX.monotone | repaired risk does not fall when claims are withheld | 600 | 0 |
| T4.obstruction | two-branch obstruction implies no zero-leakage kernel | 300 | 0 |
| T4.zero_kernel | LP solution gives Lambda = 0 | 300 | 0 |
| T5.mi_bound | I(X;V^n) <= n I(X;V) | 600 | 0 |
| T5.sticky | sticky selection keeps Lambda_n = Lambda_1 | 300 | 0 |
| T5.coverage | ln(m/delta)/q_min payments reveal every branch w.p. >= 1 - delta | 300 | 0 |

## E1 evaluation gap (Theorem 2)

| dataset | learner | attacker | AUC | evaluated | realized | gap |
|---|---|---|---|---|---|---|
| ieee_cis | lightgbm | withholding | 0.896 ± 0.002 | 36.9% ± 0.4% | 46.8% ± 1.1% | 9.9% ± 1.0% |
| ieee_cis | lightgbm | f_only | 0.896 ± 0.002 | 36.9% ± 0.4% | 76.6% ± 0.7% | 39.6% ± 0.9% |
| ieee_cis | lightgbm | full | 0.896 ± 0.002 | 36.9% ± 0.4% | 79.0% ± 1.5% | 42.0% ± 1.8% |
| ieee_cis | xgboost | withholding | 0.896 ± 0.001 | 36.2% ± 0.7% | 46.6% ± 0.9% | 10.4% ± 1.1% |
| ieee_cis | xgboost | f_only | 0.896 ± 0.001 | 36.2% ± 0.7% | 75.1% ± 0.7% | 38.9% ± 1.3% |
| ieee_cis | xgboost | full | 0.896 ± 0.001 | 36.2% ± 0.7% | 77.4% ± 1.6% | 41.2% ± 1.7% |
| ieee_cis | mlp | withholding | 0.838 ± 0.008 | 45.6% ± 1.4% | 67.2% ± 2.2% | 21.6% ± 2.4% |
| ieee_cis | mlp | f_only | 0.838 ± 0.008 | 45.6% ± 1.4% | 83.2% ± 1.7% | 37.6% ± 2.9% |
| ieee_cis | mlp | full | 0.838 ± 0.008 | 45.6% ± 1.4% | 92.5% ± 2.7% | 46.9% ± 3.4% |
| baf | lightgbm | withholding | 0.892 ± 0.000 | 42.1% ± 0.5% | 44.3% ± 0.3% | 2.2% ± 0.4% |
| baf | lightgbm | f_only | 0.892 ± 0.000 | 42.1% ± 0.5% | 100.0% | 57.9% ± 0.5% |
| baf | lightgbm | full | 0.892 ± 0.000 | 42.1% ± 0.5% | 100.0% | 57.9% ± 0.5% |
| baf | xgboost | withholding | 0.892 ± 0.000 | 42.5% ± 0.4% | 44.8% ± 0.3% | 2.3% ± 0.2% |
| baf | xgboost | f_only | 0.892 ± 0.000 | 42.5% ± 0.4% | 100.0% | 57.5% ± 0.4% |
| baf | xgboost | full | 0.892 ± 0.000 | 42.5% ± 0.4% | 100.0% | 57.5% ± 0.4% |
| baf | mlp | withholding | 0.843 ± 0.021 | 60.3% ± 7.3% | 64.3% ± 7.4% | 4.0% ± 0.9% |
| baf | mlp | f_only | 0.843 ± 0.021 | 60.3% ± 7.3% | 100.0% | 39.7% ± 7.3% |
| baf | mlp | full | 0.843 ± 0.021 | 60.3% ± 7.3% | 100.0% | 39.7% ± 7.3% |
| synthetic | lightgbm | withholding | 0.907 ± 0.000 | 40.5% ± 0.4% | 49.9% ± 0.5% | 9.4% ± 0.6% |
| synthetic | lightgbm | f_only | 0.907 ± 0.000 | 40.5% ± 0.4% | 80.7% ± 0.5% | 40.2% ± 0.7% |
| synthetic | lightgbm | full | 0.907 ± 0.000 | 40.5% ± 0.4% | 87.7% ± 0.2% | 47.2% ± 0.2% |
| synthetic | xgboost | withholding | 0.907 ± 0.000 | 40.4% ± 0.3% | 49.0% ± 0.5% | 8.6% ± 0.7% |
| synthetic | xgboost | f_only | 0.907 ± 0.000 | 40.4% ± 0.3% | 79.4% ± 0.6% | 39.0% ± 0.7% |
| synthetic | xgboost | full | 0.907 ± 0.000 | 40.4% ± 0.3% | 86.8% ± 1.0% | 46.4% ± 1.2% |
| synthetic | mlp | withholding | 0.903 ± 0.004 | 42.4% ± 0.8% | 44.9% ± 1.4% | 2.5% ± 1.3% |
| synthetic | mlp | f_only | 0.903 ± 0.004 | 42.4% ± 0.8% | 74.7% ± 1.7% | 32.4% ± 1.3% |
| synthetic | mlp | full | 0.903 ± 0.004 | 42.4% ± 0.8% | 77.4% ± 2.0% | 35.0% ± 1.7% |

## E2 cost of each defence

**synthetic**

| model | AUC | legit held | fraud acc. truthful | withholding | full attack | fraud value acc. (full) |
|---|---|---|---|---|---|---|
| naive | 0.907 ± 0.000 | 5.1% ± 0.0% | 40.7% ± 0.3% | 52.9% ± 0.5% | 89.6% ± 0.5% | 65.1% ± 0.8% |
| unconstrained | 0.907 ± 0.000 | 5.1% ± 0.1% | 40.5% ± 0.4% | 49.9% ± 0.5% | 87.7% ± 0.2% | 62.3% ± 0.8% |
| monotone_w | 0.907 ± 0.000 | 5.2% ± 0.0% | 40.8% ± 0.4% | 40.8% ± 0.4% | 81.0% ± 0.6% | 53.2% ± 0.9% |
| repaired_w | 0.882 ± 0.005 | 5.2% ± 0.2% | 51.5% ± 1.8% | 51.5% ± 1.8% | 89.4% ± 2.5% | 71.3% ± 8.5% |
| repaired_full | 0.473 ± 0.041 | 4.9% ± 0.2% | 98.5% ± 0.2% | 98.5% ± 0.2% | 98.5% ± 0.2% | 99.9% ± 0.0% |
| control_typed | 0.898 ± 0.000 | 5.1% ± 0.0% | 42.6% ± 0.4% | 42.6% ± 0.4% | 42.6% ± 0.4% | 16.8% ± 0.2% |

**ieee_cis**

| model | AUC | legit held | fraud acc. truthful | withholding | full attack | fraud value acc. (full) |
|---|---|---|---|---|---|---|
| naive | 0.896 ± 0.002 | 5.7% ± 0.0% | 36.9% ± 0.4% | 46.8% ± 1.1% | 79.0% ± 1.5% | 88.8% ± 0.5% |
| monotone_w | 0.894 ± 0.002 | 5.8% ± 0.0% | 37.0% ± 0.6% | 37.0% ± 0.6% | 76.5% ± 1.2% | 86.9% ± 1.1% |
| repaired_w | 0.881 ± 0.007 | 5.8% ± 0.2% | 41.8% ± 2.6% | 41.8% ± 2.6% | 75.6% ± 2.4% | 88.3% ± 2.7% |
| repaired_full | 0.722 ± 0.021 | 4.7% ± 0.3% | 67.2% ± 3.0% | 67.2% ± 3.0% | 67.2% ± 3.0% | 87.1% ± 1.6% |
| control_typed | 0.882 ± 0.001 | 5.8% ± 0.1% | 39.9% ± 0.5% | 39.9% ± 0.5% | 39.9% ± 0.5% | 44.9% ± 0.6% |

**baf**

| model | AUC | legit held | fraud acc. truthful | withholding | full attack | fraud value acc. (full) |
|---|---|---|---|---|---|---|
| naive | 0.892 ± 0.000 | 6.0% ± 0.0% | 42.1% ± 0.5% | 44.3% ± 0.3% | 100.0% |  |
| monotone_w | 0.891 ± 0.000 | 6.0% ± 0.1% | 41.9% ± 0.6% | 41.9% ± 0.6% | 100.0% |  |
| repaired_w | 0.896 ± 0.013 | 6.0% ± 0.2% | 41.0% ± 3.5% | 41.0% ± 3.5% | 100.0% |  |
| repaired_full | 0.671 ± 0.029 | 3.3% ± 0.5% | 92.1% ± 2.5% | 92.1% ± 2.5% | 92.1% ± 2.5% |  |
| control_typed | 0.774 ± 0.001 | 7.0% ± 0.2% | 64.1% ± 0.9% | 64.1% ± 0.9% | 64.1% ± 0.9% |  |

Full repair of the unconstrained synthetic model, exact enumeration against the box relaxation (subsample):

| repair | AUC | legit held | fraud accepted |
|---|---|---|---|
| exact_full_repair | 0.772 | 4.8% | 66.7% |
| box_full_repair | 0.444 | 2.5% | 99.3% |
| unrepaired | 0.913 | 5.2% | 37.3% |

### Gate A

- ieee_cis: realized gap of the naive model 42.0 points; control-typed AUC cost 0.014 -> Gate A **passes** on this dataset
- baf: realized gap of the naive model 57.9 points; control-typed AUC cost 0.118 -> Gate A **fails** on this dataset

## E3 splitting (Theorem 3), PaySim

| rule | K | legit held | fraud value accepted | max exposure | bound holds |
|---|---|---|---|---|---|
| stateless_cap | 1 | 3.5% | 9.6% | 799798 |  |
| stateless_cap | 4 | 3.5% | 34.6% | 3200820 |  |
| stateless_cap | 16 | 3.5% | 100.0% | 10000000 |  |
| stateless_cap | 64 | 3.5% | 100.0% | 10000000 |  |
| cumulative_cap | 1 | 3.5% | 9.6% | 799798 | True |
| cumulative_cap | 4 | 3.5% | 20.4% | 800468 | True |
| cumulative_cap | 16 | 3.5% | 29.0% | 800964 | True |
| cumulative_cap | 64 | 3.5% | 29.4% | 800992 | True |
| stateless_model | 1 | 5.1% | 16.3% | 2000720 |  |
| stateless_model | 4 | 5.1% | 47.8% | 8006430 |  |
| stateless_model | 16 | 5.1% | 100.0% | 10000000 |  |
| stateless_model | 64 | 5.1% | 100.0% | 10000000 |  |
| cumulative_model | 1 | 3.3% | 9.0% | 918088 |  |
| cumulative_model | 4 | 3.3% | 19.0% | 918671 |  |
| cumulative_model | 16 | 3.3% | 26.5% | 919130 |  |
| cumulative_model | 64 | 3.3% | 27.7% | 919139 |  |

| counter | trials | trials over bound | worst accepted total | claim p99 ms |
|---|---|---|---|---|
| atomic | 40 | 0 | 10000 | 0.401 |
| racy | 40 | 40 | 32000 | 2.127 |

## E4 agentic adversaries (Gate B)

| model | truthful | withhold_split | persuasive_text | merchant_retry | chain_strip | full |
|---|---|---|---|---|---|---|
| naive | 32.0% ± 0.2% | 57.4% ± 0.4% | 61.6% ± 0.6% | 51.2% ± 0.7% | 51.5% ± 0.3% | 82.7% ± 0.2% |
| unconstrained | 32.0% ± 0.3% | 56.4% ± 0.6% | 61.7% ± 0.4% | 51.0% ± 0.9% | 49.3% ± 0.4% | 82.5% ± 0.2% |
| monotone_w | 32.6% ± 0.4% | 40.2% ± 0.3% | 61.8% ± 0.6% | 50.5% ± 0.8% | 32.6% ± 0.4% | 79.7% ± 0.6% |
| control_typed | 41.7% ± 0.3% | 41.7% ± 0.3% | 41.7% ± 0.3% | 41.7% ± 0.3% | 41.7% ± 0.3% | 41.7% ± 0.3% |
| control_typed_text_hold | 37.7% ± 0.2% | 37.7% ± 0.2% | 43.8% ± 0.3% | 37.7% ± 0.2% | 37.7% ± 0.2% | 43.8% ± 0.3% |

Gate B **passes** (control-typed gains under no adversary; the naive model gains under 4 of 4).

## E5 repeated agent payments (Theorems 4 and 5)

| policy | selection | Λ1 (bits) | Λ5 | Λ10 |
|---|---|---|---|---|
| P1_three_branch | DET | 0.460 | 0.460 | 0.460 |
| P1_three_branch | exp1.0_fresh | 0.125 | 0.759 | 0.991 |
| P1_three_branch | exp1.0_sticky | 0.125 | 0.125 | 0.125 |
| P2_two_branch | DET | 0.181 | 0.181 | 0.181 |
| P2_two_branch | exp1.0_fresh | 0.075 | 0.259 | 0.349 |
| P2_two_branch | exp1.0_sticky | 0.075 | 0.075 | 0.075 |
| P3_four_branch | DET | 0.603 | 0.603 | 0.603 |
| P3_four_branch | exp1.0_fresh | 0.147 | 1.036 | 1.443 |
| P3_four_branch | exp1.0_sticky | 0.147 | 0.147 | 0.147 |

Zero-leakage feasible (Theorem 4): P1_three_branch False, P2_two_branch False, P3_four_branch False

| policy | strategy | payment n | branches revealed | attacker lift |
|---|---|---|---|---|
| P1_three_branch | DET | 1 | 1.00 | 28.6% |
| P1_three_branch | DET | 4 | 1.13 | 29.1% |
| P1_three_branch | DET | 12 | 1.40 | 29.2% |
| P1_three_branch | RAND_fresh | 1 | 1.00 | 9.3% |
| P1_three_branch | RAND_fresh | 4 | 1.55 | 22.0% |
| P1_three_branch | RAND_fresh | 12 | 2.01 | 13.8% |
| P1_three_branch | RAND_sticky | 1 | 1.00 | 8.5% |
| P1_three_branch | RAND_sticky | 4 | 1.17 | 9.7% |
| P1_three_branch | RAND_sticky | 12 | 1.53 | 11.1% |
| P2_two_branch | DET | 1 | 1.00 | 15.0% |
| P2_two_branch | DET | 4 | 1.07 | 14.8% |
| P2_two_branch | DET | 12 | 1.22 | 15.2% |
| P2_two_branch | RAND_fresh | 1 | 1.00 | 10.2% |
| P2_two_branch | RAND_fresh | 4 | 1.19 | 10.7% |
| P2_two_branch | RAND_fresh | 12 | 1.43 | 10.7% |
| P2_two_branch | RAND_sticky | 1 | 1.00 | 10.1% |
| P2_two_branch | RAND_sticky | 4 | 1.09 | 10.4% |
| P2_two_branch | RAND_sticky | 12 | 1.27 | 10.6% |
| P3_four_branch | DET | 1 | 1.00 | 34.3% |
| P3_four_branch | DET | 4 | 1.15 | 33.9% |
| P3_four_branch | DET | 12 | 1.47 | 33.3% |
| P3_four_branch | RAND_fresh | 1 | 1.00 | 10.2% |
| P3_four_branch | RAND_fresh | 4 | 1.79 | 25.7% |
| P3_four_branch | RAND_fresh | 12 | 2.48 | 16.4% |
| P3_four_branch | RAND_sticky | 1 | 1.00 | 11.1% |
| P3_four_branch | RAND_sticky | 4 | 1.20 | 9.1% |
| P3_four_branch | RAND_sticky | 12 | 1.66 | 9.4% |

## E6 payment-path overhead

| stage | p50 ms | p99 ms | batch ms / payment | bytes |
|---|---|---|---|---|
| issue_credential | 0.103 | 0.129 |  |  |
| issue_cumulative | 0.044 | 0.059 |  |  |
| present | 0.021 | 0.029 |  |  |
| verify | 0.223 | 0.264 |  |  |
| ct_score | 0.435 | 0.576 |  |  |
| receiver_total | 0.661 | 0.819 |  |  |
| capsule_bytes |  |  |  | 1635 |
| score_unconstrained | 0.404 | 0.470 | 0.003 |  |
| score_exact_w_repair | 1.288 | 2.510 | 0.487 |  |
| score_box_w_repair | 3.811 | 4.026 | 0.157 |  |
| score_box_full_repair | 3.813 | 4.432 | 0.158 |  |
| exact_w_repair_upset_size |  |  |  |  |

## E7 binding layer

0 of 7793 targeted mutations (26 attack types) accepted by the full verifier; withholding is accepted by design (300 trials) and handled by the decision rule.

## E8 typing sensitivity

| dataset | change | kind | control-typed AUC | AUC cost | naive realized |
|---|---|---|---|---|---|
| baf | baseline typing | baseline | 0.769 | 0.120 | 100.0% |
| baf | customer_age -> T | optimistic | 0.787 | 0.102 | 100.0% |
| baf | current_address_months_count -> T | optimistic | 0.796 | 0.092 | 100.0% |
| baf | prev_address_months_count -> T | optimistic | 0.794 | 0.095 | 100.0% |
| baf | days_since_request -> T | optimistic | 0.772 | 0.116 | 100.0% |
| baf | device_distinct_emails_8w -> F | pessimistic | 0.764 | 0.125 | 100.0% |
| baf | device_fraud_count -> F | pessimistic | 0.769 | 0.120 | 100.0% |
| baf | foreign_request -> F | pessimistic | 0.769 | 0.120 | 100.0% |
| baf | has_other_cards -> F | pessimistic | 0.750 | 0.140 | 100.0% |
| baf | verify_income_employment | verified_upgrade | 0.791 | 0.098 | 100.0% |
| baf | verify_identity_fields | verified_upgrade | 0.839 | 0.049 | 100.0% |
| baf | verify_all_self_reported | verified_upgrade | 0.859 | 0.029 | 94.3% |
| ieee_cis | baseline typing | baseline | 0.879 | 0.015 | 75.0% |
| ieee_cis | group:D -> F | pessimistic | 0.879 | 0.017 | 100.0% |
| ieee_cis | group:V -> F | pessimistic | 0.881 | 0.014 | 100.0% |
| ieee_cis | group:id_num -> F | pessimistic | 0.875 | 0.021 | 92.1% |
| ieee_cis | ProductCD -> F | pessimistic | 0.877 | 0.016 | 75.6% |
| ieee_cis | group:dist -> T | optimistic | 0.880 | 0.014 | 71.3% |
| ieee_cis | group:M -> T | optimistic | 0.884 | 0.008 | 75.5% |
| ieee_cis | group:id_flags -> T | optimistic | 0.879 | 0.015 | 74.8% |
| ieee_cis | verify_email_and_address | verified_upgrade | 0.887 | 0.007 | 64.9% |
| ieee_cis | verify_device | verified_upgrade | 0.881 | 0.014 | 70.5% |

## E9 rollout stage

| no-claim banks | model | truthful | withholding | gain |
|---|---|---|---|---|
| 0 | naive | 39.9% ± 1.5% | 53.8% ± 1.2% | 13.9% ± 0.6% |
| 0 | unconstrained | 39.4% ± 1.2% | 51.2% ± 1.3% | 11.8% ± 2.0% |
| 0 | monotone_w | 39.8% ± 1.3% | 39.8% ± 1.3% | 0.0% |
| 0 | control_typed | 42.1% ± 1.3% | 42.1% ± 1.3% | 0.0% |
| 0.1 | naive | 39.8% ± 2.2% | 53.5% ± 1.6% | 13.7% ± 1.9% |
| 0.1 | unconstrained | 39.8% ± 1.8% | 50.6% ± 0.1% | 10.8% ± 1.8% |
| 0.1 | monotone_w | 40.4% ± 2.0% | 40.4% ± 2.0% | 0.0% |
| 0.1 | control_typed | 42.8% ± 1.8% | 42.8% ± 1.8% | 0.0% |
| 0.3 | naive | 40.9% ± 1.6% | 51.7% ± 2.3% | 10.8% ± 2.1% |
| 0.3 | unconstrained | 41.6% ± 1.6% | 50.3% ± 0.4% | 8.7% ± 1.3% |
| 0.3 | monotone_w | 41.5% ± 2.1% | 41.5% ± 2.1% | 0.0% |
| 0.3 | control_typed | 43.7% ± 1.6% | 43.7% ± 1.6% | 0.0% |
| 0.5 | naive | 43.0% ± 1.8% | 50.2% ± 2.2% | 7.1% ± 2.3% |
| 0.5 | unconstrained | 43.1% ± 1.9% | 48.6% ± 1.8% | 5.5% ± 2.1% |
| 0.5 | monotone_w | 43.1% ± 2.2% | 43.1% ± 2.2% | 0.0% |
| 0.5 | control_typed | 45.1% ± 1.8% | 45.1% ± 1.8% | 0.0% |
| 0.7 | naive | 45.4% ± 1.2% | 48.9% ± 1.8% | 3.5% ± 0.6% |
| 0.7 | unconstrained | 44.8% ± 1.6% | 47.7% ± 1.9% | 3.0% ± 0.3% |
| 0.7 | monotone_w | 45.5% ± 1.7% | 45.5% ± 1.7% | 0.0% |
| 0.7 | control_typed | 46.8% ± 2.0% | 46.8% ± 2.0% | 0.0% |

## E10 measured comparison of designs (agentic workload)

| design | legit held | truthful | withholding | F only | full | claim bits seen |
|---|---|---|---|---|---|---|
| raw_evidence | 5.0% ± 0.1% | 39.8% ± 0.2% | 55.9% ± 0.2% | 81.8% ± 0.6% | 83.6% ± 0.2% | 5.45 |
| sdjwt_unconstrained | 5.0% ± 0.0% | 39.8% ± 0.3% | 54.3% ± 0.4% | 81.5% ± 0.6% | 83.5% ± 0.2% | 5.45 |
| vista_v2 | 5.1% ± 0.1% | 47.0% ± 0.1% | 47.0% ± 0.1% | 47.0% ± 0.1% | 47.0% ± 0.1% | 5.45 |
| opaque_score | 4.9% ± 0.0% | 39.9% ± 0.2% | 39.9% ± 0.2% | 81.4% ± 0.7% | 81.4% ± 0.7% | 6.28 |
| policy_token | 4.0% | 92.9% | 92.9% | 98.6% ± 0.0% | 98.6% ± 0.0% | 2.08 |
| vista_v1 | 4.0% | 92.9% | 92.9% | 98.6% ± 0.0% | 98.6% ± 0.0% | 5.45 |
| vista_v1_model | 4.1% ± 0.0% | 42.0% ± 0.2% | 51.4% ± 0.4% | 72.5% ± 0.5% | 73.1% ± 0.5% | 5.45 |
| ap2_mandate_plus_model | 5.1% ± 0.0% | 41.6% ± 0.3% | 51.5% ± 0.3% | 71.8% ± 0.6% | 72.5% ± 0.5% | 5.45 |

## E11 attacker strength and knowledge

| model | attacker | knowledge | fraud accepted |
|---|---|---|---|
| naive | truthful | white_box | 40.5% |
| naive | coord_1_round | white_box | 87.8% |
| naive | coord_2_rounds | white_box | 89.6% |
| naive | coord_3_rounds | white_box | 89.6% |
| naive | exhaustive_w_plus_2_rounds | white_box | 89.6% |
| naive | surrogate_lightgbm_30pct | transfer | 86.4% |
| naive | surrogate_xgboost_30pct | transfer | 85.6% |
| unconstrained | truthful | white_box | 41.0% |
| unconstrained | coord_1_round | white_box | 85.7% |
| unconstrained | coord_2_rounds | white_box | 88.0% |
| unconstrained | coord_3_rounds | white_box | 88.3% |
| unconstrained | exhaustive_w_plus_2_rounds | white_box | 87.9% |
| unconstrained | surrogate_lightgbm_30pct | transfer | 82.7% |
| unconstrained | surrogate_xgboost_30pct | transfer | 82.8% |
| monotone_w | truthful | white_box | 40.2% |
| monotone_w | coord_1_round | white_box | 79.2% |
| monotone_w | coord_2_rounds | white_box | 80.1% |
| monotone_w | coord_3_rounds | white_box | 80.1% |
| monotone_w | exhaustive_w_plus_2_rounds | white_box | 80.1% |
| monotone_w | surrogate_lightgbm_30pct | transfer | 76.5% |
| monotone_w | surrogate_xgboost_30pct | transfer | 76.4% |
| control_typed | truthful | white_box | 42.8% |
| control_typed | coord_1_round | white_box | 42.8% |
| control_typed | coord_2_rounds | white_box | 42.8% |
| control_typed | coord_3_rounds | white_box | 42.8% |
| control_typed | exhaustive_w_plus_2_rounds | white_box | 42.8% |
| control_typed | surrogate_lightgbm_30pct | transfer | 42.8% |
| control_typed | surrogate_xgboost_30pct | transfer | 42.8% |

## E13 repair cost against lattice size

| claims | lattice | payments | exact ms | exact calls | box ms | box excess (mean) |
|---|---|---|---|---|---|---|
| 3 | 64 | random | 0.25 | 11 | 0.06 | 0.008 |
| 3 | 64 | nothing_shown | 0.17 | 64 | 0.27 | 0.057 |
| 4 | 256 | random | 0.46 | 33 | 0.08 | 0.017 |
| 4 | 256 | nothing_shown | 0.38 | 256 | 0.34 | 0.158 |
| 5 | 1024 | random | 0.59 | 107 | 0.07 | 0.028 |
| 5 | 1024 | nothing_shown | 1.21 | 1024 | 0.31 | 0.335 |
| 6 | 4096 | random | 0.73 | 193 | 0.08 | 0.019 |
| 6 | 4096 | nothing_shown | 4.30 | 4096 | 0.35 | 0.520 |
| 7 | 16384 | random | 1.17 | 571 | 0.09 | 0.020 |
| 7 | 16384 | nothing_shown | 16.20 | 16384 | 0.39 | 0.496 |
| 8 | 65536 | random | 2.15 | 1363 | 0.10 | 0.027 |
| 8 | 65536 | nothing_shown | 62.35 | 65536 | 0.43 | 0.629 |
| 9 | 262144 | random | 3.12 | 2105 | 0.43 | 0.017 |
| 9 | 262144 | nothing_shown | 269.61 | 262144 | 1.37 | 0.790 |

## Figures

- `figures/fig_e10_comparison.png`
- `figures/fig_e12_curves_baf.png`
- `figures/fig_e12_curves_ieee_cis.png`
- `figures/fig_e12_curves_synthetic.png`
- `figures/fig_e13_repair_scaling.png`
- `figures/fig_e1_evaluation_gap.png`
- `figures/fig_e2_defences.png`
- `figures/fig_e3_splitting_exposure.png`
- `figures/fig_e4_agentic.png`
- `figures/fig_e5_repeated_leakage.png`
- `figures/fig_e9_rollout.png`
