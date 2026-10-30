# Feature typing per dataset

Generated from the dataset loaders (`src/vista2/datasets/`). `uncertain` marks a typing that is a modelling
assumption; E8 moves those features to another type and reports the effect. W columns are integer coded with
0 = nothing shown. For F columns, *reach* is the manipulation the attacker may apply.

## Synthetic real-time payments



| feature | type | reach | uncertain | note |
|---|---|---|---|---|
| auth_strength | W | chain |  | issuer-signed threshold claim |
| device_continuity | W | chain |  | issuer-signed threshold claim |
| credential_age | W | chain |  | issuer-signed threshold claim |
| phishing_resistant | W | chain |  | issuer-signed threshold claim |
| dynamic_binding | W | chain |  | issuer-signed threshold claim |
| account_age | W | chain |  | issuer-signed threshold claim |
| hour | F | complete |  | sender picks the time |
| amount | F | split |  | per-payment amount, split-closed |
| category | F | step |  | unsigned purpose code; relabelled one step |
| cap_auth_strength | T |  |  | registry: bank can issue this claim |
| cap_device_continuity | T |  |  | registry: bank can issue this claim |
| cap_credential_age | T |  |  | registry: bank can issue this claim |
| cap_phishing_resistant | T |  |  | registry: bank can issue this claim |
| cap_dynamic_binding | T |  |  | registry: bank can issue this claim |
| cap_account_age | T |  |  | registry: bank can issue this claim |
| signed_mcc | T |  |  | acquirer-signed merchant category |
| beneficiary_new | T |  |  | receiver-observed |
| beneficiary_inflow_24h | T |  |  | receiver-observed |
| cum_spend_24h | T |  |  | issuer-signed cumulative claim |

## Agentic workload



| feature | type | reach | uncertain | note |
|---|---|---|---|---|
| mandate_kind | W | chain |  | delegated-authority chain, min over links |
| scope_coverage | W | chain |  | delegated-authority chain, min over links |
| agent_identity | W | chain |  | delegated-authority chain, min over links |
| tool_policy | W | chain |  | delegated-authority chain, min over links |
| user_reconfirmation | W | chain |  | delegated-authority chain, min over links |
| auth_strength | W | chain |  | human authentication claim |
| device_continuity | W | chain |  | human authentication claim |
| hour | F | complete |  |  |
| amount | F | split |  |  |
| merchant_id | F | complete |  | choice among equivalent merchants; retries |
| text_score | F | complete |  | agent-written justification |
| is_agent | T |  |  |  |
| cap_human | T |  |  |  |
| cum_mandate_spend | T |  |  |  |
| cart_mcc | T |  |  |  |
| merchant_age_days | T |  |  |  |
| cap_mandate_kind | T |  |  |  |
| cap_scope_coverage | T |  |  |  |
| cap_agent_identity | T |  |  |  |
| cap_tool_policy | T |  |  |  |
| cap_user_reconfirmation | T |  |  |  |

## Bank Account Fraud (Base)



| feature | type | reach | uncertain | note |
|---|---|---|---|---|
| phone_home_valid | W | chain |  | verified; applicant may omit |
| phone_mobile_valid | W | chain |  | verified; applicant may omit |
| bank_months_count | W | chain |  | verified; applicant may omit |
| income | F | complete |  | self-reported or chosen by the applicant |
| name_email_similarity | F | complete |  | self-reported or chosen by the applicant |
| customer_age | F | complete | yes | self-reported or chosen by the applicant |
| current_address_months_count | F | complete | yes | self-reported or chosen by the applicant |
| prev_address_months_count | F | complete | yes | self-reported or chosen by the applicant |
| days_since_request | F | complete | yes | self-reported or chosen by the applicant |
| intended_balcon_amount | F | complete |  | self-reported or chosen by the applicant |
| payment_type | F | complete |  | self-reported or chosen by the applicant |
| employment_status | F | complete |  | self-reported or chosen by the applicant |
| housing_status | F | complete |  | self-reported or chosen by the applicant |
| email_is_free | F | complete |  | self-reported or chosen by the applicant |
| proposed_credit_limit | F | complete |  | self-reported or chosen by the applicant |
| source | F | complete |  | self-reported or chosen by the applicant |
| session_length_in_minutes | F | complete |  | self-reported or chosen by the applicant |
| device_os | F | complete |  | self-reported or chosen by the applicant |
| keep_alive_session | F | complete |  | self-reported or chosen by the applicant |
| credit_risk_score | T |  |  | bureau or bank observed |
| velocity_6h | T |  |  | bureau or bank observed |
| velocity_24h | T |  |  | bureau or bank observed |
| velocity_4w | T |  |  | bureau or bank observed |
| zip_count_4w | T |  |  | bureau or bank observed |
| bank_branch_count_8w | T |  |  | bureau or bank observed |
| date_of_birth_distinct_emails_4w | T |  |  | bureau or bank observed |
| device_distinct_emails_8w | T |  | yes | bureau or bank observed |
| device_fraud_count | T |  | yes | bureau or bank observed |
| foreign_request | T |  | yes | bureau or bank observed |
| has_other_cards | T |  | yes | bureau or bank observed |

## IEEE-CIS

Typing applied by `datasets/ieee_cis.py` (the V columns kept are a correlation-pruned subset chosen at load time):

| group | type | reach | uncertain |
|---|---|---|---|
| M1-M9 match flags | W | poset (nothing shown below each recorded value) | yes |
| id_12, id_15, id_16, id_28, id_29 (found / not found) | W | chain | yes |
| id_35-id_38 binary flags | W | poset | yes |
| TransactionAmt | F | split (counterpart: cum_amt_24h) | |
| hour, weekday (from TransactionDT) | F | complete | |
| P_emaildomain, R_emaildomain, addr1, addr2, DeviceType, DeviceInfo, id_30, id_31, id_33 | F | complete | |
| dist1, dist2 | F | complete | yes |
| card1-card6, C1-C14 | T | | |
| ProductCD, D1-D15, V subset, other id_ columns | T | | yes |
| cum_amt_24h (card spend in previous 24 h incl. this one) | T | issuer-signed cumulative claim | |
