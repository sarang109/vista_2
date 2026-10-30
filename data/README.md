# Datasets

Nothing in this directory is committed. Each dataset is read from the path below; experiments whose data is
missing are skipped and say so in the run log.

| dataset | path | how to get it | used by |
|---|---|---|---|
| IEEE-CIS Fraud Detection | `data/ieee_cis/train_transaction.csv`, `data/ieee_cis/train_identity.csv` | Kaggle competition `ieee-fraud-detection`; accept the competition rules on the website first, then `kaggle competitions download ieee-fraud-detection -f train_transaction.csv -p data/ieee_cis` (and the same for `train_identity.csv`) and unzip | E1, E2, E8 |
| Bank Account Fraud (NeurIPS 2022), Base variant | `data/baf/Base.csv` | Kaggle dataset `sgpjesus/bank-account-fraud-dataset-neurips-2022` (CC BY-NC-SA 4.0): `kaggle datasets download sgpjesus/bank-account-fraud-dataset-neurips-2022 -f Base.csv -p data/baf` and unzip | E1, E2, E8 |
| PaySim | `data/paysim.csv` and `data/paysim_cache/*.npy` | `python -m vista2.datasets.paysim` downloads the public log (`PS_20174392719_1491204439457_log.csv`, 494 MB, no account; Hugging Face mirror of the Kaggle dataset "Synthetic Financial Datasets For Fraud Detection"), checks its SHA-256 and the published statistics (6,362,620 rows, 8,213 frauds) and builds the column cache; `--csv PATH` uses a copy you already have | E3 |
| synthetic and agentic workloads | generated | `vista2.datasets.synthetic`, `vista2.datasets.agentic` | E0, E2, E4, E5, E6, E9–E11 |

The IEEE-CIS loader writes `data/ieee_cis/typed.parquet` on first use so later runs skip the CSV join.

PaySim is used for its amounts, timing and payer identities. Its payers almost never repeat (5,688 of
4,957,641 originating accounts appear more than once), so the repeated-payment study (E5) uses the agent
subscription workload instead.
