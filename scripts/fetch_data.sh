#!/usr/bin/env bash
# Download the public datasets into data/. BAF and IEEE-CIS need the Kaggle CLI with an API token, and for
# IEEE-CIS the competition rules accepted on kaggle.com by the same account. PaySim needs no account.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/baf data/ieee_cis data/paysim_cache

if [ ! -f data/baf/Base.csv ]; then
  kaggle datasets download sgpjesus/bank-account-fraud-dataset-neurips-2022 -f Base.csv -p data/baf
  (cd data/baf && unzip -o -q Base.csv.zip && rm Base.csv.zip)
fi

for f in train_transaction.csv train_identity.csv; do
  if [ ! -f "data/ieee_cis/$f" ]; then
    kaggle competitions download ieee-fraud-detection -f "$f" -p data/ieee_cis || \
      echo "IEEE-CIS: accept the competition rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules"
    (cd data/ieee_cis && for z in *.zip; do [ -f "$z" ] && unzip -o -q "$z" && rm "$z"; done) || true
  fi
done

# PaySim: download the public log (494 MB, no account), verify it against the published statistics and
# its SHA-256, and build the column cache E3 reads. Pass an existing copy with PAYSIM_CSV=/path/to/file.csv
if [ ! -f data/paysim_cache/amount_minor.npy ]; then
  python=${PYTHON:-.venv/bin/python}
  if [ -n "${PAYSIM_CSV:-}" ]; then
    "$python" -m vista2.datasets.paysim --csv "$PAYSIM_CSV"
  else
    "$python" -m vista2.datasets.paysim
  fi
fi
