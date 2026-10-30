#!/usr/bin/env bash
# Download the public datasets into data/. Needs the Kaggle CLI with an API token, and for IEEE-CIS the
# competition rules accepted on kaggle.com by the same account.
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

# PaySim column cache from a VISTA v1 checkout (set VISTA_V1 to its path)
if [ ! -f data/paysim_cache/amount_minor.npy ] && [ -n "${VISTA_V1:-}" ]; then
  cp "$VISTA_V1"/data/paysim_cache/* data/paysim_cache/
fi
