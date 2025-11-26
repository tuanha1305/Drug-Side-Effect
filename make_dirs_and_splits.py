import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os

# Create results and images directories if they don't exist
os.makedirs('results', exist_ok=True)
os.makedirs('images', exist_ok=True)

# Load data and exclude Ibuprofen
csv_path = 'data/drug_SMILES_750.csv'
df = pd.read_csv(csv_path, header=None, names=['drug', 'smiles'])
df = df[~df['drug'].str.lower().eq('ibuprofen')]
df_749_path = 'data/drug_SMILES_749_no_ibuprofen.csv'
df.to_csv(df_749_path, index=False, header=False)
print(f"Saved: {df_749_path}, rows={len(df)}")

# Generate 5-fold CV splits
n = len(df)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = [{'fold': int(i), 'train_idx': tr.tolist(), 'test_idx': te.tolist()} for i, (tr, te) in enumerate(kf.split(np.arange(n)))]

splits_path = 'cv_splits_5fold.json'
with open(splits_path, 'w', encoding='utf-8') as f:
    json.dump(splits, f, ensure_ascii=False, indent=2)
print(f'Saved {splits_path}')