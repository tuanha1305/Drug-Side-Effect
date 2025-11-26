import os
import numpy as np
from sklearn.model_selection import KFold

def prepare_folds(k=5, n_samples=750, seed=42, outdir="folds"):
    """Tạo index cho các fold và lưu ra file .npy"""
    os.makedirs(outdir, exist_ok=True)
    drugs = np.arange(n_samples)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(drugs)):
        np.save(f"{outdir}/train_idx_{fold}.npy", train_idx)
        np.save(f"{outdir}/val_idx_{fold}.npy", val_idx)
    print(f"✅ Created {k}-fold split in {outdir}/")

def run_cv(k=5, epochs=100, patience=10, lr=1e-4):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("folds", exist_ok=True)

    # Nếu chưa có split thì tạo mới
    if not os.path.exists("folds/train_idx_0.npy"):
        prepare_folds(k=k)

    for fold in range(k):
        cmd = f"""
        python train.py \
            --train_idx folds/train_idx_{fold}.npy \
            --val_idx folds/val_idx_{fold}.npy \
            --epochs {epochs} \
            --patience {patience} \
            --lr {lr} \
            --save_path checkpoints/fold_{fold}.pt \
            --log_path logs/fold_{fold}.json
        """
        print(f"🚀 Running Fold {fold}: {cmd}")
        os.system(cmd)

if __name__ == "__main__":
    run_cv(k=5, epochs=100, patience=10, lr=1e-4)
