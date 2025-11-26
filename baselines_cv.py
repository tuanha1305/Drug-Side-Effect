import json
import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy.io import loadmat

from Net import drug2emb_encoder


RESULTS_DIR = "results"
IMAGES_DIR = "images"


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_vocab_size(csv_path: str) -> int:
    import numpy as np
    import pandas as pd

    # đọc mà không phụ thuộc header
    df = pd.read_csv(csv_path, header=None)

    # nếu có cột tên 'index' và là số
    if 'index' in df.columns:
        s = pd.to_numeric(df['index'], errors='coerce')
        if s.notna().sum() > 0:
            return int(s.max()) + 1

    # nếu cột đầu là số
    s0 = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    if s0.notna().sum() > 0:
        return int(s0.max()) + 1

    # fallback: dùng số dòng làm vocab_size
    return int(len(df))


def load_labels_750(mat_path: str) -> np.ndarray:
    from scipy.io import loadmat
    import numpy as np

    m = loadmat('data/raw_frequency_750.mat', squeeze_me=True, struct_as_record=False)
    R = m.get('R', None)
    if R is None:
        raise RuntimeError("Không tìm thấy biến 'R' trong data/raw_frequency_750.mat")
    R = np.array(R)
    # Đưa về (750, num_se)
    if R.ndim > 2:
        R = np.squeeze(R)
    if R.shape[0] != 750 and R.shape[1] == 750:
        R = R.T
    if R.shape[0] != 750:
        raise RuntimeError(f"Kích thước R không hợp lệ: {R.shape}, cần (750, num_se)")
    return (R > 0).astype(np.int32)

def build_bpe_bag_features(smiles: List[str], vocab_size: int) -> np.ndarray:
    X = np.zeros((len(smiles), vocab_size), dtype=np.float32)
    for i, s in enumerate(smiles):
        idxs, mask = drug2emb_encoder(s)
        # đếm tần suất các chỉ số subword trong câu (bỏ padding=0)
        for t in idxs.tolist():
            if t <= 0 or t >= vocab_size:
                continue
            X[i, t] += 1.0
        # chuẩn hóa theo độ dài có thể giúp ổn định
        length = float(mask.sum())
        if length > 0:
            X[i] /= length
    return X




def align_labels_to_749(df749: pd.DataFrame, df750: pd.DataFrame, labels750: np.ndarray) -> np.ndarray:
    # map tên thuốc ở 750 -> row index
    name_to_row: Dict[str, int] = {}
    for i, name in enumerate(df750.iloc[:, 0].astype(str).tolist()):
        name_to_row[name.strip()] = i

    aligned = []
    for name in df749.iloc[:, 0].astype(str).tolist():
        idx = name_to_row.get(name.strip(), None)
        if idx is None:
            raise RuntimeError(f"Thuốc '{name}' trong 749 không khớp với danh sách 750 để lấy nhãn")
        aligned.append(labels750[idx])
    Y = np.stack(aligned, axis=0)
    # đảm bảo nhị phân (0/1)
    Y = (Y > 0).astype(np.int32)
    return Y


def evaluate_multi_label(y_true: np.ndarray, y_pred_scores: np.ndarray, decision: float = 0.5) -> Dict[str, float]:
    y_pred_bin = (y_pred_scores >= decision).astype(int)
    metrics: Dict[str, float] = {}
    # an toàn khi có lớp rỗng
    with np.errstate(invalid='ignore'):
        metrics["f1_micro"] = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    try:
        metrics["ap_micro"] = average_precision_score(y_true, y_pred_scores, average="micro")
    except Exception:
        metrics["ap_micro"] = float("nan")
    try:
        metrics["roc_auc_micro"] = roc_auc_score(y_true, y_pred_scores, average="micro")
    except Exception:
        metrics["roc_auc_micro"] = float("nan")
    return metrics


def run_fold_models(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, Y_te: np.ndarray) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=200)))
        ]),
        "RF": OneVsRestClassifier(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
        "MLP": OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, random_state=42)),
    }

    # LinearSVC không cho predict_proba; dùng decision_function rồi chuẩn hóa sigmoid gần đúng
    models_svm = OneVsRestClassifier(LinearSVC())

    for name, model in models.items():
        model.fit(X_tr, Y_tr)
        try:
            scores = model.predict_proba(X_te)
        except Exception:
            # fallback nếu không hỗ trợ predict_proba
            scores = model.decision_function(X_te)
            scores = 1 / (1 + np.exp(-scores))
        results[name] = evaluate_multi_label(Y_te, scores)

    # SVM tuyến tính
    models_svm.fit(X_tr, Y_tr)
    scores = models_svm.decision_function(X_te)
    scores = 1 / (1 + np.exp(-scores))
    results["LinearSVC"] = evaluate_multi_label(Y_te, scores)

    return results


def aggregate_results(per_fold: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    # gom theo mô hình -> metric -> list
    agg: Dict[str, Dict[str, List[float]]] = {}
    for fold_res in per_fold:
        for model_name, metrics in fold_res.items():
            agg.setdefault(model_name, {})
            for m, v in metrics.items():
                agg[model_name].setdefault(m, []).append(float(v))

    out: Dict[str, Dict[str, float]] = {}
    for model_name, md in agg.items():
        out.setdefault(model_name, {})
        for m, arr in md.items():
            arr_np = np.asarray(arr, dtype=float)
            out[model_name][f"{m}_mean"] = float(np.nanmean(arr_np))
            out[model_name][f"{m}_std"] = float(np.nanstd(arr_np))
    return out


def save_bar_chart(summary: Dict[str, Dict[str, float]], out_path: str) -> None:
    labels = list(summary.keys())
    f1_micro = [summary[m]["f1_micro_mean"] for m in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, f1_micro, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    plt.ylabel("F1-micro (mean)")
    plt.title("Baselines 5-fold (BPE-bag features)")
    for i, v in enumerate(f1_micro):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Chỉ chạy 1 fold cụ thể (0-4). Mặc định: chạy đủ 5 fold")
    args = parser.parse_args()
    ensure_dirs()

    # dữ liệu
    df_749 = pd.read_csv("data/drug_SMILES_749_no_ibuprofen.csv", header=None)
    df_750 = pd.read_csv("data/drug_SMILES_750.csv", header=None)
    labels_750 = load_labels_750("data/side_effect_label_750.mat")

    # căn hàng nhãn theo thứ tự 749
    Y = align_labels_to_749(df_749, df_750, labels_750)

    # đặc trưng BPE-bag
    vocab_size = load_vocab_size("data/subword_units_map_chembl_freq_1500.csv")
    smiles = df_749.iloc[:, 1].astype(str).tolist()
    X = build_bpe_bag_features(smiles, vocab_size)

    # CV splits
    splits = json.load(open("cv_splits_5fold.json", "r", encoding="utf-8"))

    per_fold_results: List[Dict[str, Dict[str, float]]] = []
    if args.fold is not None:
        # chạy đúng 1 fold
        target = None
        for sp in splits:
            if int(sp.get("fold", -1)) == int(args.fold):
                target = sp
                break
        if target is None:
            raise ValueError(f"Không tìm thấy fold={args.fold} trong cv_splits_5fold.json")

        tr = np.asarray(target["train_idx"], dtype=int)
        te = np.asarray(target["test_idx"], dtype=int)
        X_tr, Y_tr = X[tr], Y[tr]
        X_te, Y_te = X[te], Y[te]
        fold_res = run_fold_models(X_tr, Y_tr, X_te, Y_te)
        per_fold_results.append(fold_res)

        # lưu riêng cho fold
        out_json = os.path.join(RESULTS_DIR, f"baselines_cv_fold{args.fold}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"fold": int(args.fold), "results": fold_res}, f, ensure_ascii=False, indent=2)
        print("Saved:", out_json)
    else:
        # chạy đủ 5 fold
        for sp in splits:
            tr = np.asarray(sp["train_idx"], dtype=int)
            te = np.asarray(sp["test_idx"], dtype=int)

            X_tr, Y_tr = X[tr], Y[tr]
            X_te, Y_te = X[te], Y[te]

            fold_res = run_fold_models(X_tr, Y_tr, X_te, Y_te)
            per_fold_results.append(fold_res)

        summary = aggregate_results(per_fold_results)

        # lưu kết quả tổng
        with open(os.path.join(RESULTS_DIR, "baselines_cv.json"), "w", encoding="utf-8") as f:
            json.dump({"per_fold": per_fold_results, "summary": summary}, f, ensure_ascii=False, indent=2)

        save_bar_chart(summary, os.path.join(IMAGES_DIR, "baselines_cv.png"))
        print("Saved:", os.path.join(RESULTS_DIR, "baselines_cv.json"))
        print("Saved:", os.path.join(IMAGES_DIR, "baselines_cv.png"))


if __name__ == "__main__":
    main()


