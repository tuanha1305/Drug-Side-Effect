#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script dự đoán tác dụng phụ của Ibuprofen
Sử dụng model đã train từ main.py và lưu kết quả đầy đủ
"""

import numpy as np
import pandas as pd
import torch
import json
import os
import argparse
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None
    print("⚠ Thiếu thư viện 'seaborn' – sẽ bỏ qua bước vẽ heatmap. Cài đặt bằng: pip install seaborn")
from Net import drug2emb_encoder, Trans
from utils import mse, rmse, spearman
from scipy.io import loadmat

def load_ibuprofen_data():
    """Load dữ liệu Ibuprofen và side effects"""
    print("=== Loading Ibuprofen Data ===")
    
    # Load SMILES của Ibuprofen
    df = pd.read_csv('data/drug_SMILES_750.csv', header=None, names=['drug', 'smiles'])
    ibuprofen_row = df[df['drug'].str.lower() == 'ibuprofen']
    if len(ibuprofen_row) == 0:
        print("Không tìm thấy Ibuprofen trong dữ liệu!")
        return None, None, None, None
    
    ibuprofen_smiles = ibuprofen_row.iloc[0]['smiles']
    print(f"✓ SMILES của Ibuprofen: {ibuprofen_smiles}")
    
    # Load substructure data
    try:
        SE_index = np.load("data/sub/SE_sub_index_50_32.npy").astype(int)
        SE_mask = np.load("data/sub/SE_sub_mask_50_32.npy")
        print(f"✓ Loaded substructure data: {SE_index.shape}")
    except FileNotFoundError as e:
        print(f"✗ Không tìm thấy file substructure: {e}")
        return None, None, None, None
    
    # Load side effect names
    try:
        side_effect_data = loadmat('data/raw_frequency_750.mat')
        side_effects = side_effect_data['sideeffects'].flatten()
        print(f"✓ Loaded {len(side_effects)} side effect names")
    except Exception as e:
        print(f"⚠ Không load được tên side effects: {e}")
        side_effects = [f"SE_{i}" for i in range(994)]
    
    # Load ground truth labels (nếu có)
    try:
        labels_data = loadmat('data/raw_frequency_750.mat')
        R = labels_data['R']  # (750, 994)
        # Tìm Ibuprofen trong danh sách 750 drugs
        drugs = labels_data['drugs'].flatten()
        ibuprofen_idx = None
        for i, drug in enumerate(drugs):
            if 'ibuprofen' in str(drug).lower():
                ibuprofen_idx = i
                break
        
        if ibuprofen_idx is not None:
            ground_truth = R[ibuprofen_idx] > 0  # Binary labels
            print(f"✓ Loaded ground truth for Ibuprofen: {np.sum(ground_truth)} positive effects")
        else:
            ground_truth = None
            print("⚠ Không tìm thấy ground truth cho Ibuprofen")
    except Exception as e:
        print(f"⚠ Không load được ground truth: {e}")
        ground_truth = None
    
    return ibuprofen_smiles, SE_index, SE_mask, side_effects, ground_truth

def load_trained_model(checkpoint_path: str = None):
    """Load model đã train. Nếu chỉ định checkpoint_path thì load đúng file đó."""
    print("\n=== Loading Trained Model ===")
    
    device = torch.device('cpu')
    model = Trans().to(device)
    
    # Nếu có chỉ định checkpoint cụ thể
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✓ Loaded trained model from {checkpoint_path}")
                return model, device
            except Exception as e:
                print(f"⚠ Không load được model từ {checkpoint_path}: {e}")
        else:
            print(f"⚠ Checkpoint không tồn tại: {checkpoint_path}")

    # Tìm checkpoint mới nhất
    checkpoint_dir = 'checkpoints'
    if os.path.exists(checkpoint_dir):
        # Tìm file checkpoint (có thể là .pth hoặc không có extension)
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pth') or f.isdigit():
                checkpoints.append(f)
        
        if checkpoints:
            # Sắp xếp theo số epoch (0, 10, 20, 50, ...)
            def extract_epoch(filename):
                try:
                    return int(filename)
                except:
                    return 0
            latest_checkpoint = sorted(checkpoints, key=extract_epoch)[-1]
            model_path = os.path.join(checkpoint_dir, latest_checkpoint)
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✓ Loaded trained model from {model_path} (new format)")
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"✓ Loaded trained model from {model_path} (legacy format)")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"✓ Loaded trained model from {model_path} (weights only)")
                return model, device
            except Exception as e:
                print(f"⚠ Không load được model từ checkpoint: {e}")
                return None, None
    
    # Nếu không có checkpoint, dừng lại
    print("❌ KHÔNG TÌM THẤY CHECKPOINT!")
    print("   Vui lòng chạy training trước:")
    print("   python main.py --save_model --epoch 50 --lr 0.0001")
    print("   Hoặc chờ training hoàn thành...")
    return None, None

def predict_ibuprofen_side_effects(model, device, SE_index, SE_mask, side_effects):
    """Dự đoán tác dụng phụ của Ibuprofen"""
    print("\n=== Predicting Side Effects ===")
    
    # Encode Ibuprofen SMILES
    drug_emb, drug_mask = drug2emb_encoder("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    print(f"✓ Drug embedding shape: {drug_emb.shape}")
    
    model.eval()  # Đảm bảo model ở chế độ evaluation
    predictions = []
    
    print("Đang dự đoán cho 994 side effects...")
    with torch.no_grad():
        for se_id in range(994):
            # Chuẩn bị input
            se_emb = torch.tensor(SE_index[se_id], dtype=torch.long)
            se_mask = torch.tensor(SE_mask[se_id], dtype=torch.float32)
            drug_emb_tensor = torch.tensor(drug_emb, dtype=torch.long).unsqueeze(0)
            drug_mask_tensor = torch.tensor(drug_mask, dtype=torch.float32).unsqueeze(0)
            se_emb_tensor = se_emb.unsqueeze(0)
            se_mask_tensor = se_mask.unsqueeze(0)
            
            # Dự đoán
            output, _, _ = model(drug_emb_tensor, se_emb_tensor, drug_mask_tensor, se_mask_tensor)
            pred_score = output.item()
            predictions.append(pred_score)
    
    predictions = np.array(predictions)
    print(f"✓ Completed predictions for {len(predictions)} side effects")
    
    return predictions

def predict_with_checkpoints_ensemble(checkpoint_paths, SE_index, SE_mask):
    """Trung bình dự đoán qua nhiều checkpoints (CPU)."""
    device = torch.device('cpu')
    all_preds = []
    for ckpt in checkpoint_paths:
        model, _ = load_trained_model(ckpt)
        if model is None:
            continue
        model.eval()  # Đảm bảo model ở chế độ evaluation
        preds = predict_ibuprofen_side_effects(model, device, SE_index, SE_mask, None)
        all_preds.append(preds)
    if not all_preds:
        return None
    return np.mean(np.vstack(all_preds), axis=0)

def analyze_predictions(predictions, side_effects, ground_truth=None):
    """Phân tích và tạo báo cáo kết quả"""
    print("\n=== Analyzing Results ===")
    
    # Top predictions
    top_indices = np.argsort(predictions)[::-1]
    
    # Tạo báo cáo chi tiết
    results = []
    for i, idx in enumerate(top_indices):
        se_name = side_effects[idx] if idx < len(side_effects) else f"SE_{idx}"
        score = predictions[idx]
        
        # So sánh với ground truth nếu có
        is_ground_truth = ground_truth[idx] if ground_truth is not None else None
        
        results.append({
            'rank': i + 1,
            'side_effect_id': int(idx),
            'side_effect_name': str(se_name),
            'prediction_score': float(score),
            'is_ground_truth': bool(is_ground_truth) if is_ground_truth is not None else None
        })
    
    # Thống kê
    stats = {
        'total_side_effects': len(predictions),
        'mean_score': float(np.mean(predictions)),
        'max_score': float(np.max(predictions)),
        'min_score': float(np.min(predictions)),
        'std_score': float(np.std(predictions)),
        'high_confidence_count': int(np.sum(predictions > 0.7)),
        'medium_confidence_count': int(np.sum((predictions > 0.5) & (predictions <= 0.7))),
        'low_confidence_count': int(np.sum(predictions <= 0.5))
    }
    
    # Tính overlap với ground truth nếu có
    if ground_truth is not None:
        top_20_pred = set(top_indices[:20])
        top_50_pred = set(top_indices[:50])
        top_100_pred = set(top_indices[:100])
        
        ground_truth_pos = set(np.where(ground_truth)[0])
        total_gt = len(ground_truth_pos)
        
        def prf_at_k(top_set, k):
            tp = len(top_set & ground_truth_pos)
            precision = tp / k if k > 0 else 0.0
            recall = tp / total_gt if total_gt > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return tp, precision, recall, f1

        tp20, p20, r20, f20 = prf_at_k(top_20_pred, 20)
        tp50, p50, r50, f50 = prf_at_k(top_50_pred, 50)
        tp100, p100, r100, f100 = prf_at_k(top_100_pred, 100)

        overlap_stats = {
            'total_ground_truth': total_gt,
            'overlap_at_20': tp20,
            'precision_at_20': p20,
            'recall_at_20': r20,
            'f1_at_20': f20,
            'overlap_at_50': tp50,
            'precision_at_50': p50,
            'recall_at_50': r50,
            'f1_at_50': f50,
            'overlap_at_100': tp100,
            'precision_at_100': p100,
            'recall_at_100': r100,
            'f1_at_100': f100
        }
        stats.update(overlap_stats)
    
    return results, stats

def save_results(results, stats, predictions, side_effects):
    """Lưu kết quả vào các file khác nhau"""
    print("\n=== Saving Results ===")
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # 1. JSON - Kết quả chi tiết
    with open('results/ibuprofen_detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'drug': 'Ibuprofen',
            'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'statistics': stats,
            'all_predictions': results
        }, f, ensure_ascii=False, indent=2)
    print("✓ Saved detailed results to results/ibuprofen_detailed_results.json")
    
    # 2. CSV - Top predictions
    top_50_df = pd.DataFrame(results[:50])
    top_50_df.to_csv('results/ibuprofen_top50_predictions.csv', index=False, encoding='utf-8')
    print("✓ Saved top 50 predictions to results/ibuprofen_top50_predictions.csv")
    
    # 3. CSV - Tất cả predictions
    all_predictions_df = pd.DataFrame({
        'side_effect_id': range(len(predictions)),
        'side_effect_name': [side_effects[i] if i < len(side_effects) else f"SE_{i}" for i in range(len(predictions))],
        'prediction_score': predictions
    })
    all_predictions_df = all_predictions_df.sort_values('prediction_score', ascending=False)
    all_predictions_df.to_csv('results/ibuprofen_all_predictions.csv', index=False, encoding='utf-8')
    print("✓ Saved all predictions to results/ibuprofen_all_predictions.csv")
    
    # 4. Tạo biểu đồ
    create_visualizations(predictions, results, stats)
    
    return True

def save_ground_truth_ranking(results, ground_truth, output_path='results/ibuprofen_ground_truth_ranking.csv'):
    """Xuất bảng xếp hạng đầy đủ cho toàn bộ ground-truth của Ibuprofen.
    Gồm: rank, side_effect_id, side_effect_name, prediction_score, in_top50.
    """
    if ground_truth is None:
        return False
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df_gt = df[df['is_ground_truth'] == True].copy()
    if df_gt.empty:
        # Trường hợp không có cờ GT (hoặc không khớp), suy luận theo mask ground_truth
        # Map theo side_effect_id
        df['is_ground_truth_from_mask'] = df['side_effect_id'].map(lambda j: bool(ground_truth[int(j)]))
        df_gt = df[df['is_ground_truth_from_mask'] == True].copy()
    # Đánh dấu nằm trong Top-50
    df_gt['in_top50'] = df_gt['rank'] <= 50
    # Sắp xếp theo rank tăng dần
    df_gt = df_gt.sort_values('rank')
    df_gt[['rank','side_effect_id','side_effect_name','prediction_score','in_top50']].to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Saved ground-truth full ranking to {output_path}")
    return True

def save_topk_csv(results, k=20, path='results/ibuprofen_top20_with_gt.csv'):
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results[:k])
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"✓ Saved Top-{k} with ground-truth flags to {path}")

def save_interaction_heatmap(model, se_id, SE_index, SE_mask, output_path='images/ibuprofen_substructure_se_heatmap.png'):
    """Tạo và lưu heatmap tương tác (50x50) giữa substructures thuốc và side effect se_id"""
    os.makedirs('images', exist_ok=True)
    drug_emb, drug_mask = drug2emb_encoder("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")

    with torch.no_grad():
        drug_emb_tensor = torch.tensor(drug_emb, dtype=torch.long).unsqueeze(0)
        drug_mask_tensor = torch.tensor(drug_mask, dtype=torch.float32).unsqueeze(0)
        se_emb_tensor = torch.tensor(SE_index[se_id], dtype=torch.long).unsqueeze(0)
        se_mask_tensor = torch.tensor(SE_mask[se_id], dtype=torch.float32).unsqueeze(0)

        # Dùng hàm forward trả về interaction map
        if hasattr(model, 'forward_with_interaction'):
            score, interaction_map = model.forward_with_interaction(
                drug_emb_tensor, se_emb_tensor, drug_mask_tensor, se_mask_tensor
            )
            interaction = interaction_map.squeeze(0).cpu().numpy()  # (50, 50)
        else:
            # Fallback: không có interaction map
            return False

    if sns is None:
        print("⚠ seaborn chưa được cài. Bỏ qua vẽ heatmap. Chạy: pip install seaborn")
        return False

    plt.figure(figsize=(8, 6))
    sns.heatmap(interaction, cmap='viridis')
    plt.title(f'Interaction Heatmap (Ibuprofen vs SE {se_id})')
    plt.xlabel('Side-effect substructures')
    plt.ylabel('Drug substructures')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved interaction heatmap to {output_path}")
    return True

def create_visualizations(predictions, results, stats):
    """Tạo các biểu đồ trực quan"""
    print("Creating visualizations...")
    
    # 1. Top 20 predictions bar chart
    plt.figure(figsize=(12, 8))
    top_20 = results[:20]
    names = [r['side_effect_name'][:30] + '...' if len(r['side_effect_name']) > 30 else r['side_effect_name'] for r in top_20]
    scores = [r['prediction_score'] for r in top_20]
    
    plt.barh(range(len(names)), scores, color='skyblue')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Prediction Score')
    plt.title('Top 20 Predicted Side Effects for Ibuprofen')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('images/ibuprofen_top20_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved top 20 bar chart to images/ibuprofen_top20_predictions.png")
    
    # 2. Score distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(stats['mean_score'], color='red', linestyle='--', label=f'Mean: {stats["mean_score"]:.3f}')
    plt.axvline(0.5, color='orange', linestyle='--', label='Threshold: 0.5')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores for Ibuprofen')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/ibuprofen_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved score distribution to images/ibuprofen_score_distribution.png")
    
    # 3. Confidence levels pie chart
    plt.figure(figsize=(8, 8))
    labels = ['High (>0.7)', 'Medium (0.5-0.7)', 'Low (≤0.5)']
    sizes = [stats['high_confidence_count'], stats['medium_confidence_count'], stats['low_confidence_count']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Confidence Levels of Predictions')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('images/ibuprofen_confidence_levels.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved confidence levels to images/ibuprofen_confidence_levels.png")

def print_summary(results, stats):
    """In tóm tắt kết quả"""
    print("\n" + "="*60)
    print("                    KẾT QUẢ DỰ ĐOÁN IBUPROFEN")
    print("="*60)
    
    print(f"\n📊 THỐNG KÊ TỔNG QUAN:")
    print(f"   • Tổng số side effects: {stats['total_side_effects']}")
    print(f"   • Score trung bình: {stats['mean_score']:.4f}")
    print(f"   • Score cao nhất: {stats['max_score']:.4f}")
    print(f"   • Score thấp nhất: {stats['min_score']:.4f}")
    print(f"   • Độ lệch chuẩn: {stats['std_score']:.4f}")
    
    print(f"\n🎯 MỨC ĐỘ TIN CẬY:")
    print(f"   • Cao (>0.7): {stats['high_confidence_count']} effects")
    print(f"   • Trung bình (0.5-0.7): {stats['medium_confidence_count']} effects")
    print(f"   • Thấp (≤0.5): {stats['low_confidence_count']} effects")
    
    if 'overlap_at_20' in stats:
        print(f"\n✅ SO SÁNH VỚI GROUND TRUTH:")
        print(f"   • Overlap @ Top-20: {stats['overlap_at_20']}/{20} ({stats['precision_at_20']:.1%})")
        print(f"   • Overlap @ Top-50: {stats['overlap_at_50']}/{50} ({stats['precision_at_50']:.1%})")
        print(f"   • Overlap @ Top-100: {stats['overlap_at_100']}/{100} ({stats['precision_at_100']:.1%})")
        print(f"   • Tổng ground truth: {stats['total_ground_truth']} effects")
    
    print(f"\n🏆 TOP 10 TÁC DỤNG PHỤ DỰ ĐOÁN:")
    for i, result in enumerate(results[:10]):
        gt_mark = " ✓" if result.get('is_ground_truth') else ""
        print(f"   {i+1:2d}. {result['side_effect_name']:<30} - {result['prediction_score']:.4f}{gt_mark}")
    
    print(f"\n💾 FILES ĐÃ LƯU:")
    print(f"   • results/ibuprofen_detailed_results.json")
    print(f"   • results/ibuprofen_top50_predictions.csv")
    print(f"   • results/ibuprofen_all_predictions.csv")
    print(f"   • images/ibuprofen_top20_predictions.png")
    print(f"   • images/ibuprofen_score_distribution.png")
    print(f"   • images/ibuprofen_confidence_levels.png")
    
    print("="*60)

def main():
    """Hàm chính"""
    print("🚀 BẮT ĐẦU DỰ ĐOÁN TÁC DỤNG PHỤ IBUPROFEN")
    print("="*60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Đường dẫn checkpoint cụ thể để load model')
    parser.add_argument('--checkpoints', type=str, nargs='*', default=None, help='Danh sách nhiều checkpoints để ensemble')
    args = parser.parse_args()

    # 1. Load dữ liệu
    data = load_ibuprofen_data()
    if data[0] is None:
        return
    ibuprofen_smiles, SE_index, SE_mask, side_effects, ground_truth = data
    
    # 2–3. Dự đoán (đơn lẻ hoặc ensemble)
    if args.checkpoints:
        predictions = predict_with_checkpoints_ensemble(args.checkpoints, SE_index, SE_mask)
        if predictions is None:
            print("❌ Ensemble thất bại do không load được checkpoints")
            return
        model, device = load_trained_model(args.checkpoints[-1])
        if model is not None:
            model.eval()  # Đảm bảo model ở chế độ evaluation
    else:
        model, device = load_trained_model(args.checkpoint)
        if model is None:
            print("❌ Không thể tiếp tục vì không có model đã train!")
            return
        model.eval()  # Đảm bảo model ở chế độ evaluation
        predictions = predict_ibuprofen_side_effects(model, device, SE_index, SE_mask, side_effects)
    
    # 4. Phân tích kết quả
    results, stats = analyze_predictions(predictions, side_effects, ground_truth)
    
    # 5. Lưu kết quả
    save_results(results, stats, predictions, side_effects)
    # 5a. Xuất bảng xếp hạng đầy đủ cho 35 ground-truth
    save_ground_truth_ranking(results, ground_truth, output_path='results/ibuprofen_ground_truth_ranking.csv')
    
    # 5b. Heatmap cho top-1 side effect
    top1_id = int(np.argsort(predictions)[::-1][0])
    save_interaction_heatmap(model, top1_id, SE_index, SE_mask, output_path='images/ibuprofen_substructure_se_heatmap.png')
    
    # 6. In tóm tắt
    print_summary(results, stats)
    
    print("\n✅ HOÀN THÀNH DỰ ĐOÁN IBUPROFEN!")

if __name__ == "__main__":
    main()