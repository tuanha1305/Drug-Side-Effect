import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from Net import Trans, drug2emb_encoder
from baselines_cv import load_labels_750, load_vocab_size, align_labels_to_749, evaluate_multi_label

RESULTS_DIR = "results"
IMAGES_DIR = "images"

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

def prepare_hstrans_data(smiles_list, labels, batch_size=32):
    """Chuẩn bị dữ liệu cho HSTrans model"""
    drug_sequences = []
    drug_masks = []
    se_sequences = []
    se_masks = []
    
    # Tạo SE sequences (dummy - sẽ được thay thế trong forward)
    dummy_se = np.zeros((50,), dtype=np.int32)
    dummy_se_mask = np.ones((50,), dtype=np.int32)
    
    for smile in smiles_list:
        # Encode drug
        drug_idx, drug_mask = drug2emb_encoder(smile)
        drug_sequences.append(drug_idx)
        drug_masks.append(drug_mask)
        
        # Dummy SE (sẽ được xử lý trong model)
        se_sequences.append(dummy_se)
        se_masks.append(dummy_se_mask)
    
    # Convert to tensors
    drug_tensor = torch.tensor(np.array(drug_sequences), dtype=torch.long)
    drug_mask_tensor = torch.tensor(np.array(drug_masks), dtype=torch.long)
    se_tensor = torch.tensor(np.array(se_sequences), dtype=torch.long)
    se_mask_tensor = torch.tensor(np.array(se_masks), dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return drug_tensor, drug_mask_tensor, se_tensor, se_mask_tensor, labels_tensor

def train_hstrans_fold(model, train_loader, val_loader, num_epochs=50, lr=0.0001, patience=10):
    """Train HSTrans cho một fold"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_drug, batch_drug_mask, batch_se, batch_se_mask, batch_labels in train_loader:
            batch_drug = batch_drug.to(device)
            batch_drug_mask = batch_drug_mask.to(device)
            batch_se = batch_se.to(device)
            batch_se_mask = batch_se_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - chỉ dùng drug input
            scores, _, _ = model(batch_drug, batch_se, batch_drug_mask, batch_se_mask)
            scores = scores.squeeze(-1)  # Remove last dimension
            
            # Tính loss cho từng side effect
            loss = 0.0
            for i in range(scores.shape[1]):
                loss += criterion(scores[:, i], batch_labels[:, i])
            loss = loss / scores.shape[1]  # Average loss
            
        loss.backward()
        optimizer.step()

            train_loss += loss.item()

        # Validation
    model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_drug, batch_drug_mask, batch_se, batch_se_mask, batch_labels in val_loader:
                batch_drug = batch_drug.to(device)
                batch_drug_mask = batch_drug_mask.to(device)
                batch_se = batch_se.to(device)
                batch_se_mask = batch_se_mask.to(device)
                batch_labels = batch_labels.to(device)
                
                scores, _, _ = model(batch_drug, batch_se, batch_drug_mask, batch_se_mask)
                scores = scores.squeeze(-1)
                
                loss = 0.0
                for i in range(scores.shape[1]):
                    loss += criterion(scores[:, i], batch_labels[:, i])
                loss = loss / scores.shape[1]
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses

def evaluate_hstrans(model, test_loader, device):
    """Evaluate HSTrans model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_drug, batch_drug_mask, batch_se, batch_se_mask, batch_labels in test_loader:
            batch_drug = batch_drug.to(device)
            batch_drug_mask = batch_drug_mask.to(device)
            batch_se = batch_se.to(device)
            batch_se_mask = batch_se_mask.to(device)
            
            scores, _, _ = model(batch_drug, batch_se, batch_drug_mask, batch_se_mask)
            scores = scores.squeeze(-1)
            
            # Convert to probabilities
            probs = torch.sigmoid(scores).cpu().numpy()
            all_predictions.append(probs)
            all_labels.append(batch_labels.numpy())
    
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    
    return evaluate_multi_label(labels, predictions)

def main():
    print("=== HSTrans 5-Fold Cross Validation ===")
    ensure_dirs()
    
    # Load data
    print("Loading data...")
    df_749 = pd.read_csv("data/drug_SMILES_749_no_ibuprofen.csv", header=None, names=['drug', 'smiles'])
    df_750 = pd.read_csv("data/drug_SMILES_750.csv", header=None, names=['drug', 'smiles'])
    labels_750 = load_labels_750("data/side_effect_label_750.mat")
    
    # Align labels to 749 drugs
    labels_749 = align_labels_to_749(df_749, df_750, labels_750)
    print(f"Labels shape: {labels_749.shape}")
    
    # Load CV splits
    with open("cv_splits_5fold.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    # Results storage
    fold_results = []
    
    for fold_idx, split in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1}/5 ===")
        
        # Get train/test indices
        train_idx = split['train_idx']
        test_idx = split['test_idx']
        
        # Split data
        train_smiles = df_749.iloc[train_idx]['smiles'].tolist()
        test_smiles = df_749.iloc[test_idx]['smiles'].tolist()
        train_labels = labels_749[train_idx]
        test_labels = labels_749[test_idx]
        
        # Prepare data
        train_drug, train_drug_mask, train_se, train_se_mask, train_labels_tensor = prepare_hstrans_data(
            train_smiles, train_labels, batch_size=16
        )
        test_drug, test_drug_mask, test_se, test_se_mask, test_labels_tensor = prepare_hstrans_data(
            test_smiles, test_labels, batch_size=16
        )
        
        # Create data loaders
        train_dataset = TensorDataset(train_drug, train_drug_mask, train_se, train_se_mask, train_labels_tensor)
        test_dataset = TensorDataset(test_drug, test_drug_mask, test_se, test_se_mask, test_labels_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Create and train model
        model = Trans()
        print(f"Training HSTrans on {len(train_smiles)} samples...")
        
        trained_model, train_losses, val_losses = train_hstrans_fold(
            model, train_loader, test_loader, num_epochs=50, lr=0.0001, patience=10
        )
        
        # Evaluate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        metrics = evaluate_hstrans(trained_model, test_loader, device)
        
        print(f"Fold {fold_idx + 1} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
    
    # Calculate average metrics
    print("\n=== Average Results ===")
    avg_metrics = {}
    for metric in fold_results[0]['metrics'].keys():
        values = [result['metrics'][metric] for result in fold_results]
        avg_metrics[metric] = np.mean(values)
        std_metrics = np.std(values)
        print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics:.4f}")
    
    # Save results
    results = {
        'model': 'HSTrans',
        'cv_folds': 5,
        'avg_metrics': avg_metrics,
        'fold_results': fold_results
    }
    
    with open(os.path.join(RESULTS_DIR, "hstrans_cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for i, result in enumerate(fold_results):
        plt.plot(result['train_losses'], label=f'Fold {i+1}', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i, result in enumerate(fold_results):
        plt.plot(result['val_losses'], label=f'Fold {i+1}', alpha=0.7)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "hstrans_cv_training.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {RESULTS_DIR}/hstrans_cv_results.json")
    print(f"Training curves saved to {IMAGES_DIR}/hstrans_cv_training.png")

if __name__ == "__main__":
    main()
