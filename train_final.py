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
from baselines_cv import load_labels_750, align_labels_to_749

RESULTS_DIR = "results"
MODELS_DIR = "models"

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

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

def train_final_hstrans(model, train_loader, num_epochs=100, lr=0.0001, patience=15):
    """Train HSTrans final model trên toàn bộ 749 drugs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_drug, batch_drug_mask, batch_se, batch_se_mask, batch_labels in progress_bar:
            batch_drug = batch_drug.to(device)
            batch_drug_mask = batch_drug_mask.to(device)
            batch_se = batch_se.to(device)
            batch_se_mask = batch_se_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            scores, _, _ = model(batch_drug, batch_se, batch_drug_mask, batch_se_mask)
            scores = scores.squeeze(-1)  # Remove last dimension
            
            # Tính loss cho từng side effect
            loss = 0.0
            for i in range(scores.shape[1]):
                loss += criterion(scores[:, i], batch_labels[:, i])
            loss = loss / scores.shape[1]  # Average loss
            
                loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "hstrans_final_best.pth"))
            else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "hstrans_final_best.pth")))
    
    return model, train_losses

def evaluate_final_model(model, test_loader, device):
    """Evaluate final model"""
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
    
    # Calculate metrics
    y_pred_bin = (predictions >= 0.5).astype(int)
    
    metrics = {}
    with np.errstate(invalid='ignore'):
        metrics["f1_micro"] = f1_score(labels, y_pred_bin, average="micro", zero_division=0)
        metrics["f1_macro"] = f1_score(labels, y_pred_bin, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(labels, y_pred_bin, average="weighted", zero_division=0)
        metrics["auprc_micro"] = average_precision_score(labels, predictions, average="micro")
        metrics["auprc_macro"] = average_precision_score(labels, predictions, average="macro")
        metrics["auprc_weighted"] = average_precision_score(labels, predictions, average="weighted")
        metrics["auroc_micro"] = roc_auc_score(labels, predictions, average="micro")
        metrics["auroc_macro"] = roc_auc_score(labels, predictions, average="macro")
        metrics["auroc_weighted"] = roc_auc_score(labels, predictions, average="weighted")
    
    return metrics, predictions, labels

def main():
    print("=== Training Final HSTrans Model on 749 Drugs ===")
    ensure_dirs()
    
    # Load data
    print("Loading data...")
    df_749 = pd.read_csv("data/drug_SMILES_749_no_ibuprofen.csv", header=None, names=['drug', 'smiles'])
    df_750 = pd.read_csv("data/drug_SMILES_750.csv", header=None, names=['drug', 'smiles'])
    labels_750 = load_labels_750("data/side_effect_label_750.mat")
    
    # Align labels to 749 drugs
    labels_749 = align_labels_to_749(df_749, df_750, labels_750)
    print(f"Training data: {len(df_749)} drugs, {labels_749.shape[1]} side effects")
    
    # Prepare data
    smiles_list = df_749['smiles'].tolist()
    drug_tensor, drug_mask_tensor, se_tensor, se_mask_tensor, labels_tensor = prepare_hstrans_data(
        smiles_list, labels_749, batch_size=32
    )
    
    # Create data loader
    train_dataset = TensorDataset(drug_tensor, drug_mask_tensor, se_tensor, se_mask_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create and train model
    model = Trans()
    print("Training final HSTrans model...")
    
    trained_model, train_losses = train_final_hstrans(
        model, train_loader, num_epochs=100, lr=0.0001, patience=15
    )
    
    # Evaluate on training set (for reference)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating on training set...")
    metrics, predictions, labels = evaluate_final_model(trained_model, train_loader, device)
    
    print("\n=== Final Model Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save final model
    torch.save(trained_model.state_dict(), os.path.join(MODELS_DIR, "hstrans_final.pth"))
    print(f"\nFinal model saved to {MODELS_DIR}/hstrans_final.pth")
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'final_metrics': metrics,
        'num_drugs': len(df_749),
        'num_side_effects': labels_749.shape[1]
    }
    
    with open(os.path.join(RESULTS_DIR, "hstrans_final_training.json"), "w", encoding="utf-8") as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.title('HSTrans Final Model Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("images", "hstrans_final_training.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to {RESULTS_DIR}/hstrans_final_training.json")
    print(f"Training curve saved to images/hstrans_final_training.png")
    
    # Create drug-SE mapping for inference
    drug_se_mapping = {}
    for i, drug_name in enumerate(df_749['drug'].tolist()):
        drug_se_mapping[drug_name] = {
            'smiles': smiles_list[i],
            'side_effects': labels_749[i].tolist(),
            'predicted_probs': predictions[i].tolist()
        }
    
    with open(os.path.join(RESULTS_DIR, "drug_se_mapping_749.json"), "w", encoding="utf-8") as f:
        json.dump(drug_se_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Drug-SE mapping saved to {RESULTS_DIR}/drug_se_mapping_749.json")
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
