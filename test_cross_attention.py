"""
Script test để kiểm tra khối Cross-Attention và Residual Fusion
Kiểm tra:
1. Model có thể khởi tạo được không
2. Forward pass hoạt động đúng không
3. Shape của các tensor
4. Gradient flow
5. Test với dữ liệu giả
"""

import torch
import torch.nn as nn
import numpy as np
from Net import Trans

def test_model_initialization():
    """Test 1: Kiểm tra model có thể khởi tạo được không"""
    print("=" * 60)
    print("TEST 1: Kiểm tra khởi tạo model")
    print("=" * 60)
    
    try:
        model = Trans()
        print("✓ Model khoi tao thanh cong")
        
        # Đếm số tham số
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Tong so tham so: {total_params:,}")
        print(f"✓ Tham so co the train: {trainable_params:,}")
        
        # Kiểm tra các layer mới
        print("\n✓ Kiem tra cac layer moi:")
        print(f"  - residual_fusion_drug: {model.residual_fusion_drug is not None}")
        print(f"  - residual_fusion_side: {model.residual_fusion_side is not None}")
        print(f"  - gate_drug: {model.gate_drug is not None}")
        print(f"  - gate_side: {model.gate_side is not None}")
        print(f"  - crossAttentionencoder: {model.crossAttentionencoder is not None}")
        print(f"  - CrossAttention flag: {model.CrossAttention}")
        
        return True, model
    except Exception as e:
        print(f"✗ Loi khi khoi tao model: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, device='cpu'):
    """Test 2: Kiểm tra forward pass"""
    print("\n" + "=" * 60)
    print("TEST 2: Kiểm tra forward pass")
    print("=" * 60)
    
    try:
        model.eval()
        batch_size = 4
        seq_len = 50
        
        # Tạo dữ liệu giả
        Drug = torch.randint(0, 100, (batch_size, seq_len)).long()
        SE = torch.randint(0, 100, (batch_size, seq_len)).long()
        DrugMask = torch.ones((batch_size, seq_len)).long()
        SEMask = torch.ones((batch_size, seq_len)).long()
        
        print(f"✓ Input shapes:")
        print(f"  - Drug: {Drug.shape}")
        print(f"  - SE: {SE.shape}")
        print(f"  - DrugMask: {DrugMask.shape}")
        print(f"  - SEMask: {SEMask.shape}")
        
        # Forward pass
        with torch.no_grad():
            output, drug_out, se_out = model(Drug, SE, DrugMask, SEMask)
        
        print(f"\n✓ Output shapes:")
        print(f"  - output (score): {output.shape}")
        print(f"  - drug_out: {drug_out.shape}")
        print(f"  - se_out: {se_out.shape}")
        
        # Kiểm tra output hợp lệ
        assert output.shape[0] == batch_size, f"Batch size khong khop: {output.shape[0]} != {batch_size}"
        assert output.shape[1] == 1, f"Output dimension khong dung: {output.shape[1]} != 1"
        assert not torch.isnan(output).any(), "Output chua NaN!"
        assert not torch.isinf(output).any(), "Output chua Inf!"
        
        print("✓ Forward pass thanh cong, khong co NaN/Inf")
        return True
    except Exception as e:
        print(f"✗ Loi trong forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_attention_output_shapes(model, device='cpu'):
    """Test 3: Kiểm tra shape của các tensor trong cross-attention"""
    print("\n" + "=" * 60)
    print("TEST 3: Kiểm tra shape của cross-attention và residual fusion")
    print("=" * 60)
    
    try:
        model.eval()
        batch_size = 2
        seq_len = 50
        hidden_size = 200
        
        Drug = torch.randint(0, 100, (batch_size, seq_len)).long()
        SE = torch.randint(0, 100, (batch_size, seq_len)).long()
        DrugMask = torch.ones((batch_size, seq_len)).long()
        SEMask = torch.ones((batch_size, seq_len)).long()
        
        # Forward đến trước cross-attention
        Drug = Drug.to(device)
        SE = SE.to(device)
        DrugMask = DrugMask.to(device)
        SEMask = SEMask.to(device)
        
        DrugMask_expanded = DrugMask.unsqueeze(1).unsqueeze(2)
        DrugMask_expanded = (1.0 - DrugMask_expanded.float()) * -10000.0
        
        emb_d = model.embDrug(Drug)
        x_d = model.encoderDrug(emb_d.float(), DrugMask_expanded.float(), False)
        
        SEMask_expanded = SEMask.unsqueeze(1).unsqueeze(2)
        SEMask_expanded = (1.0 - SEMask_expanded.float()) * -10000.0
        
        emb_e = model.embSide(SE)
        x_e = model.encoderSide(emb_e.float(), SEMask_expanded.float(), False)
        
        print(f"✓ Embeddings sau encoder:")
        print(f"  - x_d shape: {x_d.shape}")
        print(f"  - x_e shape: {x_e.shape}")
        
        if model.CrossAttention:
            # Test cross-attention
            x_d_original = x_d.clone()
            x_e_original = x_e.clone()
            
            print(f"\n✓ Trước cross-attention:")
            print(f"  - x_d_original shape: {x_d_original.shape}")
            print(f"  - x_e_original shape: {x_e_original.shape}")
            
            cross_output = model.crossAttentionencoder([x_d.float(), x_e.float()], DrugMask_expanded.float(), True)
            
            print(f"\n✓ Sau cross-attention:")
            print(f"  - cross_output shape: {cross_output.shape}")
            print(f"  - cross_output[0] (x_d_cross) shape: {cross_output[0].shape}")
            print(f"  - cross_output[1] (x_e_cross) shape: {cross_output[1].shape}")
            
            # Test residual fusion
            x_d_cross = cross_output[0]
            x_e_cross = cross_output[1]
            
            batch_size, seq_len, hidden_size = x_d_original.shape
            x_d_flat = x_d_original.view(-1, hidden_size)
            x_d_cross_flat = x_d_cross.view(-1, hidden_size)
            x_e_flat = x_e_original.view(-1, hidden_size)
            x_e_cross_flat = x_e_cross.view(-1, hidden_size)
            
            x_d_concat = torch.cat([x_d_flat, x_d_cross_flat], dim=-1)
            x_e_concat = torch.cat([x_e_flat, x_e_cross_flat], dim=-1)
            
            print(f"\n✓ Trong residual fusion:")
            print(f"  - x_d_concat shape: {x_d_concat.shape}")
            print(f"  - x_e_concat shape: {x_e_concat.shape}")
            
            gate_d = model.gate_drug(x_d_concat)
            gate_e = model.gate_side(x_e_concat)
            x_d_fused = model.residual_fusion_drug(x_d_concat)
            x_e_fused = model.residual_fusion_side(x_e_concat)
            
            print(f"  - gate_d shape: {gate_d.shape}")
            print(f"  - gate_e shape: {gate_e.shape}")
            print(f"  - x_d_fused shape: {x_d_fused.shape}")
            print(f"  - x_e_fused shape: {x_e_fused.shape}")
            
            x_d_final = (gate_d * x_d_fused + (1 - gate_d) * x_d_flat).view(batch_size, seq_len, hidden_size)
            x_e_final = (gate_e * x_e_fused + (1 - gate_e) * x_e_flat).view(batch_size, seq_len, hidden_size)
            
            print(f"\n✓ Sau residual fusion:")
            print(f"  - x_d_final shape: {x_d_final.shape}")
            print(f"  - x_e_final shape: {x_e_final.shape}")
            
            # Kiểm tra shape đúng
            assert x_d_final.shape == x_d_original.shape, f"Shape khong khop: {x_d_final.shape} != {x_d_original.shape}"
            assert x_e_final.shape == x_e_original.shape, f"Shape khong khop: {x_e_final.shape} != {x_e_original.shape}"
            
            print("✓ Tat ca shapes deu dung!")
        
        return True
    except Exception as e:
        print(f"✗ Loi khi kiem tra shapes: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(model, device='cpu'):
    """Test 4: Kiểm tra gradient flow"""
    print("\n" + "=" * 60)
    print("TEST 4: Kiểm tra gradient flow")
    print("=" * 60)
    
    try:
        model.train()
        batch_size = 2
        seq_len = 50
        
        Drug = torch.randint(0, 100, (batch_size, seq_len)).long()
        SE = torch.randint(0, 100, (batch_size, seq_len)).long()
        DrugMask = torch.ones((batch_size, seq_len)).long()
        SEMask = torch.ones((batch_size, seq_len)).long()
        labels = torch.randn((batch_size, 1))
        
        Drug = Drug.to(device)
        SE = SE.to(device)
        DrugMask = DrugMask.to(device)
        SEMask = SEMask.to(device)
        labels = labels.to(device)
        
        # Forward
        output, _, _ = model(Drug, SE, DrugMask, SEMask)
        
        # Loss
        loss = nn.MSELoss()(output, labels)
        
        # Backward
        loss.backward()
        
        # Kiểm tra gradient
        has_gradient = False
        no_gradient_layers = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradient = True
                if param.grad.abs().max() < 1e-8:
                    no_gradient_layers.append(name)
            else:
                no_gradient_layers.append(name)
        
        print(f"✓ Loss: {loss.item():.6f}")
        print(f"✓ Có gradient: {has_gradient}")
        
        if no_gradient_layers:
            print(f"[WARNING] Cac layer khong co gradient ({len(no_gradient_layers)}):")
            for name in no_gradient_layers[:5]:  # Chỉ hiển thị 5 đầu tiên
                print(f"    - {name}")
            if len(no_gradient_layers) > 5:
                print(f"    ... va {len(no_gradient_layers) - 5} layer khac")
        else:
            print("✓ Tat ca cac layer deu co gradient")
        
        # Kiểm tra gradient của các layer mới
        print(f"\n✓ Gradient của các layer mới:")
        print(f"  - residual_fusion_drug: {model.residual_fusion_drug[0].weight.grad is not None}")
        print(f"  - residual_fusion_side: {model.residual_fusion_side[0].weight.grad is not None}")
        print(f"  - gate_drug: {model.gate_drug[0].weight.grad is not None}")
        print(f"  - gate_side: {model.gate_side[0].weight.grad is not None}")
        print(f"  - crossAttentionencoder: {model.crossAttentionencoder.layer[0].attention.self.query.weight.grad is not None}")
        
        return True
    except Exception as e:
        print(f"✗ Loi khi kiem tra gradient: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_different_batch_sizes(model, device='cpu'):
    """Test 5: Test với các batch size khác nhau"""
    print("\n" + "=" * 60)
    print("TEST 5: Test với các batch size khác nhau")
    print("=" * 60)
    
    batch_sizes = [1, 2, 4, 8]
    seq_len = 50
    
    try:
        model.eval()
        results = []
        
        for batch_size in batch_sizes:
            try:
                Drug = torch.randint(0, 100, (batch_size, seq_len)).long().to(device)
                SE = torch.randint(0, 100, (batch_size, seq_len)).long().to(device)
                DrugMask = torch.ones((batch_size, seq_len)).long().to(device)
                SEMask = torch.ones((batch_size, seq_len)).long().to(device)
                
                with torch.no_grad():
                    output, _, _ = model(Drug, SE, DrugMask, SEMask)
                
                assert output.shape[0] == batch_size
                results.append(True)
                print(f"✓ Batch size {batch_size}: OK - Output shape: {output.shape}")
            except Exception as e:
                results.append(False)
                print(f"✗ Batch size {batch_size}: FAILED - {e}")
        
        if all(results):
            print("\n✓ Tat ca batch sizes deu hoat dong tot!")
            return True
        else:
            print("\n[WARNING] Mot so batch sizes gap loi")
            return False
    except Exception as e:
        print(f"✗ Loi khi test batch sizes: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_attention_on_off(model, device='cpu'):
    """Test 6: Test bật/tắt cross-attention"""
    print("\n" + "=" * 60)
    print("TEST 6: Test bật/tắt cross-attention")
    print("=" * 60)
    
    try:
        batch_size = 2
        seq_len = 50
        
        Drug = torch.randint(0, 100, (batch_size, seq_len)).long().to(device)
        SE = torch.randint(0, 100, (batch_size, seq_len)).long().to(device)
        DrugMask = torch.ones((batch_size, seq_len)).long().to(device)
        SEMask = torch.ones((batch_size, seq_len)).long().to(device)
        
        model.eval()
        
        # Test với cross-attention ON
        model.CrossAttention = True
        with torch.no_grad():
            output_on, _, _ = model(Drug, SE, DrugMask, SEMask)
        print(f"✓ Cross-attention ON: Output shape {output_on.shape}, Mean: {output_on.mean().item():.6f}")
        
        # Test với cross-attention OFF
        model.CrossAttention = False
        with torch.no_grad():
            output_off, _, _ = model(Drug, SE, DrugMask, SEMask)
        print(f"✓ Cross-attention OFF: Output shape {output_off.shape}, Mean: {output_off.mean().item():.6f}")
        
        # Outputs nên khác nhau
        diff = (output_on - output_off).abs().mean().item()
        print(f"✓ Su khac biet trung binh: {diff:.6f}")
        
        if diff > 1e-6:
            print("✓ Cross-attention co anh huong den output (dung nhu mong doi)")
        else:
            print("[WARNING] Cross-attention khong co anh huong dang ke den output")
        
        # Bật lại
        model.CrossAttention = True
        
        return True
    except Exception as e:
        print(f"✗ Loi khi test bat/tat cross-attention: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Chạy tất cả các test"""
    import sys
    import io
    # Fix encoding cho Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("\n" + "=" * 60)
    print("BAT DAU TEST CROSS-ATTENTION VA RESIDUAL FUSION")
    print("=" * 60)
    
    # Xác định device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test 1: Khởi tạo model
    success, model = test_model_initialization()
    if not success:
        print("\n✗ Khong the tiep tuc test do loi khoi tao model")
        return
    
    model = model.to(device)
    
    # Test 2: Forward pass
    test2 = test_forward_pass(model, device)
    
    # Test 3: Kiểm tra shapes
    test3 = test_cross_attention_output_shapes(model, device)
    
    # Test 4: Gradient flow
    test4 = test_gradient_flow(model, device)
    
    # Test 5: Batch sizes
    test5 = test_with_different_batch_sizes(model, device)
    
    # Test 6: Bật/tắt cross-attention
    test6 = test_cross_attention_on_off(model, device)
    
    # Tóm tắt kết quả
    print("\n" + "=" * 60)
    print("TOM TAT KET QUA TEST")
    print("=" * 60)
    
    tests = [
        ("Khởi tạo model", success),
        ("Forward pass", test2),
        ("Kiểm tra shapes", test3),
        ("Gradient flow", test4),
        ("Batch sizes", test5),
        ("Bật/tắt cross-attention", test6)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nKết quả: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] TAT CA TESTS DEU PASS!")
    else:
        print(f"\n[WARNING] Co {total - passed} test(s) failed. Vui long kiem tra lai.")


if __name__ == "__main__":
    main()

