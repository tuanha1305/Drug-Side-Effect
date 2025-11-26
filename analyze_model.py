"""
Script to analyze number of layers and parameters in HSTrans model
"""
import torch
from Net import Trans

def count_parameters(model):
    """Count trainable and non-trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    return {
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total': total
    }

def format_number(num):
    """Format number with unit"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def analyze_model():
    print("=" * 80)
    print("HSTrans MODEL ANALYSIS")
    print("=" * 80)
    
    # Initialize model
    model = Trans()
    
    # Count total parameters
    params = count_parameters(model)
    print(f"\nTOTAL PARAMETERS:")
    print(f"  - Trainable: {params['trainable']:,} ({format_number(params['trainable'])})")
    print(f"  - Non-trainable: {params['non_trainable']:,} ({format_number(params['non_trainable'])})")
    print(f"  - Total: {params['total']:,} ({format_number(params['total'])})")
    
    # Analyze each component
    print(f"\nDETAILED COMPONENT ANALYSIS:")
    print("-" * 80)
    
    components = {
        'embDrug': model.embDrug,
        'embSide': model.embSide,
        'encoderDrug': model.encoderDrug,
        'encoderSide': model.encoderSide,
        'crossAttentionencoder': model.crossAttentionencoder,
        'decoder': model.decoder,
        'icnn': model.icnn,
        'position_embeddings': model.position_embeddings
    }
    
    total_component_params = 0
    for name, component in components.items():
        comp_params = sum(p.numel() for p in component.parameters())
        total_component_params += comp_params
        print(f"\n{name}:")
        print(f"  - Parameters: {comp_params:,} ({format_number(comp_params)})")
        print(f"  - Percentage: {comp_params/params['total']*100:.2f}%")
    
    # Count Transformer layers
    print(f"\nTRANSFORMER LAYERS:")
    print("-" * 80)
    
    # Drug Encoder
    drug_layers = len(list(model.encoderDrug.layer))
    print(f"  - Drug Encoder: {drug_layers} layers")
    
    # Side Effect Encoder
    se_layers = len(list(model.encoderSide.layer))
    print(f"  - Side Effect Encoder: {se_layers} layers")
    
    # Cross Attention Encoder
    cross_layers = len(list(model.crossAttentionencoder.layer))
    print(f"  - Cross Attention Encoder: {cross_layers} layers")
    
    total_transformer_layers = drug_layers + se_layers + cross_layers
    print(f"  - Total Transformer layers: {total_transformer_layers} layers")
    
    # Analyze Decoder
    print(f"\nDECODER LAYERS:")
    print("-" * 80)
    decoder_layers = 0
    for i, layer in enumerate(model.decoder):
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm1d)):
            decoder_layers += 1
            layer_params = sum(p.numel() for p in layer.parameters())
            print(f"  - Layer {decoder_layers} ({type(layer).__name__}): {layer_params:,} params")
    print(f"  - Total Decoder layers: {decoder_layers} layers")
    
    # Configuration parameters
    print(f"\nCONFIGURATION PARAMETERS:")
    print("-" * 80)
    print(f"  - Vocab size (input_dim_drug): 2586")
    print(f"  - Embedding dimension: 200")
    print(f"  - Max sequence length: 50")
    print(f"  - Transformer hidden size: 200")
    print(f"  - Intermediate size: 512")
    print(f"  - Number of attention heads: 8")
    print(f"  - Drug Encoder layers: 8")
    print(f"  - Side Effect Encoder layers: 8")
    print(f"  - Cross Attention layers: 2")
    print(f"  - Dropout rate: 0.1")
    print(f"  - Cross Attention: {'ENABLED' if model.CrossAttention else 'DISABLED'}")
    
    # Calculate parameters by formula (estimate)
    print(f"\nPARAMETER ESTIMATION BY FORMULA:")
    print("-" * 80)
    
    # Embedding layers
    emb_params = 2 * (2586 * 200 + 50 * 200)  # 2 embeddings (drug + side)
    print(f"  - Embedding layers: ~{emb_params:,} ({format_number(emb_params)})")
    
    # Each Transformer layer has:
    # - Self-attention: 4 * (hidden_size^2) = 4 * 200^2 = 160,000
    # - Feed-forward: 2 * (hidden_size * intermediate_size) = 2 * 200 * 512 = 204,800
    # - LayerNorm: 2 * hidden_size = 400
    # Total per layer: ~365,200
    params_per_transformer_layer = 4 * 200 * 200 + 2 * 200 * 512 + 2 * 200
    print(f"  - Each Transformer layer: ~{params_per_transformer_layer:,} params")
    
    # Drug Encoder
    drug_encoder_params = 8 * params_per_transformer_layer
    print(f"  - Drug Encoder (8 layers): ~{drug_encoder_params:,} ({format_number(drug_encoder_params)})")
    
    # Side Effect Encoder
    se_encoder_params = 8 * params_per_transformer_layer
    print(f"  - Side Effect Encoder (8 layers): ~{se_encoder_params:,} ({format_number(se_encoder_params)})")
    
    # Cross Attention Encoder
    cross_encoder_params = 2 * params_per_transformer_layer
    print(f"  - Cross Attention Encoder (2 layers): ~{cross_encoder_params:,} ({format_number(cross_encoder_params)})")
    
    # Decoder
    decoder_params = 6912 * 512 + 512 + 512 * 64 + 64 + 64 * 32 + 32 + 32 * 1 + 1
    decoder_params += 512 * 2 + 64 * 2  # BatchNorm params (estimate)
    print(f"  - Decoder: ~{decoder_params:,} ({format_number(decoder_params)})")
    
    # CNN
    cnn_params = 1 * 3 * 3 * 3 + 3  # Conv2d(1, 3, 3)
    print(f"  - CNN: ~{cnn_params:,} ({format_number(cnn_params)})")
    
    estimated_total = emb_params + drug_encoder_params + se_encoder_params + cross_encoder_params + decoder_params + cnn_params
    print(f"  - Estimated total: ~{estimated_total:,} ({format_number(estimated_total)})")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    analyze_model()
