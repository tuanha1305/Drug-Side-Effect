import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder_MultipleLayers, Embeddings
import torch
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE


class Trans(torch.nn.Module):
    def __init__(self):
        super(Trans, self).__init__()

        # activation and regularization
        self.relu = nn.ReLU()

        input_dim_drug = 2586
        transformer_emb_size_drug = 200
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        # Embedding encoding layer
        self.embDrug = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        self.embSide = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        # Transformer layer
        self.encoderDrug = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

        self.encoderSide = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

        # Cross Attention Encoder - để thực hiện cross attention giữa drug và side effect
        # Sử dụng ít layer hơn để tránh overfitting
        cross_attention_n_layer = 2  # Có thể điều chỉnh số layer
        self.crossAttentionencoder = Encoder_MultipleLayers(cross_attention_n_layer,
                                                             transformer_emb_size_drug,
                                                             transformer_intermediate_size_drug,
                                                             transformer_num_attention_heads_drug,
                                                             transformer_attention_probs_dropout,
                                                             transformer_hidden_dropout_rate)

        # Residual Fusion Layer - kết hợp output của cross-attention với input ban đầu
        # Sử dụng gated fusion để học cách kết hợp tốt nhất
        self.residual_fusion_drug = nn.Sequential(
            nn.Linear(transformer_emb_size_drug * 2, transformer_emb_size_drug),
            nn.LayerNorm(transformer_emb_size_drug),
            nn.ReLU(),
            nn.Dropout(transformer_dropout_rate),
            nn.Linear(transformer_emb_size_drug, transformer_emb_size_drug)
        )
        
        self.residual_fusion_side = nn.Sequential(
            nn.Linear(transformer_emb_size_drug * 2, transformer_emb_size_drug),
            nn.LayerNorm(transformer_emb_size_drug),
            nn.ReLU(),
            nn.Dropout(transformer_dropout_rate),
            nn.Linear(transformer_emb_size_drug, transformer_emb_size_drug)
        )
        
        # Gating mechanism để điều chỉnh trọng số giữa original và cross-attention output
        self.gate_drug = nn.Sequential(
            nn.Linear(transformer_emb_size_drug * 2, transformer_emb_size_drug),
            nn.Sigmoid()
        )
        
        self.gate_side = nn.Sequential(
            nn.Linear(transformer_emb_size_drug * 2, transformer_emb_size_drug),
            nn.Sigmoid()
        )

        # Positional encoding layer
        self.position_embeddings = nn.Embedding(500, 200)


        self.dropout = 0.3

        self.decoder = nn.Sequential(
            nn.Linear(6912, 512),
            nn.ReLU(True),

            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 1)
        )

        self.icnn = nn.Conv2d(1, 3, 3, padding=0)

        # Bật Cross Attention
        self.CrossAttention = True

    def forward(self, Drug, SE, DrugMask, SEMsak):

        batch = Drug.size(0)
        
        # Lấy device từ model parameters (đảm bảo nhất quán với device của model)
        device = next(self.parameters()).device

        # Substructure encoding
        Drug = Drug.long().to(device)
        DrugMask = DrugMask.long().to(device)
        DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
        DrugMask = (1.0 - DrugMask) * -10000.0
        emb = self.embDrug(Drug)
        encoded_layers = self.encoderDrug(emb.float(), DrugMask.float(), False)
        x_d = encoded_layers

        # Side-effect substructure encoding
        SE = SE.long().to(device)
        SEMsak = SEMsak.long().to(device)
        SEMsak = SEMsak.unsqueeze(1).unsqueeze(2)
        SEMsak = (1.0 - SEMsak) * -10000.0
        embE = self.embSide(SE)
        encoded_layers = self.encoderSide(embE.float(), SEMsak.float(), False)
        x_e = encoded_layers

        if self.CrossAttention:
            # Lưu embeddings ban đầu để dùng cho residual fusion
            x_d_original = x_d.clone()
            x_e_original = x_e.clone()
            
            # Tạo combined mask cho cross-attention
            # DrugMask shape: (batch, 1, 1, seq_len_drug) - mask cho drug sequence
            # SEMsak shape: (batch, 1, 1, seq_len_side) - mask cho side effect sequence
            # Trong cross-attention, cần mask cho cả hai phía
            # Vì cả drug và side effect đều có cùng độ dài (50), có thể dùng DrugMask cho cả hai
            # Hoặc tạo combined mask nếu cần xử lý khác nhau
            combined_mask = DrugMask.float()  # Có thể mở rộng để xử lý mask riêng biệt
            
            # Cross attention: drug và side effect tương tác với nhau
            # Output có shape (2, batch, seq_len, hidden_size)
            cross_output = self.crossAttentionencoder([x_d.float(), x_e.float()], combined_mask, True)
            
            # Tách output thành x_d và x_e
            # cross_output là tensor shape (2, batch, seq_len, hidden_size)
            x_d_cross = cross_output[0]  # Drug embeddings sau cross attention
            x_e_cross = cross_output[1]  # Side effect embeddings sau cross attention
            
            # Residual Fusion: kết hợp output của cross-attention với input ban đầu
            # Sử dụng gated fusion để học cách kết hợp tốt nhất
            batch_size, seq_len, hidden_size = x_d_original.shape
            
            # Reshape để xử lý qua fusion layers: (batch * seq_len, hidden_size)
            x_d_flat = x_d_original.view(-1, hidden_size)
            x_d_cross_flat = x_d_cross.view(-1, hidden_size)
            x_e_flat = x_e_original.view(-1, hidden_size)
            x_e_cross_flat = x_e_cross.view(-1, hidden_size)
            
            # Concatenate original và cross-attention output
            x_d_concat = torch.cat([x_d_flat, x_d_cross_flat], dim=-1)  # (batch*seq_len, 2*hidden_size)
            x_e_concat = torch.cat([x_e_flat, x_e_cross_flat], dim=-1)  # (batch*seq_len, 2*hidden_size)
            
            # Gated fusion: học trọng số để kết hợp
            gate_d = self.gate_drug(x_d_concat)  # (batch*seq_len, hidden_size)
            gate_e = self.gate_side(x_e_concat)  # (batch*seq_len, hidden_size)
            
            # Fusion: kết hợp với gating
            x_d_fused = self.residual_fusion_drug(x_d_concat)  # (batch*seq_len, hidden_size)
            x_e_fused = self.residual_fusion_side(x_e_concat)  # (batch*seq_len, hidden_size)
            
            # Gated combination: gate * fused + (1 - gate) * original
            x_d = (gate_d * x_d_fused + (1 - gate_d) * x_d_flat).view(batch_size, seq_len, hidden_size)
            x_e = (gate_e * x_e_fused + (1 - gate_e) * x_e_flat).view(batch_size, seq_len, hidden_size)


        # interaction
        d_aug = torch.unsqueeze(x_d, 2).repeat(1, 1, 50, 1)
        e_aug = torch.unsqueeze(x_e, 1).repeat(1, 50, 1, 1)


        i = d_aug * e_aug
        i_v = i.permute(0, 3, 1, 2)
        i_v = torch.sum(i_v, dim=1)
        i_v = torch.unsqueeze(i_v, 1)
        i_v = F.dropout(i_v, p=self.dropout)

        f = self.icnn(i_v)

        f = f.view(int(batch), -1)

        score = self.decoder(f)

        return score, Drug, SE

    def forward_with_interaction(self, Drug, SE, DrugMask, SEMsak):
        """
        Chạy forward và trả thêm ma trận tương tác 50x50 (drug_substructure x side_substructure)
        """
        batch = Drug.size(0)
        
        # Lấy device từ model parameters (đảm bảo nhất quán với device của model)
        device = next(self.parameters()).device

        Drug = Drug.long().to(device)
        DrugMask = DrugMask.long().to(device)
        DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
        DrugMask = (1.0 - DrugMask) * -10000.0
        emb = self.embDrug(Drug)
        x_d = self.encoderDrug(emb.float(), DrugMask.float(), False)

        SE = SE.long().to(device)
        SEMsak = SEMsak.long().to(device)
        SEMsak = SEMsak.unsqueeze(1).unsqueeze(2)
        SEMsak = (1.0 - SEMsak) * -10000.0
        embE = self.embSide(SE)
        x_e = self.encoderSide(embE.float(), SEMsak.float(), False)

        if self.CrossAttention:
            # Lưu embeddings ban đầu để dùng cho residual fusion
            x_d_original = x_d.clone()
            x_e_original = x_e.clone()
            
            # Tạo combined mask cho cross-attention
            combined_mask = DrugMask.float()
            
            # Cross attention: drug và side effect tương tác với nhau
            cross_output = self.crossAttentionencoder([x_d.float(), x_e.float()], combined_mask, True)
            
            # Tách output thành x_d và x_e
            x_d_cross = cross_output[0]  # Drug embeddings sau cross attention
            x_e_cross = cross_output[1]  # Side effect embeddings sau cross attention
            
            # Residual Fusion: kết hợp output của cross-attention với input ban đầu
            batch_size, seq_len, hidden_size = x_d_original.shape
            
            # Reshape để xử lý qua fusion layers
            x_d_flat = x_d_original.view(-1, hidden_size)
            x_d_cross_flat = x_d_cross.view(-1, hidden_size)
            x_e_flat = x_e_original.view(-1, hidden_size)
            x_e_cross_flat = x_e_cross.view(-1, hidden_size)
            
            # Concatenate original và cross-attention output
            x_d_concat = torch.cat([x_d_flat, x_d_cross_flat], dim=-1)
            x_e_concat = torch.cat([x_e_flat, x_e_cross_flat], dim=-1)
            
            # Gated fusion
            gate_d = self.gate_drug(x_d_concat)
            gate_e = self.gate_side(x_e_concat)
            
            # Fusion
            x_d_fused = self.residual_fusion_drug(x_d_concat)
            x_e_fused = self.residual_fusion_side(x_e_concat)
            
            # Gated combination
            x_d = (gate_d * x_d_fused + (1 - gate_d) * x_d_flat).view(batch_size, seq_len, hidden_size)
            x_e = (gate_e * x_e_fused + (1 - gate_e) * x_e_flat).view(batch_size, seq_len, hidden_size)

        d_aug = torch.unsqueeze(x_d, 2).repeat(1, 1, 50, 1)
        e_aug = torch.unsqueeze(x_e, 1).repeat(1, 50, 1, 1)

        interaction_tensor = d_aug * e_aug  # shape: (B, 50, 50, 200)
        interaction_map = interaction_tensor.permute(0, 3, 1, 2).sum(dim=1)  # (B, 50, 50)

        i_v = torch.unsqueeze(interaction_map, 1)
        i_v = F.dropout(i_v, p=self.dropout)
        f = self.icnn(i_v)
        f = f.view(int(batch), -1)
        score = self.decoder(f)

        return score, interaction_map


def drug2emb_encoder(smile):
    vocab_path = 'data/drug_codes_chembl_freq_1500.txt'
    sub_csv = pd.read_csv('data/subword_units_map_chembl_freq_1500.csv')

    # Initialize a BPE encoder for tokenizing or encoding text
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    idx2word_d = sub_csv['index'].values  # Extract all substructure lists
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # Build a dictionary mapping substructures to indices

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # Map the SMILES substructures to their indices and create an index ndarray
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]  # Perform padding
        input_mask = [1] * max_d  # Construct a mask to cover the padded parts

    return i, np.asarray(input_mask)
