## Hướng dẫn chạy inference Ibuprofen (CPU)

### 1) Chuẩn bị
- Cài đặt môi trường Python theo `requirements.txt` (nếu có).
- Đảm bảo có thư mục `checkpoints/` chứa các file checkpoint, ví dụ: `0_70.pth`, `0_71.pth`, `latest_0.pth`...

### 2) Chạy inference dùng checkpoint mới nhất
```bash
python inference_ibuprofen.py
```
Script sẽ tự tìm checkpoint mới nhất trong `checkpoints/` và xuất kết quả vào `results/` và `images/`.

### 3) Ép dùng checkpoint cụ thể (ví dụ epoch 71)
```bash
python inference_ibuprofen.py --checkpoint checkpoints/0_71.pth
```
Nếu file không tồn tại, hãy thay bằng checkpoint gần nhất (vd `0_70.pth`).

### 4) Đầu ra chính
- `results/ibuprofen_detailed_results.json`: JSON chi tiết toàn bộ dự đoán và thống kê.
- `results/ibuprofen_top50_predictions.csv`: Top-50 side effects dự đoán.
- `results/ibuprofen_all_predictions.csv`: Tất cả 994 dự đoán.
- `images/ibuprofen_top20_predictions.png`: Biểu đồ Top-20.
- `images/ibuprofen_score_distribution.png`: Phân phối điểm dự đoán.
- `images/ibuprofen_confidence_levels.png`: Tỉ lệ mức độ tin cậy.
- `images/ibuprofen_substructure_se_heatmap.png`: Heatmap tương tác substructure (thuốc) × substructure (side effect) cho top-1 side effect.

### 5) Ghi chú
- Mặc định chạy trên CPU.
- Nếu thay đổi cấu trúc model, cần train lại hoặc đảm bảo checkpoint tương thích.
# HSTrans
Here, we introduce HSTrans, an end-to-end homogeneous substructures transformer network designed for
predicting drug-side effect frequencies. 

# Environment
* python == 3.8.5
* pytorch == 1.6
* Numpy == 1.21.0
* scikit-learn == 0.23.2
* scipy == 1.5.0
* rdkit == 2020.03.6
* matplotlib == 3.3.1
* networkx == 2.5

# Files:
1.original_data

This folder contains our original side effects and drugs data.

* **Supplementary Data 1.txt:**     
  The standardised drug side effect frequency classes used in our study. 


* **Supplementary Data 2.txt:**  
The postmarketing drug side effect associations from SIDER and OFFSIDES.
  

* **Supplementary Data 3.txt:**  
Main or High‐Level Term (HLT) Medical Dictionary for Regulatory Activities (MedDRA) terminology classification for each side effect term.
  

* **Supplementary Data 4.txt:**  
High‐Level Group Term Medical Dictionary for Regulatory Activities (MedDRA) terminology classification for each side effect term.
  


2.data
* **drug_codes_chembl_freq_1500.txt**   
Corpus of drug substructures used for learning BPE encoder.


* **drug_side.pkl**     
Frequency matrix of side effects of 750 drugs.


* **drug_SMILES_750.csv**   
SMILES of 750 drugs.


* **drug_SMILES_759.csv**   
SMILES files for 750 initial drugs and 9 independent test sets, and the position of 9 drugs is before 750 drugs.


* **raw_frequency_750.mat**  
The original frequency matrix of side effects of 750 drugs, including frequency matrix 'R', drug name 'drugs' and side effect name' sideeffect '.


* **frequency_750+9.mat**      
The original frequency matrix of side effects of drugs in 750 drugs and 9 independent test sets, and the position of 9 drugs is before 750 drugs.
  

* **data/SE_sub_index_50.npy**  
Effective substructures extracted for each side effect.


* **SE_sub_mask_50.npy**  
Substructure mask matrix for side effects.


* **side_effect_label_750.mat**  
Label vector for 994 side effects.

* **subword_units_map_chembl_freq_1500.csv**  
The substructures learned by the BPE encoder and their corresponding indices.


# Code 

main.py: Test of 750 drugs.

Net.py: It defines the model used by the code.

Encoder.py: It defines transformer encoder.

smiles2vector.py: It defines a method to calculate the smiles of drugs as vertices and edges of a graph.

utils.py: It defines performance indicators.


# Run

epoch: Define the number of epochs.

lr: Define the learning rate.

lamb: Define weights for unknown associations.


Example:
```bash
python main.py --save_model --epoch 300 --lr 0.0001
```

# Contact
If you have any questions or suggestions with the code, please let us know. Contact Kaiyi Xu at xuky@cug.edu.cn