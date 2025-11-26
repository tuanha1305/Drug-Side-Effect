import argparse
import os
import json
import pickle
import scipy
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import io
from Net import *
from smiles2vector import load_drug_smile
from math import *
import random
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as data
from sklearn.metrics import precision_score, recall_score, accuracy_score
from utils import *

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.csv'
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
input_dim = 109
gii = open('data/drug_side.pkl', 'rb')
drug_side = pickle.load(gii)
gii.close()


def Extract_positive_negative_samples(DAL, addition_negative_number=''):
    k = 0
    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]  # 按照最后一列对行排序
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    final_positive_sample = np.concatenate((final_positive_sample, final_negtive_sample), axis=0)
    return addition_negative_sample, final_positive_sample, final_negtive_sample


def loss_fun(output, label):
    # Tính loss trên cùng thiết bị của tham số truyền vào
    loss = torch.sum((output - label) ** 2)
    return loss


def identify_sub(data, k):
    print('Dang trich xuat cac tieu cau truc hieu qua')
    drug_smile = [item[1] for item in data]
    side_id = [item[0] for item in data]
    labels = [item[2] for item in data]

    # 获得SMILE-sub序号
    sub_dict = {}
    for i in range(len(drug_smile)):
        drug_sub, mask = drug2emb_encoder(drug_smile[i])
        drug_sub = drug_sub.tolist()
        sub_dict[i] = drug_sub

    # 暂存成文件
    with open(f'data/sub/my_dict_{k}.pkl', 'wb') as f:
        pickle.dump(sub_dict, f)
    # 读取文件
    with open(f'data/sub/my_dict_{k}.pkl', 'rb') as f:
        sub_dict = pickle.load(f)

    SE_sub = np.zeros((994, 2686))
    for j in range(len(drug_smile)):
        sideID = side_id[j]
        label = float(labels[j])
        for k in sub_dict[j]:
            if k == 0:
                continue
            SE_sub[int(sideID)][int(k)] += label

    np.save(f"data/sub/SE_sub_{k}.npy", SE_sub)
    SE_sub = np.load(f"data/sub/SE_sub_{k}.npy", allow_pickle=True)

    # 总和
    n = np.sum(SE_sub)
    # 计算行和
    SE_sum = np.sum(SE_sub, axis=1)
    SE_p = SE_sum / n
    # 计算列和
    Sub_sum = np.sum(SE_sub, axis=0)
    Sub_p = Sub_sum / n

    SE_sub_p = SE_sub / n

    freq = np.zeros((994, 2686))
    for i in range(994):
        print(i)
        for j in range(2686):
            freq[i][j] = ((SE_sub_p[i][j] - SE_p[i] * Sub_p[j]) / (sqrt((SE_p[i] * Sub_p[j] / n)
                                                                        * (1 - SE_p[i]) *
                                                                        (1 - Sub_p[j])))) + 1e-5
    np.save(f"data/sub/freq_{k}.npy", freq)
    freq = np.load(f"data/sub/freq_{k}.npy", allow_pickle=True)
    non_nan_values = freq[~np.isnan(freq)]
    percentile_95 = np.percentile(non_nan_values, 95)
    print("Phan vi 95%:", percentile_95)

    l = []
    SE_sub_index = np.zeros((994, 50))
    for i in range(994):
        k = 0
        sorted_indices = np.argsort(freq[i])[::-1]
        filtered_indices = sorted_indices[freq[i][sorted_indices] > percentile_95]
        l.append(len(filtered_indices))
        for j in filtered_indices:
            if k < 50:
                SE_sub_index[i][k] = j
                k = k + 1
            else:
                continue

    np.save(f"data/sub/SE_sub_index_50_{k}.npy", SE_sub_index)
    SE_sub_index = np.load(f"data/sub/SE_sub_index_50_{k}.npy")

    SE_sub_mask = SE_sub_index
    SE_sub_mask[SE_sub_mask > 0] = 1
    np.save(f"data/sub/SE_sub_mask_50_{k}.npy", SE_sub_mask)
    np.save("len_sub", l)


def trainfun(model, device, train_loader, optimizer, epoch, log_interval, test_loader):
    # Train one epoch
    model.train()
    avg_loss = []

    for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(train_loader):

        # Đưa mọi tensor lên đúng device
        Drug = Drug.to(device)
        SE = SE.to(device)
        DrugMask = DrugMask.to(device)
        SEMsak = SEMsak.to(device)
        Label = torch.FloatTensor([int(item) for item in Label]).to(device)

        optimizer.zero_grad()
        out, _, _ = model(Drug, SE, DrugMask, SEMsak)

        pred = out.to(device)

        loss = loss_fun(pred.flatten(), Label).to('cpu')

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())

    return sum(avg_loss) / len(avg_loss)


def predict(model, device, test_loader):
    # 声明为张量
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.FloatTensor([int(item) for item in Label]).to(device)
            out, _, _ = model(Drug, SE, DrugMask, SEMsak)

            location = torch.where(Label != 0)
            pred = out[location]
            label = Label[location]

            total_preds = torch.cat((total_preds, pred.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, label.detach().cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate(model, device, test_loader):
    total_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.FloatTensor([int(item) for item in Label]).to(device)
            output, _, _ = model(Drug, SE, DrugMask, SEMsak)
            pred = output.detach().cpu()
            pred = torch.Tensor(pred)

            total_preds = torch.cat((total_preds, pred), 0)
            total_label = torch.cat((total_label, Label), 0)

            pred = pred.numpy().flatten()
            pred = np.where(pred > 0.5, 1, 0)
            label = (Label.numpy().flatten() != 0).astype(int)
            label = np.where(label != 0, 1, label)

            singleDrug_auc.append(roc_auc_score(label, pred))
            singleDrug_aupr.append(average_precision_score(label, pred))

        drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
        drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
        total_preds = total_preds.numpy()
        total_label = total_label.numpy()

        total_pre_binary = np.where(total_preds > 0.5, 1, 0)
        label01 = np.where(total_label != 0, 1, total_label)

        pre_list = total_pre_binary.tolist()
        label_list = label01.tolist()

        precision = precision_score(pre_list, label_list)

        # 计算召回率
        recall = recall_score(pre_list, label_list)

        # 计算准确率
        accuracy = accuracy_score(pre_list, label_list)

        total_preds = np.where(total_preds > 0.5, 1, 0)
        total_label = np.where(total_label != 0, 1, total_label)

        pos = np.squeeze(total_preds[np.where(total_label)])
        pos_label = np.ones(len(pos))

        neg = np.squeeze(total_preds[np.where(total_label == 0)])
        neg_label = np.zeros(len(neg))

        y = np.hstack((pos, neg))
        y_true = np.hstack((pos_label, neg_label))
        auc_all = roc_auc_score(y_true, y)
        aupr_all = average_precision_score(y_true, y)

    return auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy


def main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay, log_interval, cuda_name,
         save_model, k, save_every=5, resume_path=None):
    print('\n=======================================================================================')
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('weight_decay: ', weight_decay)

    model_st = modeling.__name__
    train_losses = []

    # 确定设备
    print('CPU/GPU: ', torch.cuda.is_available())
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # 模型初始化
    model = modeling().to(device)

    # 计算模型的参数总数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Resume từ checkpoint nếu có
    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = int(ckpt.get('epoch', 0))
                # Restore random seeds để đảm bảo reproducibility
                if 'random_state' in ckpt:
                    torch.set_rng_state(ckpt['random_state'])
                if 'numpy_random_state' in ckpt:
                    np.random.set_state(ckpt['numpy_random_state'])
                print(f"Resuming from {resume_path} at epoch {start_epoch}")
            else:
                # Fallback cho checkpoint cũ chỉ có state_dict
                model.load_state_dict(ckpt)
                print(f"Loaded model weights from legacy checkpoint {resume_path}")
        except Exception as e:
            print(f"⚠ Không thể resume từ {resume_path}: {e}")

    history = []
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Nếu đã có lịch sử trước đó (resume), thử nạp để nối tiếp
    metrics_file = os.path.join('results', 'train_metrics_per_epoch.json')
    if start_epoch > 0 and os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []

    for epoch in range(start_epoch, num_epoch):
        train_loss = trainfun(model=model, device=device,
                              train_loader=training_generator,
                              optimizer=optimizer, epoch=epoch + 1, log_interval=log_interval,
                              test_loader=testing_generator)
        train_losses.append(train_loss)

        # Lưu checkpoint định kỳ và ở epoch cuối
        if ((epoch + 1) % save_every == 0) or (epoch == num_epoch - 1):
            ckpt_obj = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'random_state': torch.get_rng_state(),
                'numpy_random_state': np.random.get_state(),
            }
            ckpt_path = os.path.join('checkpoints', f'{k}_{epoch + 1}.pth')
            torch.save(ckpt_obj, ckpt_path)
            # lưu alias latest cho tiện resume
            torch.save(ckpt_obj, os.path.join('checkpoints', f'latest_{k}.pth'))

        # evaluate quickly per-epoch on test for summary metrics
        model.eval()
        with torch.no_grad():
            y_true, y_pred = predict(model=model, device=device, test_loader=testing_generator)
        ep_mse = mse(y_true, y_pred)
        ep_rmse = rmse(y_true, y_pred)
        ep_scc = spearman(y_true, y_pred)
        print(f"Epoch {epoch+1}/{num_epoch} - TrainLoss: {train_loss:.6f} - MSE: {ep_mse:.6f} - RMSE: {ep_rmse:.6f} - SCC: {ep_scc:.6f}")
        history.append({
            'epoch': int(epoch+1),
            'train_loss': float(train_loss),
            'MSE': float(ep_mse),
            'RMSE': float(ep_rmse),
            'SCC': float(ep_scc)
        })
        # Ghi file lịch sử sau mỗi epoch để tránh mất dữ liệu khi dừng giữa chừng
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print("Dang du doan")
    test_labels, test_preds = predict(model=model, device=device, test_loader=testing_generator)


    # lưu dự đoán
    os.makedirs('predictResult', exist_ok=True)
    np.save(f'predictResult/total_labels_{k}.npy', test_labels)
    np.save(f'predictResult/total_preds_{k}.npy', test_preds)

    # Tính metrics hồi quy chính: MSE, RMSE, Spearman (SCC)
    test_MSE = mse(test_labels, test_preds)
    test_RMSE = rmse(test_labels, test_preds)
    test_SCC = spearman(test_labels, test_preds)

    print("Dang danh gia")
    auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy = evaluate(model=model, device=device,
                                                                                 test_loader=testing_generator)

    # In tóm tắt và lưu file metrics
    print('Test (Regression): MSE: {:.5f}\tRMSE: {:.5f}\tSCC: {:.5f}'.format(test_MSE, test_RMSE, test_SCC))
    print('\tClassification (tham khảo): all AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}\tPrecise: {:.5f}\tRecall: {:.5f}\tACC: {:.5f}'.format(
        auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy))

    os.makedirs('results', exist_ok=True)
    # save per-epoch history (đã được ghi từng epoch; vẫn ghi lại lần nữa để chắc chắn)
    with open(os.path.join('results', 'train_metrics_per_epoch.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    metrics_path = os.path.join('results', 'metrics_original.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'MSE': float(test_MSE),
            'RMSE': float(test_RMSE),
            'SCC': float(test_SCC),
            'AUC_all': float(auc_all),
            'AUPR_all': float(aupr_all),
            'AUC_drug': float(drugAUC),
            'AUPR_drug': float(drugAUPR),
            'Precision': float(precision),
            'Recall': float(recall),
            'Accuracy': float(accuracy)
        }, f, ensure_ascii=False, indent=2)
    print('Saved metrics to', metrics_path)


class Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti, k):
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.k = k

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        d = self.df.iloc[index]['Drug_smile']
        s = int(self.df.iloc[index]['SE_id'])

        # d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)

        # 副作用的子结构是读取出来的
        SE_index = np.load(f"data/sub/SE_sub_index_50_32.npy").astype(int)
        SE_mask = np.load(f"data/sub/SE_sub_mask_50_32.npy")
        s_v = SE_index[s, :]
        input_mask_s = SE_mask[s, :]
        y = self.labels[index]
        return d_v, s_v, input_mask_d, input_mask_s, y


if __name__ == '__main__':
    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0)
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.01, help='weight_decay')
    parser.add_argument('--epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=40, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cpu', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200,
                        help='features dimensions of drugs and side effects')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model and features')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint path')

    args = parser.parse_args()

    modeling = [Trans][args.model]
    lr = args.lr
    num_epoch = args.epoch
    weight_decay = args.wd
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    save_model = args.save_model

    #  获取正负样本
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(
        drug_side, addition_negative_number='all')

    addition_negative_sample = np.vstack((addition_negative_sample, final_negative_sample))

    final_sample = final_positive_sample

    X = final_sample[:, 0::]

    final_target = final_sample[:, final_sample.shape[1] - 1]

    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    drug_dict, drug_smile = load_drug_smile(SMILES_file)


    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 1], addition_negative_sample[i, 0]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append(
            (addition_negative_sample[i, 1], addition_negative_sample[i, 0], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 1], X[i, 0]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 1], drug_smile[X[i, 0]], X[i, 2]))

    fold = 1
    kfold = StratifiedKFold(10, random_state=1, shuffle=True)

    params = {'batch_size': 128,
              'shuffle': True}

    identify_sub(data, 0)

    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        data_train = np.array(data)[train]
        data_test = np.array(data)[test]

        # 将数据转为DataFrame
        df_train = pd.DataFrame(data=data_train.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])
        df_test = pd.DataFrame(data=data_test.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])

        # 创建数据集和数据加载器
        training_set = Data_Encoder(df_train.index.values, df_train.Label.values, df_train, k)
        testing_set = Data_Encoder(df_test.index.values, df_test.Label.values, df_test, k)

        training_generator = torch.utils.data.DataLoader(training_set, **params)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params)

        main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay, log_interval,
             cuda_name, save_model, k, resume_path=args.resume)
