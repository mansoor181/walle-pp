from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn import metrics
from model import GAT
import argparse
import pickle
import torch

arg_parser = argparse.ArgumentParser(description="Model Evaluation...")

arg_parser.add_argument("--model_checkpoint", type=str, default="checkpoint")

arg_parser.add_argument("--kfold", type=int, default=10)

arg_parser.add_argument("--hid_dim", type=int, default=128)

arg_parser.add_argument("--num_head", type=int, default=8)

arg_parser.add_argument("--device", type=str, default="cuda")

arg_parser.add_argument("--threshold", type=float, default=0.1481)

args = arg_parser.parse_args()

model_checkpoint = args.model_checkpoint
kfold = args.kfold
hid_dim = args.hid_dim
num_head = args.num_head
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
threshold = args.threshold


with open("epitope3d_test_data.pkl", "rb") as f:
    test_pyg_list = pickle.load(f)


## fixed version
import os
def ensemble(direc, data_pyg_list, kfold):
    model = GAT(in_dim=1792, hid_dim=hid_dim, out_dim=1, num_head=num_head, out_head=1)

    model.to(device)

    num_unmasked_nodes = 0 
    for i in data_pyg_list:
        num_unmasked_nodes += len(i.y[i.train_mask])
    pred_ensem = torch.zeros([num_unmasked_nodes])
    
    pt_list = []
    for pt in os.listdir(f"{direc}/"):
        if pt[-3:] == ".pt":
            pt_list.append(pt)
    for pt in pt_list:
        model.load_state_dict(torch.load(f'{direc}/{pt}'))
        model.eval()
        pred_ensem_list = []
        true_label_list = []
        
        pred_label_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_pyg_list):
                batch.to(device)
                out = model(batch)
                # prediction on the surface residues with RSA 15%
                y_pred = out[batch.train_mask].reshape(-1)
                for i in y_pred:
                    pred_label_list.append(i)
                for j in batch.y[batch.train_mask]:    
                    true_label_list.append(j.cpu())
            
        pred_label_list = torch.tensor(pred_label_list)
        pred_ensem += pred_label_list
    for i in (pred_ensem / kfold): # 10 fold
        pred_ensem_list.append(i.cpu())

    return true_label_list, pred_ensem_list


true_label_list, pred_label_list = ensemble(direc=model_checkpoint, data_pyg_list=test_pyg_list, kfold=kfold)

fpr, tpr, _ = metrics.roc_curve(true_label_list, pred_label_list)
roc_auc = metrics.auc(fpr, tpr)
print("roc_auc", "{:0.2f}".format(roc_auc))

precision, recall, thresholds = metrics.precision_recall_curve(true_label_list, pred_label_list)
auc_precision_recall = metrics.auc(recall, precision)
print("roc_pr", "{:0.2f}".format(auc_precision_recall))

def to_label(pred_label_list, threshold):
    pred_binary_list = []
    for pred in pred_label_list:
        if pred > threshold:
            pred_binary_list.append(1)
        else:
            pred_binary_list.append(0)
            
    return pred_binary_list

pred_binary_list = to_label(pred_label_list, threshold)

print("f1 score :", "{:0.2f}".format(f1_score(true_label_list, pred_binary_list)))

print("BACC score :", "{:0.2f}".format(balanced_accuracy_score(true_label_list, pred_binary_list)))

print("MCC score :" ,"{:0.2f}".format(matthews_corrcoef(true_label_list, pred_binary_list)))
