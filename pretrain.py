import os, pickle, torch, argparse
import os.path as osp
import torch.nn as nn
import numpy as np
from rich.progress import track
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import unet
from utils import eval_model, relabel
cur_dir = osp.dirname(osp.abspath(__file__))
os.chdir(cur_dir)


SEED=56
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='preprocessed', choices=['raw', 'preprocessed'], type=str, help='data type')
    parser.add_argument("--data-split", default=0.9, type=float)
    parser.add_argument("--bsz", default=128, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--coeff", default=1, type=float, help='coefficient for classification loss')
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--classifier-channels", default=1, type=int)

    parser.add_argument("--device", choices=[-1,0,1,2,3], default=1, type=int, help='device number (-1 for cpu)')

    args=parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # load data and inspect

    if args.data == 'raw':
        data_full = np.load('../DRIAMS/DRIAMS-B-data.npy', allow_pickle=True)
        data = data_full[1:, :-1]
        labels = data_full[1:, -1]
    elif args.data == 'preprocessed':
        data_full = np.load('../DRIAMS/DRIAMS-BCD-preprocessed-nonsingleton.npy', allow_pickle=True)
        data = data_full[:, :-1] / np.sum(data_full[:, :-1], axis=1, keepdims=True)   # normalize data
        labels = data_full[:, -1]
    
    num_labels = len(set(labels))
    labels = relabel(labels)

    print("Data Shape {}; Labels Shape {}; Num Labels {}".format(data.shape, labels.shape, num_labels))   
    # pad length of each data vector to a multiple of 32
    pad_len = 32 - data.shape[-1] % 32
    if pad_len > 0:
        pad_data = np.zeros((data.shape[0], pad_len))
        data = np.concatenate((data, pad_data), axis=1)
    data = data[:, np.newaxis, :]
    
    # print(data.shape, cal_loss_mask[-43:])
    # assert 1==0

    data_split = args.data_split

    # Stratified split of dataset
    X_train, X_val, y_train, y_val = train_test_split(data, labels, train_size=data_split, stratify=labels, random_state=SEED)

    # Convert numpy arrays to PyTorch tensors
    data_tensor_train = torch.from_numpy(X_train)
    labels_tensor_train = torch.from_numpy(y_train)
    data_tensor_val = torch.from_numpy(X_val)
    labels_tensor_val = torch.from_numpy(y_val)



    # Create a TensorDatasets from the data and labels
    train_dataset = TensorDataset(data_tensor_train, labels_tensor_train)
    val_dataset = TensorDataset(data_tensor_val, labels_tensor_val)


    # define dataloaders
    bsz = args.bsz
    train_loader = DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=bsz, shuffle=False)
    print("Train: {}; Val: {}".format(len(train_dataset), len(val_dataset)))

    # instantiate model
    model = unet(num_classes=num_labels, classifier_channels=args.classifier_channels, vec_len=data.shape[-1])
    # my_input = torch.randn((1, 1, 1024))
    # outputs, h = model(my_input)
    # print(outputs.shape, h.shape)

    # train and evaluate
    epochs = args.epochs
    lr = args.lr
    weight_decay=args.weight_decay
    coeff = args.coeff
    device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_r = nn.MSELoss(reduction='none')
    criterion_h = nn.CrossEntropyLoss()

    trajectories = []
    best_acc = 0
    for epoch in track(range(epochs), description='training network'):
        losses_r = []
        losses_h = []
        total_correct = 0
        total_samples = 0
        for _, (data, labels) in enumerate(train_loader):

            cal_loss_mask = torch.ones_like(data)# do not calculate losss for padded portions
            cal_loss_mask[:, :, -pad_len:] = 0
            non_pad_eles = data.shape[-1] - pad_len

            cal_loss_mask = cal_loss_mask.to(device)
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device)

            optimizer.zero_grad()
            model.train()
            r, h, h0, b = model(data)

            _, predicted = torch.max(h, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.shape[0]
            
            loss_r = criterion_r(r, data)
            loss_r = loss_r * cal_loss_mask
            loss_r = loss_r.squeeze().sum() / non_pad_eles

            loss_h = criterion_h(h, labels)
            train_loss = loss_r + loss_h * coeff
            train_loss.backward()
            optimizer.step()
            losses_r.append(loss_r.detach().cpu().item())
            losses_h.append(loss_h.detach().cpu().item())

        avg_loss_r_train = sum(losses_r) / len(losses_r)
        avg_loss_h_train = sum(losses_h) / len(losses_h)
        accuracy_train = total_correct / total_samples
        avg_loss_r_val, avg_loss_h_val, accuracy_val = eval_model(model, val_loader, device, pad_len, criterion_r, criterion_h)
        print("Epoch {},  Train Loss_r: {:.5f}, Train Loss_h: {:.5f}, Train Acc: {:.4f},\
            Val Loss_r: {:.5f}, Val Loss_h: {:.5f}, Val Acc: {:.4f}".format(epoch, avg_loss_r_train, avg_loss_h_train, accuracy_train, avg_loss_r_val, avg_loss_h_val, accuracy_val))
        trajectories.append([avg_loss_r_train, avg_loss_h_train, accuracy_train, avg_loss_r_val, avg_loss_h_val, accuracy_val])
        if accuracy_val > best_acc:
            best_acc = accuracy_val
            torch.save(model.state_dict(), 'models/model_{}_coeff={}.pt'.format(args.data, args.coeff))
    
    print("Best Val Acc: {:.4f}".format(best_acc))
    with open('results/trajectories_{}_coeff={}.pkl'.format(args.data, args.coeff), 'wb') as f:
        pickle.dump(trajectories, f)