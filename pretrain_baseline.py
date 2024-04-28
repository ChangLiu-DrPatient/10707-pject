import os, pickle, torch, argparse
import os.path as osp
import torch.nn as nn
import numpy as np
from rich.progress import track
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import baseline_mlp, baseline_cnn
from utils import eval_baseline, relabel
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
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--baseline", default='cnn', choices=['mlp', 'cnn'], type=str, help='baseline model')
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
        data = data_full[:, :-1] / np.sum(data_full[:, :-1], axis=1, keepdims=True)
        labels = data_full[:, -1]
    
    num_labels = len(set(labels))
    labels = relabel(labels)
    # print(labels)
    # assert 1==0


    print("Data Shape {}; Labels Shape {}; Num Labels {}".format(data.shape, labels.shape, num_labels))   
    if args.baseline == 'cnn':
        pad_len = 32 - data.shape[-1] % 32
        if pad_len > 0:
            pad_data = np.zeros((data.shape[0], pad_len))
            data = np.concatenate((data, pad_data), axis=1)
        data = data[:, np.newaxis, :]
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
    if args.baseline == 'mlp':
        model = baseline_mlp(num_classes=num_labels, vec_len=data.shape[-1])
    if args.baseline == 'cnn':
        model = baseline_cnn(num_classes=num_labels, vec_len=data.shape[-1])
    # my_input = torch.randn((1, 1, 1024))
    # outputs, h = model(my_input)
    # print(outputs.shape, h.shape)

    # train and evaluate
    epochs = args.epochs
    lr = args.lr
    weight_decay=args.weight_decay
    device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion_h = nn.CrossEntropyLoss()

    trajectories = []
    best_acc = 0
    for epoch in track(range(epochs), description='training network'):
        losses_h = []
        total_correct = 0
        total_samples = 0
        for _, (data, targets) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)

            optimizer.zero_grad()
            model.train()
            h, h0, b = model(data)

            _, predicted = torch.max(h, 1)
            correct = (predicted == targets).sum().item()
            total_correct += correct
            total_samples += targets.shape[0]
            
            # print(type(h), type(targets))
            train_loss = criterion_h(h, targets)
            train_loss.backward()
            optimizer.step()
            losses_h.append(train_loss.detach().cpu().item())

        avg_loss_h_train = sum(losses_h) / len(losses_h)
        accuracy_train = total_correct / total_samples
        avg_loss_h_val, accuracy_val = eval_baseline(model, val_loader, device, criterion_h)
        print("Epoch {},  Train Loss: {:.5f}, Train Acc: {:.4f},\
             Val Loss: {:.5f}, Val Acc: {:.4f}".format(epoch, avg_loss_h_train, accuracy_train, avg_loss_h_val, accuracy_val))
        trajectories.append([avg_loss_h_train, accuracy_train, avg_loss_h_val, accuracy_val])
        if accuracy_val > best_acc:
            best_acc = accuracy_val
            torch.save(model.state_dict(), 'models/baseline_{}_{}.pt'.format(args.baseline, args.data))
    print("Best Val Acc: {:.4f}".format(best_acc))
    with open('results/trajectory_baseline_{}_{}.pkl'.format(args.baseline, args.data), 'wb') as f:
        pickle.dump(trajectories, f)