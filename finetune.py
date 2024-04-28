import os, pickle, torch, argparse
import os.path as osp
import torch.nn as nn
import numpy as np
from model import *
from torch.utils.data import TensorDataset, DataLoader
from utils import relabel, eval_model
from sklearn.model_selection import train_test_split
from rich.progress import track

cur_dir = osp.abspath(osp.dirname(__file__))
os.chdir(cur_dir)

SEED=56
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-split", default=0.6, type=float)
    parser.add_argument("--bsz", default=128, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--coeff", default='1.0', choices = ['0.01', '1.0', '100.0'], type=str, help='coefficient for classification loss')
    parser.add_argument("--epochs", default=120, type=int)


    parser.add_argument("--classifier-channels", default=1, type=int)

    parser.add_argument("--device", choices=[-1,0,1,2,3], default=1, type=int, help='device number (-1 for cpu)')
    parser.add_argument("--task", default='st', type=str, choices=['st', 'cluster'], help='task to fine-tune on')
    parser.add_argument("--free", default='free-b-cls', type=str, choices=['free-b-cls', 'free-b-pcls-cls', 'free-all', 'free-enc-b-pcls-cls'], help='which layers NOT to freeze')

    args=parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    model_path = 'models/model_preprocessed_coeff={}.pt'.format(args.coeff)


    # get num_labels for model instantiation
    data_full = np.load('../DRIAMS/DRIAMS-BC-preprocessed-nonsingleton.npy', allow_pickle=True)   #pretraining data
    old_labels = data_full[:, -1]
    labels = relabel(old_labels)
    new_old_label_dict = {int(new_label): int(old_label) for new_label, old_label in zip(labels, old_labels)}
    num_labels = len(set(labels))

    
    
    # load fine-tune specifics
    with open('../multi_species_data/multi_species_{}.pkl'.format(args.task), 'rb') as f:
        label_dict, label_nas = pickle.load(f)
    
    with open('../multi_species_data/multi_species_{}.txt'.format(args.task), 'r') as f:
        label_strs =  [(x.split('\t')[0], x.split('\t')[1].strip()) for x in f.readlines()]
    
    labels = np.array([label_dict[x] for x in label_strs])

    # check for and sift out singletons
    counts = np.bincount(labels)
    singletons = np.where(counts == 1)[0]
    non_singleton_indices = [i for i in range(len(labels)) if labels[i] not in singletons]
    singleton_indices = [i for i in range(len(labels)) if labels[i] in singletons]
    print("nonsingle {}, single {}".format(len(non_singleton_indices), len(singleton_indices)))

    # load fine tune data
    data = np.load('../multi_species_data/multi_species_data.npy', allow_pickle=True)


    pad_len = 32 - data.shape[-1] % 32
    if pad_len > 0:
        pad_data = np.zeros((data.shape[0], pad_len))
        data = np.concatenate((data, pad_data), axis=1)
    data = data[:, np.newaxis, :]
    print("fine tune data shape {}".format(data.shape))

    # create datasets and dataloaders
    bsz = args.bsz
    X_train, X_val, y_train, y_val = train_test_split(data[non_singleton_indices], labels[non_singleton_indices], train_size=args.data_split, stratify=labels[non_singleton_indices], random_state=56)
    np.save('xtrain_{}.npy'.format(args.task), X_train)
    np.save('ytrain_{}.npy'.format(args.task), y_train)
    np.save('xval_{}.npy'.format(args.task), X_val)
    np.save('yval_{}.npy'.format(args.task), y_val)
    assert 1==0
    print('train {}; val {}'.format(len(y_train), len(y_val)))
    
    data_tensor_train = torch.from_numpy(X_train)
    labels_tensor_train = torch.from_numpy(y_train)
    data_tensor_val = torch.from_numpy(X_val)
    labels_tensor_val = torch.from_numpy(y_val)

    train_dataset = TensorDataset(data_tensor_train, labels_tensor_train)
    val_dataset = TensorDataset(data_tensor_val, labels_tensor_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=bsz, shuffle=False)

    # load pretrained model
    model = unet(num_classes=num_labels)
    model.load_state_dict(torch.load(model_path))

    #! Freeze all layers in the model
    if args.free != 'free-all':
        for param in model.parameters():
            param.requires_grad = False
        # Un-freeze the bottleneck
        for param in model.b.parameters():
            param.requires_grad = True
        if 'pcls' in args.free:
            for param in model.pre_classifier.parameters():
                param.requires_grad = True
        if 'enc' in args.free:
            for param in model.encoder_blocks.parameters():
                param.requires_grad = True
    
    # change the classifier layer
    model.num_classes = len(label_dict)
    A = model.classifier
    B = nn.Linear(A.in_features, len(label_dict))
    model.classifier = B
    print('new number of classes: {}'.format(len(label_dict)))


    epochs = args.epochs
    lr = args.lr
    weight_decay=args.weight_decay
    coeff = args.coeff
    device = torch.device("cuda:{}".format(args.device)) if args.device >= 0 else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_r = nn.MSELoss(reduction='none')
    criterion_h = nn.CrossEntropyLoss()


    # start fine-tuning
    trajectories = []
    best_acc = 0
    for epoch in track(range(epochs), description='fine tuning network'):
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
            train_loss = loss_r + loss_h
            train_loss.backward()
            optimizer.step()
            losses_r.append(loss_r.detach().cpu().item())
            losses_h.append(loss_h.detach().cpu().item())

            # print_params(model.classifier)
            # print_params(model.encoder_blocks[0])

        avg_loss_r_train = sum(losses_r) / len(losses_r)
        avg_loss_h_train = sum(losses_h) / len(losses_h)
        accuracy_train = total_correct / total_samples
        avg_loss_r_val, avg_loss_h_val, accuracy_val = eval_model(model, val_loader, device, pad_len, criterion_r, criterion_h)
        print("Epoch {},  Train Loss_r: {:.5f}, Train Loss_h: {:.5f}, Train Acc: {:.4f},\
            Val Loss_r: {:.5f}, Val Loss_h: {:.5f}, Val Acc: {:.4f}".format(epoch, avg_loss_r_train, avg_loss_h_train, accuracy_train, avg_loss_r_val, avg_loss_h_val, accuracy_val))
        trajectories.append([avg_loss_r_train, avg_loss_h_train, accuracy_train, avg_loss_r_val, avg_loss_h_val, accuracy_val])
        if accuracy_val > best_acc:
            best_acc = accuracy_val
            torch.save(model.state_dict(), 'models/finetune_{}_{}_{}.pt'.format(args.task, args.coeff, args.free))
    with open('results/trajectories_finetune_{}_{}_{}.pkl'.format(args.task, args.coeff, args.free), 'wb') as f:
        pickle.dump(trajectories, f)      
