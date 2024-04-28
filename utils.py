import torch
import numpy as np 

# relabel labels (fill in gaps)
def relabel(labels):
    unique_labels = np.unique(labels)
    new_labels = np.zeros(labels.shape)
    for i, label in enumerate(unique_labels):
        new_labels[labels == label] = i
    return new_labels.astype(int)

# Model evaluation function
def eval_model(model, data_loader, device, pad_len, criterion_r, criterion_h):
    model.to(device)
    model.eval()
    losses_r = []
    losses_h = []
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for _, (data, labels) in enumerate(data_loader):
            cal_loss_mask = torch.ones_like(data)# do not calculate losss for padded portions
            cal_loss_mask[:, :, -pad_len:] = 0
            cal_loss_mask = cal_loss_mask.to(device)
            non_pad_eles = data.shape[-1] - pad_len
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device)
            r, h, h0, b = model(data)
            
            # Compute accuracy using h (logits) and labels
            _, predicted = torch.max(h, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.shape[0]

            loss_r = criterion_r(r, data)
            loss_r = loss_r * cal_loss_mask
            loss_r = loss_r.squeeze().sum() / non_pad_eles
            loss_h = criterion_h(h, labels)
            losses_r.append(loss_r.detach().cpu().item())
            losses_h.append(loss_h.detach().cpu().item())
        avg_loss_r = sum(losses_r) / len(losses_r)
        avg_loss_h = sum(losses_h) / len(losses_h)
        accuracy = total_correct / total_samples
        
    return avg_loss_r, avg_loss_h, accuracy

def eval_baseline(model, data_loader, device, criterion_h):
    model.to(device)
    model.eval()
    losses_h = []
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for _, (data, labels) in enumerate(data_loader):
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device)
            h, h0, b = model(data)
            
            # Compute accuracy using h (logits) and labels
            _, predicted = torch.max(h, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.shape[0]

            loss_h = criterion_h(h, labels)
            losses_h.append(loss_h.detach().cpu().item())
        avg_loss_h = sum(losses_h) / len(losses_h)
        accuracy = total_correct / total_samples
        
    return avg_loss_h, accuracy