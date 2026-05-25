"""Simple training script for attribute prediction using PyTorch.

Trains a torchvision backbone with a linear head for multi-label
classification using BCEWithLogitsLoss.
"""
import argparse
import os
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import average_precision_score
import numpy as np

from vispr.datasets.pap_dataset import PAPDataset


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    arch = arch.lower()
    if arch.startswith('resnet'):
        model = getattr(models, arch)(pretrained=pretrained)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        return model
    else:
        raise ValueError('Unsupported arch: {}'.format(arch))


def train_one_epoch(model, device, loader, optimizer, criterion, epoch, log_interval=50):
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device).float()
        target = target.to(device).float()
        optimizer.zero_grad()
        outputs = model(data)
        # _, predicted = torch.max(outputs.data, 1)
        # running_correct += (predicted == target.data).sum()

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {running_loss / (batch_idx+1):.4f}')
    train_loss = running_loss / len(loader)
    # train_acc = running_correct / len(loader)
    return train_loss


def validate(model, device, loader):
    model.eval()
    ys = []
    ys_pred = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device).float()
            outputs = model(data)
            probs = torch.sigmoid(outputs).cpu().numpy()
            ys_pred.append(probs)
            ys.append(target.numpy())
    ys = np.vstack(ys)
    ys_pred = np.vstack(ys_pred)
    # compute per-class average precision
    n_classes = ys.shape[1]
    ap_list = []
    for c in range(n_classes):
        try:
            ap = average_precision_score(ys[:, c], ys_pred[:, c])
        except Exception:
            ap = float('nan')
        ap_list.append(ap)
    mean_ap = np.nanmean(ap_list)
    return mean_ap, ap_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True, help='Annotation list (one JSON per line or path list)')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int, default=68)
    parser.add_argument('--save-path', default='model_last.pth')
    parser.add_argument('--valfile', default=None, help='Validation annotation list (optional)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PAPDataset(args.infile, im_shape=(224, 224))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print('No. of samples in train set: '+ str(len(loader.dataset)))

    model = build_model(args.arch, args.num_classes, pretrained=args.pretrained).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = math.inf
    val_loader = None
    if args.valfile:
        val_dataset = PAPDataset(args.valfile, im_shape=(224, 224))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print('No. of samples in validation set: ' + str(len(val_loader.dataset)))

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, device, loader, optimizer, criterion, epoch)
        print(f'Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}')
        # Save checkpoint each epoch
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, args.save_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.splitext(args.save_path)[0] + '_best.pth'
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_path)
            print('Saved best model to', best_path)

        if val_loader is not None:
            mean_ap, ap_list = validate(model, device, val_loader)
            print(f'Validation mAP: {mean_ap:.4f}')


if __name__ == '__main__':
    main()

