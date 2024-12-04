import os
import sys
from matplotlib import colors, pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import re

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


from utils.dataset import NNgTLDataset
from utils.model import BA_Predict, LTL_Net
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 13

model_scale_dict = {
    'tiny': 8,
    'small': 16,
    'medium': 32,
    'large': 64
}

CE = torch.nn.CrossEntropyLoss()
BCE = torch.nn.BCEWithLogitsLoss()
L1 = torch.nn.L1Loss()


def dice_loss(y_pred, y_true, epsilon=1e-5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return 1 - (2. * intersection + epsilon) / (y_pred.sum() + y_true.sum() + epsilon)


def loss_fn(out1, target, model_type='BA_Predict', alpha=1, print_loss=False):
    if model_type == 'BA_Predict':
        loss = BCE(out1, target)
    elif model_type == 'LTL_Net':
        loss1 = BCE(out1, target)
        loss2 = dice_loss(out1, target)
        loss = alpha * loss1 + (1 - alpha) * loss2
    else:
        raise ValueError('Invalid model type')
    if print_loss:
        print('Loss:', loss.item())
    return loss


def save_model(model, save_dir, epoch, model_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_{epoch}.pth'))

def save_checkpoint(model, optimizer, epoch, best_loss, best_epoch, checkpoint_dir, model_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint = {
        'epoch': epoch,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_checkpoint_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

def train_model(model, loader, optimizer, model_type='BA_Predict', alpha=1):
    model.train()
    total_loss = 0
    for batch, data in enumerate(loader):
        print_loss = False
        data.to(device)
        optimizer.zero_grad()
        out1 = model(data.map, data.x_dict, data.edge_index_dict, data.batch_dict)
        if model_type == 'BA_Predict':
            target = data.node_label
        elif model_type == 'LTL_Net':
            target = data.label
        else:
            raise ValueError('Invalid model type')
        if batch == 0:
            print_loss = True
        loss = loss_fn(out1, target, model_type=model_type, alpha=alpha, print_loss=print_loss)
        if batch % 50 == 0:
            print(f'Batch: {batch}, Loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_model(model, loader, model_type='BA_Predict', alpha=1):
    model.eval()
    total_loss = 0
    for batch, data in enumerate(loader):
        data.to(device)
        with torch.no_grad():
            out1 = model(data.map, data.x_dict, data.edge_index_dict, data.batch_dict)
            if model_type == 'BA_Predict':
                target = data.node_label
            elif model_type == 'LTL_Net':
                target = data.label
            else:
                raise ValueError('Invalid model type')
            loss = loss_fn(out1, target, model_type=model_type, alpha=alpha)
            total_loss += loss.item()
            
    return total_loss / len(loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with different scenarios')
    parser.add_argument('-m', '--model', type=str, choices=['BA_Predict', 'LTL_Net'], default='BA_Predict',
                        help='Model to train: BA_Predict or LTL_Net')
    parser.add_argument('-s', '--scale', type = str, choices=['tiny', 'small','medium','large'], default='large',help='Scale of the model: tiny, small, medium, large')
    parser.add_argument('-t', '--type', type=int, choices=[1, 2, 3], default=1,
                        help='1: Train from scratch, 2: Continue from checkpoint, 3: Transfer learning')
    parser.add_argument('-c', '--checkpoint', type=str, default="./model/checkpoint/",
                        help='Path to the checkpoint file')
    parser.add_argument('-p', '--pretrained_model', type=str, default="./model/pretrained_model/",
                        help='Path to the pretrained model')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('-i', '--save_interval', type=int, default=10,
                        help='Save checkpoint every n epochs')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    training_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data'))
    datalist = NNgTLDataset(training_data_path)
    print(f"Dataset size: {len(datalist)}")

    data = datalist[0]

    if os.path.exists('train_test_split.pkl'):
        with open('train_test_split.pkl', 'rb') as f:
            train_ids, test_ids = pickle.load(f)
        print("Loaded dataset split from file")
        print(f"Train size: {len(train_ids)}, Test size: {len(test_ids)}")
    else:
        train_ids, test_ids = train_test_split(range(len(datalist)), test_size=0.2, random_state=random_seed)
        with open('train_test_split.pkl', 'wb') as f:
            pickle.dump((train_ids, test_ids), f)

    n1 = model_scale_dict[args.scale]

    models = {
        'BA_Predict': BA_Predict(data.metadata()).to(device),
        'LTL_Net': LTL_Net(data.metadata(), n1).to(device)
    }

    optimizers = {
        'BA_Predict': torch.optim.Adam(models['BA_Predict'].parameters(), lr=1e-5),
        'LTL_Net': torch.optim.Adam(models['LTL_Net'].parameters(), lr=1e-5)
    }

    schedulers = {
        'BA_Predict': ReduceLROnPlateau(optimizers['BA_Predict'], mode='min', factor=0.1, patience=10),
        'LTL_Net': ReduceLROnPlateau(optimizers['LTL_Net'], mode='min', factor=0.1, patience=10)
    }

    train_dataset = Subset(datalist, train_ids)
    test_dataset = Subset(datalist, test_ids)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    
    model_name = args.model
    print(f'Start training {model_name}')
    model = models[model_name]
    optimizer = optimizers[model_name]
    scheduler = schedulers[model_name]
    start_epoch = 0
    best_loss = float('inf')
    if args.type == 2 and args.checkpoint:
        if os.path.isdir(args.checkpoint):
            checkpoint_path = os.path.join(args.checkpoint, model_name)
            # Find the latest checkpoint file in the directory
            checkpoint_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                checkpoint = torch.load(latest_checkpoint)
                print(f'Loaded checkpoint from {latest_checkpoint}')
            else:
                print("No checkpoint files found in the directory.")
                exit(1)
        else:
            checkpoint = torch.load(args.checkpoint)
            print(f'Loaded checkpoint from {args.checkpoint}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    elif args.type == 3 and args.pretrained_model:
        if os.path.isdir(args.pretrained_model):
            pretrained_model_path = os.path.join(args.pretrained_model, model_name)
            # Find the latest pretrained model file in the directory
            model_files = [os.path.join(pretrained_model_path, f) for f in os.listdir(pretrained_model_path) if f.endswith('.pth')]
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                pretrained_model = torch.load(latest_model)
                print(f'Loaded pretrained model from {latest_model}')
            else:
                print("No pretrained model files found in the directory.")
                exit(1)
        else:
            pretrained_model = torch.load(args.pretrained_model)
            print(f'Loaded pretrained model from {args.pretrained_model}')
        model.load_state_dict(pretrained_model)

    for epoch in range(start_epoch, args.epochs):
        total_loss = train_model(model, train_loader, optimizer, model_type=model_name)
        print(f'{model_name} Epoch: {epoch}, Training Loss: {total_loss}')
        test_loss = test_model(model, test_loader, model_type=model_name)
        print(f'{model_name} Epoch: {epoch}, Testing Loss: {test_loss}')

        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'{model_name} Epoch: {epoch}, Current Learning Rate: {current_lr}')

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            save_model(model, os.path.join('./model/', model_name), epoch, model_name + '_' + args.scale)
            print(f'New best model saved at epoch {epoch}')
        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, best_loss, best_epoch, os.path.join('./model/checkpoint/', model_name), model_name + '_' + args.scale)
            print(f'Checkpoint saved at epoch {epoch}')
        print(f'{model_name} Best Epoch: {best_epoch}, Best Loss: {best_loss}')
        torch.cuda.empty_cache()
            
