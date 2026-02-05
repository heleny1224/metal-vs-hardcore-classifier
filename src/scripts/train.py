#!/usr/bin/env python3
"""
Training script for CNN-GRU model
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.preprocessor import create_dataframe_from_directory
from src.data.dataset import AudioDataset
from src.models.cnn_gru import SpectrogramCNN_GRUNet


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs = inputs.float().to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(train_loader.dataset)


def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def main():
    parser = argparse.ArgumentParser(description='Train CNN-GRU model')
    parser.add_argument('--data_dir', type=str, default='./data_split',
                       help='Directory with processed data')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./weights',
                       help='Directory to save model weights')
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and split data
    df = create_dataframe_from_directory(args.data_dir)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_df)
    val_dataset = AudioDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    num_classes = len(df['Genre'].unique())
    model = SpectrogramCNN_GRUNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy = validate(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Training Loss: {train_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2%}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                model.state_dict(), 
                os.path.join(args.output_dir, 'DM_gtzan_best.pth')
            )
            print(f"Saved best model with accuracy: {val_accuracy:.2%}")
        
        # Save latest model
        torch.save(
            model.state_dict(),
            os.path.join(args.output_dir, 'DM_gtzan_latest.pth')
        )
    
    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == '__main__':
    main()