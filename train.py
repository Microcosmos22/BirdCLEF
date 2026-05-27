import os
import random

import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import BirdModel
from dataset import *

from datetime import datetime


"""
ARCHITECTURE:
- Audio -> Mel Spectr.
- Augmentation(time masking, freq. masking, random gain, background noisemix, mixup, time shift)

2. CNN Encoder + attention pooling
→ EfficientNet-B0 / ConvNeXt-Tiny
→ temporal attention pooling
→ multilabel classifier
"""

import matplotlib.pyplot as plt

if __name__=="__main__":
    print("train")
    DEVICE = "cpu"
    model = BirdModel(NUM_CLASSES)
    model = model.to(DEVICE)
    """model.load_state_dict(
        torch.load("../bird_model20260526_195539.pth")
    )"""

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )
    losses = []
    val = []

    for epoch in range(200):

        model.train()
        train_loss = 0

        for x, y in tqdm(train_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses.append(loss.item())

            model.eval()
            val_loss = 0
            with torch.no_grad():

                for x, y in val_loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val.append(loss.item())

        train_loss /= len(train_loader)

        if epoch % 3 == 0:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(
                model.state_dict(),
                "../bird_model"+timestamp+".pth"
            )

            plt.plot(losses)
            plt.plot(val)
            plt.savefig("plot.png")

    print(
        f"Epoch {epoch+1} | "
        f"Train {train_loss:.4f} | ")
