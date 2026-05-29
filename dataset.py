import torch
import numpy as np
import librosa
import random
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T
from pathlib import Path

SR = 32000

DURATION = 5
SAMPLES = SR * DURATION

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4



noise_files = [
    "noise/rain.wav",
    "noise/wind.wav"
]

class BirdDataset(Dataset):

    def __init__(self, df, label2id, train=True):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.train = train

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=SR)

        if len(audio) < SAMPLES:
            pad = SAMPLES - len(audio)
            audio = np.pad(audio, (0, pad))

        else:
            start = random.randint(0,len(audio) - SAMPLES)
            audio = audio[start:start + SAMPLES]
        return audio

    def audio_to_mel(self, audio):

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel = librosa.power_to_db(mel)

        return mel.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if not pd.isna(row["filename"]):
            filepath = Path("..") / Path("data") / Path("train_audio") / Path(str(row["filename"]))

            audio = self.load_audio(filepath)

            if self.train:
                """audio = wave_augment(
                    samples=audio,
                    sample_rate=SR
                )"""

            """
            X = mel melspectrogram
            Y = bird label / target
            """

            mel = self.audio_to_mel(audio)
            mel = torch.tensor(mel).unsqueeze(0)
            target = np.zeros(235)
            target[self.label2id[row["primary_label"]]] = 1

            target = torch.tensor(target).float()
        else:
            return torch.empty((0), dtype=torch.float32), torch.empty((0), dtype=torch.float32)

        return mel, target


df = pd.read_csv("../data/train.csv").iloc[:10000]
print(df.shape)
df = df.dropna(subset=["filename", "primary_label"])


labels = sorted(
    df["primary_label"].unique()
)


counts = df["primary_label"].value_counts()
df["species_count"] = df["primary_label"].map(counts)
#df = df[df["species_count"] >= 3]

"""
converts a 6 digit key into an ID (0-235)
474618 -> 0
181726 -> 1
etc.
"""
label2id = {
    k: v
    for v, k in enumerate(labels)
}

NUM_CLASSES = 235


print(f"N of species in data: {len(label2id)}")
print([
    label2id[label]
    for label in df["primary_label"].unique()
])

train_df, val_df = train_test_split(
    df,
    test_size=0.5,
    random_state=42,
    stratify = df["primary_label"]
)

train_ds = BirdDataset(
    train_df,
    label2id,
    train=True
)

val_ds = BirdDataset(
    val_df,
    label2id,
    train=False
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

if __name__=="__main__":
    print("dataset")

    labels = sorted(
        df["primary_label"].unique()
    )
    print(label2id)
