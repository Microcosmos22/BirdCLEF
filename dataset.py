import torch
import numpy as np
import librosa
import random
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T
from pathlib import Path
import matplotlib.pyplot as plt
import librosa.display
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

def df_to_species(df):
    counts = df["primary_label"].value_counts()
    df["species_count"] = df["primary_label"].map(counts)

    label2id = {
        k: v
        for v, k in enumerate(labels)
    }
    """
    print(f"N of species in data: {len(label2id)}")
    print([
        label2id[label]
        for label in df["primary_label"].unique()
    ])"""

    return df, label2id

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

def plot_single_sample(dataset, idx=0):
    """Loads a single raw audio sample, extracts features, and plots everything."""
    # 1. Fetch the metadata row
    row = dataset.df.iloc[idx]
    filepath = Path("..") / "data" / "train_audio" / str(row["filename"])

    # 2. Re-create the transformations step-by-step
    audio = dataset.load_audio(filepath)
    mel_db = dataset.audio_to_mel(audio)

    # 3. Get the PyTorch tensors from the dataset
    mel_tensor, target_tensor = dataset[idx]

    # 4. Set up the plotting grid
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle(f"Sample Species: {row['primary_label']} (ID: {dataset.label2id[row['primary_label']]})", fontsize=16, fontweight='bold')

    # Plot 1: Raw Audio Waveform
    librosa.display.waveshow(audio, sr=SR, ax=axes[0], color='b')
    axes[0].set_title("1. Audio Sequence (Waveform)", fontsize=12)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (seconds)")

    # Plot 2: Mel Spectrogram (dB)
    img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=axes[1], cmap='viridis')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    axes[1].set_title("2. Mel Spectrogram (Decibels)", fontsize=12)

    # Plot 3: PyTorch Feature Image Tensor
    # We remove the channel dimension using .squeeze(0) to plot it
    img_tensor = axes[2].imshow(mel_tensor.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='magma')
    fig.colorbar(img_tensor, ax=axes[2])
    axes[2].set_title(f"3. Feature Image (PyTorch Tensor Shape: {list(mel_tensor.shape)})", fontsize=12)
    axes[2].set_ylabel("Mel Bins")
    axes[2].set_xlabel("Time Frames")

    # Plot 4: One-Hot Encoded Target Vector
    axes[3].plot(target_tensor.numpy(), color='red', linewidth=1.5)
    axes[3].set_title(f"4. One-Hot Encoded Target Vector Y (Total Classes: {len(target_tensor)})", fontsize=12)
    axes[3].set_ylabel("Activation (0 or 1)")
    axes[3].set_xlabel("Class Index ID")
    axes[3].set_xlim(0, len(target_tensor))
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


df = pd.read_csv("../data/train.csv").iloc[:10000]
""" SAFE STRATIFY """
### Count how many "primary_label" occurrences. bool mask >=2. grab only true rows, get their index(primary_label). ####
vc = df["primary_label"].value_counts()
df = df[df["primary_label"].map(vc) > 1]

print(df.shape)
df = df.dropna(subset=["filename", "primary_label"])

NUM_CLASSES = 235

labels = sorted(
    df["primary_label"].unique()
)

"""
converts a 6 digit key into an ID (0-235)
474618 -> 0
181726 -> 1
etc.
"""

df, label2id = df_to_species(df)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify = df["primary_label"]
)

vc = val_df["primary_label"].value_counts()
val_df = val_df[val_df["primary_label"].map(vc) > 1]

val_df, test_df = train_test_split(
    val_df,
    test_size=0.5,
    random_state=42,
    stratify = val_df["primary_label"]
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

test_ds = BirdDataset(
    test_df,
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

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

train_df["latitude"] = train_df["latitude"].round(0)
train_df["longitude"] = train_df["longitude"].round(0)

lat_lon = train_df.groupby(
    ["latitude", "longitude"]
).size()
lat_lon_label = train_df.groupby(
    ["latitude", "longitude", "primary_label"]
).size()

prior = lat_lon_label / lat_lon

global_prior = df["primary_label"].value_counts(normalize=True)


if __name__=="__main__":
    print("Dataset Shapes:")
    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # Call the visualization function here
    print("\nGenerating sample diagnostic plots...")
    plot_single_sample(train_ds, idx=0)
