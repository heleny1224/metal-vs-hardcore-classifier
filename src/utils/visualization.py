"""
Visualization utilities for audio and spectrograms
"""
import matplotlib.pyplot as plt
import torch


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    """Plot audio waveform"""
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    """Plot spectrogram"""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")


def show_images_from_loader(data_loader, num_images=16, class_names=None):
    """Display spectrograms from data loader"""
    import numpy as np
    
    # Randomly select num_images indices
    indices = np.random.choice(len(data_loader.dataset), num_images, replace=False)
    
    # Extract spectrograms and labels
    spectrograms = []
    labels = []
    for idx in indices:
        spectrogram, label = data_loader.dataset[idx]
        spectrogram_data = spectrogram[0]
        spectrograms.append(spectrogram_data)
        labels.append(label)

    # Create a grid of images
    rows = int(np.ceil(num_images / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3 * rows))

    for i, (specgram, label) in enumerate(zip(spectrograms, labels)):
        ax = axes[i // 4, i % 4] if rows > 1 else axes[i % 4]
        title = f"Label: {class_names[label]}" if class_names else f"Label: {label}"
        plot_spectrogram(specgram, title=title, ax=ax)
    plt.tight_layout()
    plt.show()