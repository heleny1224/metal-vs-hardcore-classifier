#!/usr/bin/env python3
"""
Prediction script for single audio files
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torchaudio
from src.models.cnn_gru import SpectrogramCNN_GRUNet
from src.utils.audio_utils import AudioUtil


def load_model(model_path, num_classes=2):
    """Load trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramCNN_GRUNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_audio(model, audio_path, device):
    """Predict class for single audio file"""
    # Load and preprocess audio
    aud = AudioUtil.open(audio_path)
    resample = AudioUtil.resample(aud, 44100)
    rechannel = AudioUtil.rechannel(resample, 2)
    equal_dur = AudioUtil.pad_trunc(rechannel, 5500)
    melspectrogram = AudioUtil.spectro_gram(equal_dur)
    
    # Prepare input
    melspectrogram = melspectrogram.unsqueeze(0).float().to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(melspectrogram)
        probs = torch.softmax(outputs, dim=1)
        pred_label = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    
    return pred_label, confidence


def main():
    parser = argparse.ArgumentParser(description='Predict audio class')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--audio_path', type=str, required=True,
                       help='Path to audio file')
    parser.add_argument('--classes', type=str, default='metal,hardcore',
                       help='Comma-separated list of class names')
    args = parser.parse_args()
    
    # Load model
    class_names = args.classes.split(',')
    model, device = load_model(args.model_path, num_classes=len(class_names))
    
    # Predict
    pred_label, confidence = predict_audio(model, args.audio_path, device)
    
    print(f"Audio: {args.audio_path}")
    print(f"Predicted: {class_names[pred_label]}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == '__main__':
    main()