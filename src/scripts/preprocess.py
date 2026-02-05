#!/usr/bin/env python3
"""
Preprocessing script for audio data
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.preprocessor import split_audio_files, create_dataframe_from_directory


def main():
    parser = argparse.ArgumentParser(description='Preprocess audio data')
    parser.add_argument('--input_dir', type=str, default='./data_music',
                       help='Input directory with raw audio')
    parser.add_argument('--output_dir', type=str, default='./data_split',
                       help='Output directory for processed data')
    parser.add_argument('--num_segments', type=int, default=60,
                       help='Number of segments per audio file')
    args = parser.parse_args()
    
    print(f"Preprocessing audio from {args.input_dir}...")
    
    # Split audio files
    split_audio_files(args.input_dir, args.output_dir, args.num_segments)
    
    # Create and save dataframe
    df = create_dataframe_from_directory(args.output_dir)
    print(f"Created dataframe with {len(df)} samples")
    print(f"Class distribution:\n{df['Genre'].value_counts()}")
    
    # Save dataframe
    df.to_csv(os.path.join(args.output_dir, 'metadata.csv'), index=False)
    print(f"Dataframe saved to {os.path.join(args.output_dir, 'metadata.csv')}")


if __name__ == '__main__':
    main()