"""
Audio preprocessing and splitting utilities
"""
import os
import torchaudio
import pandas as pd


def split_audio(audio_file, output_dir, num_segments=5, original_name=""):
    """Split audio file into segments"""
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
    except RuntimeError:
        print(f"Failed to load: {audio_file}")
        return

    # Segment length calculation
    total_duration = waveform.size(1) / sample_rate
    segment_duration = total_duration / num_segments
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the audio into segments
    for i in range(num_segments):
        start = int(i * segment_duration * sample_rate)
        end = int((i + 1) * segment_duration * sample_rate)
        if end > waveform.size(1):
            end = waveform.size(1)
        
        segment_waveform = waveform[:, start:end]
        
        # Construct unique name for the segment
        segment_name = f"{original_name}_segment_{i + 1}.wav"
        output_file = os.path.join(output_dir, segment_name)
        
        torchaudio.save(output_file, segment_waveform, sample_rate)


def split_audio_files(input_dir, output_dir, num_segments=5):
    """Split all audio files in directory"""
    # Iterate through each class folder
    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        if os.path.isdir(class_path):
            # Create corresponding output class folder
            output_class_path = os.path.join(output_dir, class_folder)
            os.makedirs(output_class_path, exist_ok=True)
            
            # Iterate through audio files
            for audio_file in os.listdir(class_path):
                audio_file_path = os.path.join(class_path, audio_file)
                if os.path.isfile(audio_file_path):
                    # Split the audio file
                    split_audio(audio_file_path, output_class_path, 
                               num_segments, audio_file)


def create_dataframe_from_directory(data_dir):
    """Create dataframe from directory structure"""
    file_genre = []
    file_path = []
    
    # Remove .DS_Store files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))
    
    # Create dataframe
    for folder in os.listdir(data_dir):
        files_path = os.path.join(data_dir, folder)
        if os.path.isdir(files_path):
            for audio in os.listdir(files_path):
                file_genre.append(folder)
                file_path.append(os.path.join(files_path, audio))
    
    genre_df = pd.DataFrame(file_genre, columns=["Genre"])
    path_df = pd.DataFrame(file_path, columns=["Path"])
    gtzan_df = pd.concat([genre_df, path_df], axis=1)
    
    return gtzan_df