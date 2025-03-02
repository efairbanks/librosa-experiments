import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
from pathlib import Path
import soundfile as sf

def visualize_audio(y):
  # Display the audio waveform
  plt.figure(figsize=(10, 4))
  librosa.display.waveshow(y)
  plt.title("Audio Waveform")
  plt.show()

def get_audio(filename):
    # Load the audio file
    y, sr = librosa.load(filename)
    
    # Calculate number of samples for 1 second of silence
    silence_samples = sr
    
    # Create silence arrays
    silence = np.zeros(silence_samples)
    
    # Pad the audio with silence at beginning and end
    y_padded = np.concatenate([silence, y, silence])
    
    return y_padded, sr

def get_beat_samples(y, sr):
    hop_length = 512  # Initial hop length for rough detection
    
    # Compute onset envelope with higher temporal precision
    onset_env = librosa.onset.onset_strength(
        y=y, 
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
        center=True
    )
    
    # Get initial beat frames
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        tightness=100,
        trim=True
    )
    
    # Get precise onset times using multiple onset detection functions
    odf_types = ['energy', 'hfc', 'complex']
    onset_backtrack_samples = []
    
    for odf_type in odf_types:
        onset_env_precise = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length//4,  # Higher precision hop length
            aggregate=np.median
        )
        
        onset_frames_precise = librosa.onset.onset_detect(
            onset_envelope=onset_env_precise,
            sr=sr,
            hop_length=hop_length//4,
            delta=0.07,
            wait=2
        )
        
        # Convert to samples with backtracking for precise positions
        onset_samples = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop_length//4,
            units='samples',
            backtrack=True
        )
        
        if len(onset_samples) > 0:
            onset_backtrack_samples.append(onset_samples)
    
    # Combine all onset samples
    if onset_backtrack_samples:
        all_onsets = np.concatenate(onset_backtrack_samples)
        unique_onsets = np.unique(all_onsets)
    else:
        # Fallback to basic frame conversion if no onsets found
        return librosa.frames_to_samples(beat_frames, hop_length=hop_length)
    
    # For each beat frame, find the most likely onset within a small window
    beat_samples = []
    window_size = int(0.05 * sr)  # 50ms window for refinement
    
    for beat_frame in beat_frames:
        beat_sample = librosa.frames_to_samples(beat_frame, hop_length=hop_length)
        
        # Find onsets within the window around the beat
        nearby_onsets = unique_onsets[
            (unique_onsets >= beat_sample - window_size) &
            (unique_onsets <= beat_sample + window_size)
        ]
        
        if len(nearby_onsets) > 0:
            # Find the onset with highest local energy
            energies = []
            for onset in nearby_onsets:
                start = max(0, onset - hop_length//4)
                end = min(len(y), onset + hop_length//4)
                energies.append(np.sum(np.abs(y[start:end])**2))
            
            # Use the onset with highest local energy
            beat_samples.append(nearby_onsets[np.argmax(energies)])
        else:
            # If no nearby onset found, use the original beat position
            beat_samples.append(beat_sample)
    
    beat_samples = np.array(beat_samples)
    
    # Calculate average interval between beats
    if len(beat_samples) > 1:
        intervals = np.diff(beat_samples)
        avg_interval = int(np.mean(intervals))
        
        # Add an additional beat at the average interval from the last beat
        last_beat = beat_samples[-1]
        next_beat = last_beat + avg_interval
        
        # Only add if it's within the signal length
        if next_beat < len(y):
            beat_samples = np.append(beat_samples, next_beat)
    
    return beat_samples

def get_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=20)

def visualize_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.title("MFCC")
    plt.show()

def visualize_beats(y, sr, beat_samples):
    # Convert beat samples to time
    beat_times = librosa.samples_to_time(beat_samples, sr=sr)
    
    # Create the waveform plot
    plt.figure(figsize=(12, 4))
    
    # Plot the waveform
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    
    # Plot beat markers
    plt.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
    
    plt.title('Audio Waveform with Beat Markers')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

def export_beats_to_files(y, sr, beat_samples, audio_file_name):
    # Create output directory based on input filename
    output_dir = Path(f"{Path(audio_file_name).stem}_beats")
    output_dir.mkdir(exist_ok=True)
    
    # Add the end sample
    beat_samples = np.append(beat_samples, len(y))
    
    # Export each beat segment
    for i in range(len(beat_samples) - 1):
        # Get the start and end samples for this beat
        start_sample = beat_samples[i]
        end_sample = beat_samples[i + 1]
        
        # Extract the audio segment
        beat_audio = y[start_sample:end_sample]
        
        # Generate output filename
        output_file = output_dir / f"beat_{i:03d}.wav"
        
        # Save the audio segment
        sf.write(output_file, beat_audio, sr)
    
    print(f"Exported {len(beat_samples)-1} beat segments to {output_dir}/")

def get_beat_chromagrams(y_harmonic, sr, beat_samples):
    # Add the end sample
    beat_samples = np.append(beat_samples, len(y_harmonic))
    
    # List to store chromagrams for each beat
    beat_chromagrams = []
    
    # Compute chromagram for each beat segment
    for i in range(len(beat_samples) - 1):
        # Extract the beat segment
        start_sample = beat_samples[i]
        end_sample = beat_samples[i + 1]
        beat_segment = y_harmonic[start_sample:end_sample]
        
        # Compute chromagram for this beat
        # Using smaller hop length for better temporal resolution within beats
        chroma = librosa.feature.chroma_cens(
            y=beat_segment, 
            sr=sr,
            hop_length=1024
        )
        
        # Average the chromagram over time to get one chroma vector per beat
        beat_chroma = np.mean(chroma, axis=1)
        beat_chromagrams.append(beat_chroma)
    
    return np.array(beat_chromagrams)

def visualize_beat_chromagrams(beat_chromagrams):
    # Create a figure with appropriate size
    plt.figure(figsize=(15, 5))
    
    # Create heatmap of chromagrams
    librosa.display.specshow(
        beat_chromagrams.T,  # Transpose to get pitch classes on y-axis
        y_axis='chroma',
        x_axis='time'
    )
    
    plt.colorbar(label='Magnitude')
    plt.title('Chroma Features per Beat')
    plt.xlabel('Beat Number')
    plt.tight_layout()
    plt.show()

def visualize_beats_and_chroma(y, sr, beat_samples, beat_chromagrams):
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[1, 1])
    
    # Convert beat samples to time for consistent x-axis
    beat_times = librosa.samples_to_time(beat_samples, sr=sr)
    
    # Plot waveform and beats in top subplot
    librosa.display.waveshow(y, sr=sr, alpha=0.5, ax=ax1)
    ax1.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
    ax1.set_title('Audio Waveform with Beat Markers')
    ax1.set_xlabel('')  # Remove x label from top plot
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    
    # Plot chromagram in bottom subplot
    img = librosa.display.specshow(
        beat_chromagrams.T,
        y_axis='chroma',
        x_axis='time',
        x_coords=beat_times,  # Align with beat times
        ax=ax2
    )
    ax2.set_title('Chroma Features per Beat')
    ax2.set_xlabel('Time (s)')
    fig.colorbar(img, ax=ax2, label='Magnitude')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def decimate_beat_samples(beat_samples):
    """Return even-indexed elements from the beat samples array"""
    return beat_samples[::2]

def compute_recurrence_matrix(beat_chromagrams, metric='cosine', threshold=None):
    """
    Compute a recurrence matrix from beat chromagrams.
    
    Parameters:
        beat_chromagrams: numpy array of shape (n_beats, n_chroma)
        metric: distance metric ('cosine', 'euclidean', or 'correlation')
        threshold: optional float to binarize the matrix
    
    Returns:
        rec_matrix: numpy array of shape (n_beats, n_beats)
    """
    from scipy.spatial.distance import cdist
    
    # Compute distance matrix
    if metric == 'cosine':
        # Convert cosine distance to similarity
        dist_matrix = 1 - cdist(beat_chromagrams, beat_chromagrams, metric='cosine')
    elif metric == 'correlation':
        # Convert correlation distance to similarity
        dist_matrix = 1 - cdist(beat_chromagrams, beat_chromagrams, metric='correlation')
    else:
        # For euclidean, normalize and invert distances
        dist_matrix = cdist(beat_chromagrams, beat_chromagrams, metric='euclidean')
        dist_matrix = 1 - (dist_matrix / dist_matrix.max())
    
    # Apply threshold if specified
    if threshold is not None:
        rec_matrix = (dist_matrix >= threshold).astype(float)
    else:
        rec_matrix = dist_matrix
    
    return rec_matrix

def visualize_recurrence_matrix(rec_matrix, beat_times=None):
    """
    Visualize the recurrence matrix.
    
    Parameters:
        rec_matrix: numpy array of shape (n_beats, n_beats)
        beat_times: optional array of beat times in seconds
    """
    plt.figure(figsize=(10, 10))
    
    if beat_times is not None:
        plt.imshow(rec_matrix, aspect='equal', origin='lower', 
                  extent=[beat_times[0], beat_times[-1], beat_times[0], beat_times[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Time (s)')
    else:
        plt.imshow(rec_matrix, aspect='equal', origin='lower')
        plt.xlabel('Beat index')
        plt.ylabel('Beat index')
    
    plt.colorbar(label='Similarity')
    plt.title('Beat-Synchronous Recurrence Matrix')
    plt.tight_layout()
    plt.show()

def visualize_combined_analysis(y, sr, beat_samples, beat_chromagrams, rec_matrix):
    """
    Create a combined visualization of the waveform, chromagram, and recurrence matrix.
    
    Parameters:
        y: audio signal
        sr: sample rate
        beat_samples: array of beat positions in samples
        beat_chromagrams: array of chromagrams for each beat
        rec_matrix: recurrence matrix
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 2])
    
    # Plot waveform with beat markers
    ax0 = plt.subplot(gs[0, :])
    # Convert samples to time for both waveform and beats
    times = np.arange(len(y)) / sr
    beat_times = beat_samples / sr
    
    ax0.plot(times, y, color='gray', alpha=0.5)
    ax0.vlines(beat_times, -1, 1, color='r', alpha=0.5, label='Beats')
    ax0.set_xlim(0, len(y) / sr)
    ax0.set_ylim(-1, 1)
    ax0.set_title('Waveform and Beats')
    ax0.set_xlabel('Time (s)')
    ax0.legend()
    
    # Plot chromagram
    ax1 = plt.subplot(gs[1, 0])
    img = librosa.display.specshow(beat_chromagrams.T, 
                                 x_coords=beat_times,
                                 y_axis='chroma',
                                 x_axis='time',
                                 ax=ax1)
    ax1.set_title('Beat-Synchronous Chromagram')
    plt.colorbar(img, ax=ax1, format='%+2.0f dB')
    
    # Plot recurrence matrix
    ax2 = plt.subplot(gs[1, 1])
    img2 = ax2.imshow(rec_matrix, aspect='equal', origin='lower',
                     extent=[beat_times[0], beat_times[-1], beat_times[0], beat_times[-1]])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Beat-Synchronous Recurrence Matrix')
    plt.colorbar(img2, ax=ax2, label='Similarity')
    
    plt.tight_layout()
    plt.show()

def main():
    # Replace with your audio file path
    audio_file_name = "PORTA.wav"  # or .wav
    try:
        y, sr = get_audio(audio_file_name)
        beat_samples = get_beat_samples(y, sr)
        beat_samples = decimate_beat_samples(beat_samples)
        # beat_samples = decimate_beat_samples(beat_samples)
        # beat_samples = decimate_beat_samples(beat_samples)

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        beat_chromagrams = get_beat_chromagrams(y, sr, beat_samples)
        
        # Compute recurrence matrix
        rec_matrix = compute_recurrence_matrix(beat_chromagrams, metric='cosine')
        
        # Export beats to files
        export_beats_to_files(y, sr, beat_samples, audio_file_name)

        # Create combined visualization
        visualize_combined_analysis(y, sr, beat_samples, beat_chromagrams, rec_matrix)

    except Exception as e:
        print(f"Error analyzing file: {e}")


if __name__ == "__main__":
    main()