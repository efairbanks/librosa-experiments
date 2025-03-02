import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

class MusicAnalyzer:
    def __init__(self, audio_path):
        """Initialize the music analyzer with an audio file."""
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path)
        self.tempo = None
        self.key = None
        self.beats = None
        self.onset_env = None
        self.onset_frames = None
        
    def analyze_tempo(self):
        """Analyze and return the tempo of the track."""
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.tempo, self.beats = librosa.beat.beat_track(onset_envelope=self.onset_env, sr=self.sr)
        return self.tempo
    
    def analyze_key(self):
        """Analyze and return the musical key of the track."""
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chroma_avg = np.mean(chroma, axis=1)
        self.key = key_names[np.argmax(chroma_avg)]
        return self.key, chroma
    
    def analyze_onsets(self):
        """Detect note onsets in the track."""
        if self.onset_env is None:
            self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.onset_frames = librosa.onset.onset_detect(onset_envelope=self.onset_env, sr=self.sr)
        return librosa.frames_to_time(self.onset_frames, sr=self.sr)
    
    def analyze_pitch(self):
        """Analyze pitch content using pitch class profiles."""
        harmonics = librosa.effects.harmonic(self.y)
        pitches, magnitudes = librosa.piptrack(y=harmonics, sr=self.sr)
        return pitches, magnitudes
    
    def analyze_structure(self):
        """Analyze the structural segments of the track using advanced features."""
        # Compute MFCCs with more coefficients for better timbral representation
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=20)
        mfcc = librosa.util.normalize(mfcc, axis=1)

        # Compute tempogram for rhythmic structure
        oenv = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sr)
        
        # Ensure tempogram has the same number of time frames as MFCC
        if tempogram.shape[1] != mfcc.shape[1]:
            # Resample tempogram to match MFCC time resolution
            tempogram = librosa.util.fix_length(tempogram, size=mfcc.shape[1], axis=1)
        
        # Get rhythmic features (average over tempo bins for each time frame)
        tempo_features = np.mean(tempogram, axis=0, keepdims=True)
        tempo_features = librosa.util.normalize(tempo_features)
        
        # Combine MFCC and rhythmic features
        features = np.vstack([mfcc, tempo_features])
        
        # Enhanced recurrence matrix with combined features
        R = librosa.segment.recurrence_matrix(
            features,
            width=3,     # Only consider nearby frames as potentially similar
            mode='affinity',
            sym=True    # Ensure the recurrence matrix is symmetric
        )
        
        # Compute novelty curve using the diagonal of the recurrence matrix
        N = R.shape[0]
        diagonal_sum = np.array([np.sum(np.diag(R, k)) for k in range(-N//4, N//4)])
        novelty = np.diff(diagonal_sum)
        novelty = np.pad(novelty, (1, 0))  # Pad to maintain length
        
        # Detect segment boundaries from novelty peaks
        kernel_size = 7
        novelty_peaks = librosa.util.peak_pick(novelty,
                                              pre_max=kernel_size,
                                              post_max=kernel_size,
                                              pre_avg=kernel_size,
                                              post_avg=kernel_size,
                                              delta=0.1,
                                              wait=kernel_size)
        
        # Convert frame indices to timestamps
        bound_times = librosa.frames_to_time(novelty_peaks, sr=self.sr)
        
        # Also detect repeated segments using the recurrence matrix
        segment_labels = librosa.segment.agglomerative(R, len(novelty_peaks))
        
        return {
            'boundary_times': bound_times,
            'segment_labels': segment_labels,
            'recurrence_matrix': R,
            'features': features
        }
    
    def analyze_timbre(self):
        """Analyze timbral features of the track."""
        spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0]
        return {
            'centroids': spectral_centroids,
            'rolloff': spectral_rolloff,
            'bandwidth': spectral_bandwidth
        }
    
    def visualize_waveform(self):
        """Visualize the waveform of the audio."""
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(self.y, sr=self.sr)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
    
    def visualize_spectrogram(self):
        """Visualize the spectrogram of the audio."""
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()
    
    def visualize_mel_spectrogram(self):
        """Visualize the mel-scaled spectrogram."""
        mel_spect = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        mel_db = librosa.power_to_db(mel_spect, ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mel_db, sr=self.sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.show()
    
    def visualize_chromagram(self):
        """Visualize the chromagram (pitch class profile over time)."""
        _, chroma = self.analyze_key()
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        plt.show()
    
    def visualize_onsets_and_beats(self):
        """Visualize note onsets and beats."""
        if self.onset_frames is None:
            self.analyze_onsets()
        if self.beats is None:
            self.analyze_tempo()
            
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.5)
        plt.vlines(librosa.frames_to_time(self.onset_frames), -1, 1, color='r', alpha=0.5, label='Onsets')
        plt.vlines(librosa.frames_to_time(self.beats), -1, 1, color='b', alpha=0.5, label='Beats')
        plt.legend()
        plt.title(f'Waveform with Onsets and Beats (Tempo: {self.tempo:.0f} BPM)')
        
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max),
                               y_axis='log', x_axis='time')
        plt.vlines(librosa.frames_to_time(self.onset_frames), 0, self.sr/2, color='r', alpha=0.5)
        plt.vlines(librosa.frames_to_time(self.beats), 0, self.sr/2, color='b', alpha=0.5)
        plt.title('Spectrogram with Onsets and Beats')
        plt.tight_layout()
        plt.show()
    
    def visualize_structure(self):
        """Visualize the structural analysis of the track."""
        # Get structural analysis
        structure = self.analyze_structure()
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(12, 8))
        
        # Plot 1: Self-similarity matrix
        ax1 = plt.subplot(2, 1, 1)
        img1 = librosa.display.specshow(structure['recurrence_matrix'], 
                                      x_axis='time', y_axis='time',
                                      ax=ax1)
        ax1.set_aspect('equal')
        fig.colorbar(img1, ax=ax1, label='Similarity')
        ax1.set_title('Self-similarity Matrix')
        
        # Plot 2: Features and segment boundaries
        ax2 = plt.subplot(2, 1, 2)
        # Create a custom feature display
        times = librosa.times_like(structure['features'][0])
        img2 = ax2.imshow(structure['features'], 
                         aspect='auto', 
                         origin='lower',
                         extent=[times[0], times[-1], 0, structure['features'].shape[0]])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Feature dimension')
        
        # Convert segment labels to time points and add boundary lines
        boundary_times = librosa.frames_to_time(structure['segment_labels'])
        ax2.vlines(boundary_times, 0, structure['features'].shape[0]-1, 
                  color='r', alpha=0.5, label='Segment boundaries')
        fig.colorbar(img2, ax=ax2, label='Magnitude')
        ax2.legend()
        ax2.set_title('Feature representation with segment boundaries')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_all(self):
        """Perform complete analysis and return a summary."""
        # Basic features
        tempo = self.analyze_tempo()
        key, _ = self.analyze_key()
        duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Structural analysis
        structure = self.analyze_structure()
        
        # Timbral features
        timbre = self.analyze_timbre()
        
        # Onset detection
        onsets = self.analyze_onsets()
        
        return {
            'tempo': tempo,
            'key': key,
            'duration': duration,
            'num_beats': len(self.beats),
            'num_segments': len(structure['boundary_times']),
            'num_onsets': len(onsets),
            'avg_spectral_centroid': np.mean(timbre['centroids']),
            'avg_spectral_rolloff': np.mean(timbre['rolloff']),
            'avg_spectral_bandwidth': np.mean(timbre['bandwidth']),
            'segment_times': structure['boundary_times'].tolist(),
            'segment_labels': structure['segment_labels'].tolist()
        }
