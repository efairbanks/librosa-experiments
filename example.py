from music_analyzer import MusicAnalyzer
import librosa
import matplotlib.pyplot as plt

def main():
    # Replace with your audio file path
    audio_file = "WORRIEDBYTE.wav"  # or .wav
    
    try:
        print("Loading and analyzing audio file...")
        analyzer = MusicAnalyzer(audio_file)
        
        # Perform complete analysis
        results = analyzer.analyze_all()
        
        # Print detailed analysis results
        print("\n=== Analysis Results ===")
        print(f"Basic Information:")
        print(f"- Duration: {results['duration']:.2f} seconds")
        print(f"- Musical Key: {results['key']}")
        
        print("\nRhythmic Analysis:")
        print(f"- Tempo: {results['tempo']:.1f} BPM")
        print(f"- Number of Beats: {results['num_beats']}")
        print(f"- Number of Note Onsets: {results['num_onsets']}")
        
        print("\nStructural Analysis:")
        print(f"- Number of Segments: {results['num_segments']}")
        print("- Segment Boundaries (seconds):")
        for i, time in enumerate(results['segment_times']):
            label = chr(65 + results['segment_labels'][i] % 26)  # Convert to letters A, B, C, etc.
            print(f"  {time:.2f}s - Section {label}")
        
        print("\nTimbral Features:")
        print(f"- Average Spectral Centroid: {results['avg_spectral_centroid']:.2f} Hz")
        print(f"- Average Spectral Rolloff: {results['avg_spectral_rolloff']:.2f} Hz")
        print(f"- Average Spectral Bandwidth: {results['avg_spectral_bandwidth']:.2f} Hz")
        
        # Generate all available visualizations
        print("\n=== Generating Visualizations ===")
        
        print("1. Basic Waveform...")
        analyzer.visualize_waveform()
        
        print("2. Spectrogram...")
        analyzer.visualize_spectrogram()
        
        print("3. Mel Spectrogram...")
        analyzer.visualize_mel_spectrogram()
        
        print("4. Chromagram (Pitch Content)...")
        analyzer.visualize_chromagram()
        
        print("5. Onsets and Beats...")
        analyzer.visualize_onsets_and_beats()
        
        print("6. Structural Analysis...")
        analyzer.visualize_structure()
        
        # Additional pitch analysis visualization
        print("7. Pitch Analysis...")
        pitches, magnitudes = analyzer.analyze_pitch()
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(pitches, y_axis='linear', x_axis='time')
        plt.colorbar(label='Frequency (Hz)')
        plt.title('Pitch Contours')
        plt.tight_layout()
        plt.show()
        
        print("\nAnalysis complete! The visualizations show:")
        print("1. Waveform: The raw audio signal")
        print("2. Spectrogram: Frequency content over time")
        print("3. Mel Spectrogram: Frequency content adapted to human hearing")
        print("4. Chromagram: Musical pitch class distribution")
        print("5. Onsets and Beats: Rhythmic structure")
        print("6. Structure: Self-similarity and segment boundaries")
        print("7. Pitch: Detailed frequency tracking")
        print("\nClose the visualization windows to exit.")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main()
