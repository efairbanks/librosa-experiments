# Music Analyzer

A Python-based tool for analyzing musical audio files. This project provides functionality to extract and visualize various musical features including:

- Tempo detection
- Key detection
- Beat tracking
- Waveform visualization
- Spectrogram analysis
- Musical structure analysis

## Installation

1. Install UV if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Usage

1. Import the MusicAnalyzer class:
```python
from music_analyzer import MusicAnalyzer
```

2. Create an analyzer instance with your audio file:
```python
analyzer = MusicAnalyzer("path_to_your_audio.mp3")
```

3. Analyze and visualize:
```python
# Get complete analysis
results = analyzer.analyze_all()

# Generate visualizations
analyzer.visualize_waveform()
analyzer.visualize_spectrogram()
analyzer.visualize_beats()
```

See `example.py` for a complete usage example.

## Supported File Formats

The analyzer supports common audio formats including:
- WAV
- MP3
- OGG
- FLAC

## Requirements

See `requirements.txt` for a complete list of dependencies.
