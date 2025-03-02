# Experiments /w librosa

Right now this does some beat-by-beat harmonic analysis of musical audio signals and also generates a recurrence/correllation matrix that shows the harmonic similarities between beats.

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

`uv run simpler_example.py`

## Requirements

See `requirements.txt` for a complete list of dependencies.
