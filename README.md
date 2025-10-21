# 🐟 Voice2FishLog

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

A professional desktop application for real-time speech-to-structured fish catch logging. Converts voice commands into structured entries (length in cm) using advanced speech recognition and natural language processing.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
- [Speech Recognition Engines](#-speech-recognition-engines)
- [Parser System](#-parser-system)
- [Noise Profiles](#-noise-profiles)
- [Evaluation Pipeline](#-evaluation-pipeline)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🎯 Overview

Voice2FishLog enables fishery workers to log catch data hands-free through voice commands. The application processes natural speech like *"salmon twenty three point five"* and converts it to structured data entries with species identification and measurements in centimeters.

**Use Cases:**
- Commercial fishing vessels logging catch data
- Research vessels documenting marine life
- Fish markets recording inventory
- Aquaculture facilities tracking stock

**Technical Highlights:**
- **8 Speech Recognition Engines** - Multiple ASR backends for flexibility
- **Offline Capable** - Works without internet once models are cached
- **Real-time Processing** - WebRTC VAD + adaptive noise suppression
- **Robust Parsing** - Handles spelling variants, units, and natural speech patterns
- **Production Ready** - Comprehensive testing, evaluation pipeline, and error handling

---

## ✨ Key Features

### Speech Recognition
- 🎤 **Multi-Engine Support**: faster-whisper (default), WhisperX, Vosk, Google Cloud Speech, Wav2Vec2, AssemblyAI, Gemini, Chirp
- 🔇 **Noise Suppression**: Adaptive noise reduction with configurable profiles
- 🎙️ **Voice Activity Detection**: WebRTC VAD for intelligent audio segmentation
- 🔊 **Audio Storage**: Optional segment saving for debugging and analysis

### Data Processing
- 🐠 **Species Recognition**: Fuzzy matching for 100+ fish species with variant handling
- 📏 **Measurement Parsing**: Robust number extraction supporting decimals and ranges
- 🔄 **Unit Conversion**: Automatic conversion from various spoken units to cm
- ✏️ **ASR Correction**: Post-recognition text correction for common errors
- 🌐 **Text Normalization**: Handles spelling variants, abbreviations, and colloquialisms

### User Interface
- 🖥️ **PyQt6 GUI**: Modern, responsive desktop interface
- 📊 **Real-time Table**: Live data display with edit capabilities
- 📈 **Reports Tab**: Length distribution histograms and statistics
- ⚙️ **Settings Panel**: Engine selection, noise profiles, and configuration
- 🔄 **Undo/Redo**: Full transaction history with rollback support
- 💾 **Excel Export**: Automatic logging to .xlsx files

### Data Management
- 📁 **Excel Logging**: Structured haul logs with automatic file management
- 📝 **Session Logging**: Detailed session records with timestamps
- ☁️ **Google Sheets Backup**: Optional cloud backup integration
- 🔒 **Data Validation**: Input validation and error recovery

### Development & Testing
- ✅ **Comprehensive Tests**: Unit, integration, and end-to-end test coverage
- 📊 **Evaluation Pipeline**: Automated ASR engine benchmarking
- 🎯 **Metrics Dashboard**: WER, CER, DER, exact match, and MAE metrics
- 🔬 **Production Replay**: Simulate real-world conditions in testing

---

## 🏗️ Architecture

The application follows **Clean Architecture** principles with SOLID design patterns:

```
┌─────────────────────────────────────────────────────┐
│                   Presentation Layer                 │
│  (PyQt6 GUI, MainWindow, Widgets, Event Handlers)   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│                  Application Layer                   │
│    (Use Cases, Services, Business Logic)            │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│                    Domain Layer                      │
│  (Parser, Species Matcher, Number Parser)           │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│                Infrastructure Layer                  │
│ (Speech Recognizers, Config, Logger, Database)      │
└─────────────────────────────────────────────────────┘
```

### Design Patterns Implemented

- **Dependency Injection**: Testable component composition
- **Factory Pattern**: Speech recognizer creation with registry
- **Strategy Pattern**: Pluggable parsing and matching algorithms
- **Facade Pattern**: Unified configuration and service interfaces
- **Observer Pattern**: Event-driven speech recognition callbacks
- **Singleton Pattern**: Session logger and application state management
- **Use Case Pattern**: Business logic isolation from UI

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- 4GB RAM minimum (8GB+ recommended for larger models)
- Microphone or audio input device

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd fish-logging

# Create virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Run

```bash
# Launch application
python main.py
```

On first launch:
1. The default ASR model (faster-whisper base.en) will download automatically (~150MB)
2. The GUI will open with default settings
3. Grant microphone permissions when prompted
4. Click "Start Recording" and speak: *"salmon 23.5"*
5. See the parsed entry appear in the table

---

## 📦 Installation

### Detailed Setup

#### 1. System Requirements

**Minimum:**
- Python 3.11+
- 4GB RAM
- 2GB disk space (for models)
- Microphone/audio input

**Recommended:**
- Python 3.11+
- 8GB+ RAM
- NVIDIA GPU with CUDA (for faster inference)
- Quality microphone with noise cancellation

#### 2. Virtual Environment Setup

```bash
# Create environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.11+
```

#### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: GPU acceleration (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Model Downloads

Models download automatically on first use. To pre-download:

```bash
# faster-whisper (default)
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu')"

# Vosk (optional)
# Download from https://alphacephei.com/vosk/models
# Extract to ~/.cache/vosk/
```

#### 5. Configuration

```bash
# Copy example configuration (if provided)
cp .env.example .env

# Edit configuration
nano .env
```

#### 6. Verify Installation

```bash
# Run tests
python run_tests.py

# Launch application
python main.py
```

### Docker Installation (Alternative)

```bash
# Build image
docker build -t fish-logging .

# Run with audio device
docker run --device /dev/snd -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix fish-logging
```

---

## ⚙️ Configuration

### Configuration Files

All configuration files are located in the `config/` directory:

| File | Purpose | Required |
|------|---------|----------|
| `species.json` | Species lexicon with fuzzy matching aliases | ✅ Yes |
| `numbers.json` | Spoken number variants and mappings | ✅ Yes |
| `units.json` | Unit conversion rules (cm, inch, etc.) | ✅ Yes |
| `asr_corrections.json` | Post-ASR text corrections | ✅ Yes |
| `google_sheets.json` | Google Sheets backup credentials | ⬜ Optional |
| `user_settings.json` | User preferences (auto-generated) | ⬜ Optional |

### Species Configuration (`species.json`)

```json
{
  "salmon": {
    "scientific_name": "Salmo salar",
    "aliases": ["salman", "samon", "sammon", "atlantic salmon"],
    "category": "finfish"
  },
  "cod": {
    "scientific_name": "Gadus morhua",
    "aliases": ["atlantic cod", "cod fish", "court"],
    "category": "finfish"
  }
}
```

**Adding New Species:**
1. Add entry to `species.json`
2. Include common misspellings in `aliases`
3. Restart application

### Numbers Configuration (`numbers.json`)

```json
{
  "20": ["twenty", "20"],
  "21": ["twenty one", "twenty-one", "21"],
  "0.5": ["half", "point five", ".5"]
}
```

### ASR Corrections (`asr_corrections.json`)

```json
{
  "court": "cod",
  "salman": "salmon",
  "tree": "three",
  "to": "two"
}
```

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
# Speech Recognition API Keys
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
ASSEMBLYAI_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here

# Application Settings
SPEECH_ENGINE=faster-whisper
NOISE_PROFILE=mixed
NUMBERS_ONLY=false

# Database
EXCEL_OUTPUT_PATH=logs/hauls/logs.xlsx
SESSION_LOG_DIR=logs/sessions

# Audio
SAVE_AUDIO_SEGMENTS=false
AUDIO_SEGMENTS_DIR=audio/segments
```

### Application Settings

Configure via GUI Settings tab or `config/user_settings.json`:

```json
{
  "speech": {
    "engine": "faster-whisper",
    "numbers_only": false,
    "language": "en",
    "noise_profile": "mixed"
  },
  "audio": {
    "save_segments": false,
    "segments_dir": "audio/segments"
  },
  "database": {
    "excel_output_path": "logs/hauls/logs.xlsx",
    "backup_enabled": true
  },
  "ui": {
    "theme": "default",
    "window_size": [1200, 800],
    "auto_save_interval": 30
  }
}
```

---

## 📖 Usage Guide

### Basic Workflow

1. **Launch Application**
   ```bash
   python main.py
   ```

2. **Select ASR Engine** (Settings Tab)
   - Choose from dropdown: faster-whisper, whisperx, vosk, etc.
   - Configure noise profile based on environment

3. **Start Recording** (Main Tab)
   - Click "Start Recording" button
   - Status indicator shows "Listening..."

4. **Speak Entry**
   - Say: *"salmon twenty three point five"*
   - Or: *"cod 45 centimeters"*
   - Or: *"mackerel thirty to thirty five"*

5. **Review Entry**
   - Parsed entry appears in table
   - Edit if needed by double-clicking cells

6. **Save/Undo**
   - Entries auto-save to Excel
   - Use "Undo Last Entry" if needed

7. **View Reports** (Reports Tab)
   - Length distribution histograms
   - Species frequency charts
   - Session statistics

### Voice Commands

#### Fish Entry Patterns

```
Pattern: [species] [number] [optional_unit]

Examples:
✅ "salmon 23.5"
✅ "cod forty five centimeters"
✅ "mackerel 30 to 35"
✅ "haddock twenty three point five cm"
✅ "tuna 120 santim"
```

#### Special Commands

```
❌ "cancel" - Discards current entry
❌ "undo" - Removes last saved entry
❌ "clear" - Clears all entries (confirmation required)
```

### Number Recognition

The parser supports multiple spoken number formats:

```
Integers: "twenty three" → 23
Decimals: "twenty three point five" → 23.5
Ranges: "twenty to twenty five" → "20-25"
Fractions: "twenty three and a half" → 23.5
```

### Units Supported

```
Metric:
- cm, centimeter, centimetre, santim, santimetre
- m, meter, metre

Imperial:
- inch, inches, in
- foot, feet, ft
```

All measurements are converted to centimeters automatically.

### GUI Components

#### Main Tab
- **Transcription Panel**: Shows raw ASR output
- **Parsed Text Panel**: Displays normalized interpretation
- **Entry Table**: Live data with edit capability
- **Control Buttons**: Start/Stop, Undo, Clear
- **Status Bar**: Connection status and activity indicator

#### Settings Tab
- **ASR Engine**: Dropdown selection
- **Noise Profile**: clean/human/engine/mixed
- **Numbers Only Mode**: Toggle for numeric-only parsing
- **Audio Saving**: Enable segment capture for debugging

#### Reports Tab
- **Length Distribution**: Histogram by species
- **Species Frequency**: Pie chart of catches
- **Session Summary**: Count, avg, min, max statistics
- **Export**: Save reports as PNG/PDF

---

## 🎙️ Speech Recognition Engines

### Engine Comparison

| Engine | Speed | Accuracy | Offline | GPU Support | Best For |
|--------|-------|----------|---------|-------------|----------|
| **faster-whisper** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Yes | Production (default) |
| **whisperx** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Yes | High accuracy needs |
| **vosk** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ Yes | ⬜ No | Low resource devices |
| **wav2vec2** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Yes | ✅ Yes | Research/customization |
| **google** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⬜ No | N/A | Cloud-based accuracy |
| **assemblyai** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⬜ No | N/A | Streaming + features |
| **gemini** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⬜ No | N/A | Multimodal context |
| **chirp** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⬜ No | N/A | Low-resource languages |

### faster-whisper (Default)

**Setup:**
```python
# No setup needed - downloads automatically
# Model: base.en (~150MB)
```

**Advantages:**
- Excellent accuracy/speed balance
- Offline operation
- GPU acceleration
- Low memory footprint
- Production-ready

**Configuration:**
```python
# In settings or .env
SPEECH_ENGINE=faster-whisper
```

### WhisperX

**Setup:**
```bash
pip install whisperx
```

**Advantages:**
- Word-level timestamps
- Speaker diarization
- Highest accuracy
- Better alignment

**Use When:**
- Maximum accuracy needed
- Analyzing recorded audio
- Timestamp precision important

### Vosk

**Setup:**
```bash
pip install vosk
# Download model from https://alphacephei.com/vosk/models
# Extract to ~/.cache/vosk/
```

**Advantages:**
- Fastest inference
- Lowest resource usage
- Small model sizes (50MB)
- Works on Raspberry Pi

**Use When:**
- Limited hardware
- Embedded systems
- Speed over accuracy

### Google Cloud Speech

**Setup:**
```bash
# 1. Enable Cloud Speech-to-Text API in Google Cloud Console
# 2. Create service account and download JSON key
# 3. Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

**Advantages:**
- Industry-leading accuracy
- Automatic punctuation
- Profanity filtering
- 125+ languages

**Use When:**
- Internet available
- Accuracy is critical
- Budget for API costs (~$0.024/min)

### AssemblyAI

**Setup:**
```bash
# Get API key from https://www.assemblyai.com/
export ASSEMBLYAI_API_KEY=your_api_key
```

**Advantages:**
- Real-time streaming
- Speaker labels
- Content moderation
- Easy integration

### Gemini 2.5 Pro

**Setup:**
```bash
# Get API key from https://makersuite.google.com/
export GEMINI_API_KEY=your_api_key
```

**Advantages:**
- Multimodal understanding
- Context awareness
- Natural conversation

**Use When:**
- Complex queries
- Context important
- Multimodal input

### Wav2Vec2

**Setup:**
```bash
pip install transformers torch
```

**Advantages:**
- Self-supervised learning
- Fine-tunable
- Research-friendly
- Domain adaptation

**Use When:**
- Custom vocabulary
- Domain-specific terms
- Research projects

---

## 🔍 Parser System

### Architecture

```
Input: "salmon twenty three point five cm"
         ↓
┌─────────────────────────┐
│   Text Normalizer       │ ← ASR corrections
│   (asr_corrections.json)│
└───────────┬─────────────┘
            ↓
"salmon 23.5 cm"
            ↓
┌─────────────────────────┐
│   Species Matcher       │ ← Species database
│   (species.json)        │
└───────────┬─────────────┘
            ↓
species: "salmon"
            ↓
┌─────────────────────────┐
│   Number Parser         │ ← Number mappings
│   (numbers.json)        │
└───────────┬─────────────┘
            ↓
length: 23.5
            ↓
┌─────────────────────────┐
│   Unit Converter        │ ← Unit definitions
│   (units.json)          │
└───────────┬─────────────┘
            ↓
Output: {species: "salmon", length_cm: 23.5}
```

### Species Matcher

**Strategies:**
1. **Exact Match**: Direct dictionary lookup
2. **Fuzzy Match**: Levenshtein distance (threshold: 85%)
3. **Alias Match**: Check predefined aliases

**Example:**
```python
Input: "salman"
Step 1: Exact match → ❌ Not found
Step 2: Fuzzy match → ✅ "salmon" (similarity: 87%)
Output: "salmon"
```

### Number Parser

**Strategies:**
1. **Word to Number**: "twenty three" → 23
2. **Decimal Parsing**: "point five" → 0.5
3. **Range Detection**: "20 to 25" → "20-25"
4. **Fraction Handling**: "and a half" → 0.5

**Examples:**
```python
"twenty three" → 23
"twenty three point five" → 23.5
"twenty to twenty five" → "20-25"
"twenty three and a half" → 23.5
```

### Text Normalizer

**Operations:**
1. Lowercase conversion
2. Punctuation removal
3. Extra whitespace cleanup
4. ASR error correction
5. Unit standardization

**Example:**
```python
Input: "Court, Twenty-Three  Santim."
Step 1: Lowercase → "court, twenty-three  santim."
Step 2: Punctuation → "court twenty-three santim"
Step 3: Whitespace → "court twenty-three santim"
Step 4: ASR correction → "cod twenty-three santim"
Step 5: Unit standard → "cod twenty-three cm"
Output: "cod 23 cm"
```

---

## 🔊 Noise Profiles

Optimize performance for your acoustic environment:

### Profile Comparison

| Profile | Environment | VAD Mode | Suppression | Best For |
|---------|-------------|----------|-------------|----------|
| **clean** | Studio/Office | 1 (Lenient) | Gentle | Quiet environments |
| **human** | Crowded spaces | 2 (Moderate) | Moderate | Background voices |
| **engine** | Vessels/Machinery | 3 (Aggressive) | Strong | Constant noise |
| **mixed** | Variable conditions | 2 (Moderate) | Adaptive | General purpose |

### Detailed Profiles

#### Clean Profile
**Use When:**
- Quiet office environment
- Studio recording
- Minimal background noise

**Settings:**
```python
vad_mode: 1          # Lenient voice detection
suppression: 0.3     # Gentle noise reduction
frame_duration: 30ms # Standard frame size
```

#### Human Profile
**Use When:**
- Crowded fish markets
- Multiple people talking
- Background conversations

**Settings:**
```python
vad_mode: 2          # Moderate voice detection
suppression: 0.5     # Moderate noise reduction
highpass_cutoff: 200Hz # Remove low rumble
```

#### Engine Profile
**Use When:**
- Fishing vessels
- Engine rooms
- Constant machinery noise

**Settings:**
```python
vad_mode: 3          # Aggressive voice detection
suppression: 0.7     # Strong noise reduction
highpass_cutoff: 300Hz # Aggressive low-cut
```

#### Mixed Profile (Default)
**Use When:**
- Variable conditions
- Outdoor environments
- Unknown noise levels

**Settings:**
```python
vad_mode: 2          # Balanced detection
suppression: adaptive # Adapts to noise floor
min_speech: 0.5s     # Minimum segment length
max_segment: 4.0s    # Maximum segment length
```

### Configuring Noise Profiles

**Via GUI:**
1. Open Settings tab
2. Select "Noise Profile" dropdown
3. Choose appropriate profile
4. Click "Apply"

**Via Configuration File:**
```json
{
  "speech": {
    "noise_profile": "engine"
  }
}
```

**Via Environment Variable:**
```bash
export NOISE_PROFILE=engine
```

---


---

## 📊 Evaluation Pipeline

Comprehensive ASR engine benchmarking system for data-driven model selection.

### Overview

The evaluation pipeline enables:
- **Multi-model comparison**: Test multiple ASR engines simultaneously
- **Reproducible results**: Deterministic evaluation with full configuration tracking
- **Production simulation**: Replay mode mimics real-world segmentation
- **Rich metrics**: WER, CER, DER, exact match, MAE, and latency
- **Visualization**: Automated charts and reports

### Quick Start

```bash
# Run evaluation with default configuration
python -m evaluation.run_evaluation

# Run with limited samples (smoke test)
python -m evaluation.run_evaluation --max-samples 10

# Enable production replay mode
python -m evaluation.run_evaluation --production-replay

# Use custom configuration
python -m evaluation.run_evaluation --model-specs custom_specs.json
```

### Configuration

#### Model Specifications (`evaluation/presets/model_specs.json`)

```json
{
  "dataset_json": "tests/data/numbers.json",
  "audio_root": "tests/audio",
  "concat_number": true,
  "production_replay": false,
  "model_specs": [
    {
      "name": "faster-whisper",
      "sizes": ["tiny.en", "base.en", "small.en"],
      "compute_types": ["int8", "float16"],
      "beam_sizes": [1, 5]
    },
    {
      "name": "whisperx",
      "sizes": ["base.en"],
      "compute_types": ["float16"]
    },
    {
      "name": "vosk",
      "models": ["vosk-model-small-en-us-0.15"]
    }
  ]
}
```

#### Dataset Format (`tests/data/numbers.json`)

```json
{
  "samples": [
    {
      "audio_file": "audio/sample1.wav",
      "expected_text": "twenty three",
      "expected_number": 23,
      "metadata": {
        "speaker": "male",
        "noise_level": "clean"
      }
    }
  ]
}
```

### Evaluation Modes

#### Standard Mode (Default)
- Full audio files processed at once
- Fastest evaluation
- Good for model comparison

#### Production Replay Mode
- Simulates real-time segmentation
- Uses WebRTC VAD + noise controller
- Slower but realistic performance measurement

```bash
python -m evaluation.run_evaluation --production-replay
```

### Metrics Explained

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **WER** | Word Error Rate | 0-100% | Lower |
| **CER** | Character Error Rate | 0-100% | Lower |
| **DER** | Digit Error Rate | 0-100% | Lower |
| **Exact Match** | Perfect predictions | 0-100% | Higher |
| **MAE** | Mean Absolute Error (numbers) | 0+ | Lower |
| **Latency** | Processing time (ms) | 0+ | Lower |
| **Memory** | Peak RAM usage (MB) | 0+ | Lower |

### Output Artifacts

```
evaluation_outputs/
└── eval_2025-10-21_14-30-15/
    ├── results.parquet           # Raw results (columnar)
    ├── results.xlsx               # Excel spreadsheet
    ├── summary_statistics.json    # Aggregated metrics
    ├── config.json                # Full run configuration
    └── plots/
        ├── wer_comparison.png     # WER by model
        ├── latency_distribution.png
        ├── accuracy_vs_speed.png
        └── confusion_matrix.png
```

### Results Analysis

#### View Results

```python
import pandas as pd

# Load results
df = pd.read_parquet("evaluation_outputs/latest/results.parquet")

# Filter by model
whisper_results = df[df['model_name'] == 'faster-whisper']

# Calculate statistics
print(df.groupby('model_name')['wer'].describe())

# Plot comparisons
import seaborn as sns
sns.boxplot(data=df, x='model_name', y='wer')
```

#### Command Line Analysis

```bash
# Generate summary report
python -m evaluation.visualization \
    --results evaluation_outputs/latest/results.parquet \
    --output report.html

# Compare two runs
python -m evaluation.visualization \
    --compare run1/results.parquet run2/results.parquet
```

### Extending the Pipeline

#### Add New Model

1. Create recognizer class in `speech/`
2. Register in factory:
```python
# speech/factory.py
from speech.my_recognizer import MyRecognizer

RecognizerRegistry.register(
    name="my-model",
    factory=MyRecognizer,
    description="My custom ASR model"
)
```

3. Add to evaluation config:
```json
{
  "model_specs": [
    {
      "name": "my-model",
      "param1": ["value1", "value2"]
    }
  ]
}
```

#### Add New Metric

```python
# evaluation/metrics.py
def my_custom_metric(hypothesis: str, reference: str) -> float:
    """Calculate custom metric."""
    # Implementation
    return score

# evaluation/pipeline.py
from evaluation.metrics import my_custom_metric

# Add to metric computation
metrics["my_metric"] = my_custom_metric(hyp, ref)
```

---

## 🛠️ Development

### Development Setup

```bash
# Install development dependencies
# Optional and recommended, do these in virtual environment.
pip install -r requirements.txt 
pip install pytest pytest-cov black isort mypy pylint

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Style

**Formatting:**
```bash
# Format code
black .

# Sort imports
isort .

# Combined
black . && isort .
```

**Linting:**
```bash
# Type checking
mypy .

# Linting
pylint app/ parser/ speech/

# Combined
mypy . && pylint app/ parser/ speech/
```

### Project Structure

```
fish-logging/
├── app/                    # Application layer
│   ├── application.py      # Main app class
│   ├── startup.py          # App initialization
│   ├── services.py         # Service layer
│   └── use_cases.py        # Business logic
├── gui/                    # Presentation layer
│   ├── MainWindow.py       # Main window
│   ├── table_manager.py    # Table operations
│   ├── speech_event_handler.py
│   ├── status_presenter.py
│   └── widgets/            # UI components
├── parser/                 # Domain layer (core logic)
│   ├── parser.py               # Main FishParser
│   ├── species_matcher.py      # Species recognition
│   ├── number_parser.py        # Number extraction
│   ├── text_normalizer.py      # Text cleaning
│   ├── text_utils.py           # Text utilities
│   └── config.py               # Parser configuration
├── speech/                  # Infrastructure (ASR)
│   ├── base_recognizer.py      # Abstract base class
│   ├── factory.py              # Recognizer factory
│   ├── faster_whisper_recognizer.py
│   ├── whisperx_recognizer.py
│   ├── vosk_recognizer.py
│   ├── google_speech_recognizer.py
│   ├── assemblyai_recognizer.py
│   ├── gemini_recognizer.py
│   ├── wav2vec2_recognizer.py
│   ├── insanely_faster_whisper.py
│   └── noise_profiles.py       # Noise configuration
├── config/                  # Configuration
│   ├── config.py               # Config data classes
│   ├── service.py              # Config facade
│   ├── species.json            # Species database
│   ├── numbers.json            # Number mappings
│   ├── units.json              # Unit definitions
│   ├── asr_corrections.json    # ASR corrections
│   ├── google_sheets.json      # Sheets credentials
│   └── user_settings.json      # User preferences
├── logger/                  # Logging infrastructure
│   ├── excel_logger.py         # Excel file output
│   └── session_logger.py       # Session tracking
├── noise/                   # Audio processing
│   ├── controller.py           # Noise pipeline
│   ├── simple_controller.py    # Simplified version
│   └── suppressor.py           # Noise suppression
├── services/                # External services
│   ├── audio_saver.py          # Audio segment storage
│   └── google_sheets_backup.py # Cloud backup
├── core/                    # Core utilities
│   ├── container.py            # DI container
│   ├── exceptions.py           # Custom exceptions
│   ├── error_handler.py        # Error handling
│   └── result.py               # Result types
├── evaluation/              # Testing & benchmarking
│   ├── pipeline.py             # Evaluation orchestrator
│   ├── config.py               # Evaluation config
│   ├── metrics.py              # Metric calculations
│   ├── normalization.py        # Text normalization
│   ├── visualization.py        # Result plotting
│   ├── run_evaluation.py       # CLI runner
│   ├── README_TEST_EVAL.md     # Evaluation docs
│   ├── datasets/               # Test datasets
│   └── presets/
│       └── model_specs.json    # Model configurations
│
├── reports/                 # Report generation
│   ├── length_distribution_report.py
│   └── output/                 # Generated reports
│
├── tests/                   # Test suite
│   ├── test_core.py
│   ├── test_use_cases.py
│   ├── test_speech_factory.py
│   ├── test_config_service.py
│   ├── number_parser_test.py
│   ├── test_numbers_integration.py
│   ├── test_noise_profiles.py
│   ├── test_evaluation_pipeline.py
│
├── logs/                    # Output logs
│   ├── hauls/                  # Excel haul logs
│   └── sessions/               # Session logs
│
├── audio/                   # Audio data
│   ├── evaluation/             # Evaluation audio
│   └── segments/               # Saved segments (optional)
│
├── assets/                  # Static assets
│   ├── bg.jpg                  # Background image
│   ├── audio/                  # Audio assets
│   └── icons/                  # Application icons
│
├── scripts/                 # Utility scripts
│   ├── add_noise.py            # Audio augmentation
│   ├── generate_dataset_json.py
│   └── realtime_clean_wav.py
│
└── evaluation_outputs/      # Evaluation results
    └── eval_YYYYMMDD_HHMMSS/   # Timestamped runs
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Application entry point, initializes and starts app |
| `app/application.py` | Main application class managing lifecycle |
| `app/use_cases.py` | Business logic separated from UI |
| `gui/MainWindow.py` | Main GUI window coordinating UI components |
| `parser/parser.py` | Core parsing logic for fish entries |
| `speech/factory.py` | Factory for creating ASR recognizers |
| `config/service.py` | Facade for simplified configuration access |
| `evaluation/pipeline.py` | Automated ASR benchmarking system |
| `logger/excel_logger.py` | Excel file management and writing |
| `noise/controller.py` | Real-time audio processing pipeline |

---

## 🤝 Adding New Features


### Getting Started

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Write/update tests**
5. **Run test suite**:
   ```bash
   python run_tests.py
   ```
6. **Format code**:
   ```bash
   black . && isort .
   ```
7. **Commit changes**:
   ```bash
   git commit -m "Add: description of your changes"
   ```
8. **Push to fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Submit pull request**

### Contribution Types

- 🐛 **Bug fixes**: Fix issues with existing functionality
- ✨ **New features**: Add new capabilities
- 📚 **Documentation**: Improve docs, examples, or guides
- 🧪 **Tests**: Add or improve test coverage
- 🎨 **UI improvements**: Enhance user interface
- ⚡ **Performance**: Optimize speed or resource usage
- ♻️ **Refactoring**: Improve code quality

### Code Standards

- Follow PEP 8 style guide
- Use type hints for function signatures
- Write docstrings for public APIs
- Maintain test coverage >80%
- Keep functions focused and small (<50 lines)
- Use meaningful variable names


---
### Technologies
- **PyQt6**: GUI framework
- **OpenAI Whisper**: Speech recognition models
- **faster-whisper**: Optimized Whisper inference
- **WebRTC VAD**: Voice activity detection
- **RapidFuzz**: Fuzzy string matching
- **Pandas**: Data manipulation
- **Loguru**: Logging framework
---



Made by Ali Altun in his internship, 08/2025 - 10/2025. 