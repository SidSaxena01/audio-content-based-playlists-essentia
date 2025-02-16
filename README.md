# Essentia Playlists Analysis

## Overview
- Analyze a music collection (using the MusAV dataset) to extract audio descriptors such as tempo, key, loudness, embeddings, music styles, voice/instrumental classification, danceability, and emotion (arousal/valence) using Essentia.
- Generate statistical reports and visualizations on these features (e.g. distributions of music styles, tempo, loudness, keys, and emotion space).
- Develop a Streamlit application:
  • for creating playlists based on descriptor queries including tempo ranges, binary selectors for vocal/instrumental, and range selectors for danceability and emotion.
  • for generating playlists based on track similarity using averaged embeddings from Discogs-Effnet and MSD-MusicCNN.
- The project also includes creating a written report discussing design decisions, evaluation of extracted features (highlighting successes and limitations).

## Deliverables

- The code used to generate the features from the audio  
      [audio_analysis.py](audio_analysis.py) — This file contains the Python code that processes your audio files to extract features.

- The features that you extracted from the audio  
   [results/](results/). — The extracted features are stored in this directory. 

- The code for generating the overview report from the features.  
   a[analysis_report.ipynb](analysis_report.ipynb) — This notebook generates the detailed report with visualizations.

- The code for the user interface (two separate apps, one for queries by descriptors, another for similarity). We should be able to run this on our computer to generate our own playlists.
   - The user interface is a singular Streamlit app located in the playlist_generation/ [directory](playlist_generation/):
   - [playlist_generation/main.py](playlist_generation/main.py): Entry point for the app. Sets up page configuration, sidebar options, and routes to different pages.
   - [playlist_generation/ui_components.py](playlist_generation/ui_components.py): Contains functions to render various UI components such as descriptor queries, similarity searches, and genre/style filtering.
   - [playlist_generation/utils.py](playlist_generation/utils.py): Provides utility functions for tasks like validating directories, processing JSON analyses, and creating playlists.
   - [playlist_generation/music_library.py](playlist_generation/music_library.py): Manages the loading and processing of audio analysis data, including computing statistics and filtering tracks.

- A report (~2 pages) describing the decisions you took in all steps, when generating the features, computing statistics overview, building the interface, along with your personal opinion of the quality of the system in terms of its capability to generate playlists. Include your observations on the quality of the extracted features, including examples of good and bad extracted features that you encountered. Report can be larger than 2 pages if you need space to show Figures.
- The comprehensive written report, detailing decisions, motivations, observations, analysis, visualizations, opinions, and recommendations, is available in a markdown file, [Report.md](Report.md).


## How to Run

### Prerequisites

- Python 3.10
- Required
- Required Python packages (install via `pip install -r requirements.txt`)
- Essentia and its Python bindings installed
- Pre-trained TensorFlow models placed in the correct directories (see below)

### Directory Structure
Place your data as follows:
- Audio files: Place your `.mp3` files in a directory (e.g., `data/audio`).
- Models: Place all model files inside a directory (e.g., `models`), matching expected subpaths:
  - `models/embeddings/`
  - `models/genre/`
  - `models/voice-instrumental/`
  - `models/danceability/`
  - `models/valence-arousal/`

### Running the Audio Analysis

#### 1. Run audio_analysis.py
This script processes your audio files and outputs analysis in JSON format.

Example command:
```
python audio_analysis.py /path/to/data/audio /path/to/output/json --models_dir /path/to/models --workers 4
```
- `/path/to/data/audio`: Directory containing your `.mp3` audio files.
- `/path/to/output/json`: Directory where the JSON analysis results will be stored.
- `--models_dir`: Directory where model files are located.
- `--workers`: (Optional) Number of parallel processes.

Command used for this project:
```
uv run audio_analysis.py data/MusAV results/ --workers 11
```
- `data/MusAV`: Directory containing the `.mp3` audio files.
- `results/`: Directory where the JSON analysis results are stored.
The whole analysis took somewhere between 10-15 minutes due to multiprocessing, reducing the time from approximately 50 minutes. The number of workers was determined by getting the CPU count minus one.
- `--workers`: CPU Count - 1 


#### 2. Analysis & Reports

- An interactive analysis is available via the Jupyter Notebook: [analysis_report.ipynb](analysis_report.ipynb).
- The written report is documented in this [file](Report.md)
- The `report.py` script generates static reports (plots and TSV files) when executed.

#### 3. Run report.py
This script reads the JSON files produced by `audio_analysis.py` and generates visualizations and TSV reports.

Example command:
```
python report.py /path/to/output/json /path/to/report/output
```
- `/path/to/output/json`: Directory with JSON analysis files.
- `/path/to/report/output`: Directory where the generated report files (plots, TSV) will be saved.

Command used for this project:
```
uv run report.py results/ reports/
```
- `results/`: Directory with JSON analysis files.
- `reports/`: Directory where the generated report files (plots, TSV) are saved.
### Running the Streamlit App

To launch the Streamlit app for playlist generation, follow these steps:

1. Open a terminal and change directory to the playlist_generation folder:
   ```
   cd playlist_generation
   ```
2. Run the app using Streamlit:
   ```
   streamlit run main.py
   ```

Command used for this project:
```
uv run streamlit run main.py
```

3. The app will open in your default web browser. Use the sidebar to configure your library directories and load your music library.

## Additional Notes
- Ensure all dependencies in `requirements.txt` are installed.
- Verify the file structure in the models directory matches the expectations in the scripts.
- For the MusAV dataset, the analysis files are in the [results/](results/) directory and the generated plots and TSV files are located in the [reports/](reports/) directory.
- For any issues, check the console output for error messages.
