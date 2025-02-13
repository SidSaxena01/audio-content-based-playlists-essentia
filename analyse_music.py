"""
analyze_music.py

A standalone Python script to analyze an entire audio collection (MP3 files)
using Essentia. For each track the script extracts:

  - Tempo (BPM) via RhythmExtractor2013 and TempoCNN,
  - Key (with three profiles: temperley, krumhansl, edma),
  - Loudness (integrated loudness in LUFS via LoudnessEBUR128),
  - Embeddings from Discogs-Effnet and MSD-MusicCNN (averaged for similarity),
  - Danceability (both via a classifier using discogs embeddings and via the DSP algorithm),
  - Arousal and valence (emotion) using an ML model (with MSD-MusicCNN embeddings),
  - Voice/Instrumental classification (using discogs embeddings),
  - Genre / Music style (using discogs embeddings).

The script recursively finds all MP3 files under the given directory.
It handles errors gracefully and supports resumability by storing results in a JSON file.
"""

import os
import glob
import json
import argparse
import traceback
import numpy as np
import logging

from tqdm import tqdm

# Import Essentia algorithms and models
import essentia
import essentia.standard as es
from essentia.standard import (
    AudioLoader,
    MonoMixer,
    Resample,
    RhythmExtractor2013,
    BpmHistogramDescriptors,
    KeyExtractor,
    LoudnessEBUR128,
    TempoCNN,
    TensorflowPredict2D,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredictMusiCNN,
    Danceability,
    StereoTrimmer,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global constant: directory holding the models
MODELS_DIRECTORY = "models"


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.
    Converts numpy.float32, numpy.float64, numpy.int32, numpy.int64, and arrays to native Python types.
    """
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_models():
    """
    Instantiate and return all models and algorithms required for processing.
    They are created only once and then re-used for each track.
    """
    models = {}

    # TempoCNN model: search for available .pb files
    pb_models = glob.glob(os.path.join(MODELS_DIRECTORY, "tempo", "*.pb"))
    if pb_models:
        tempo_cnn_model_path = pb_models[0]
        logging.info("Using TempoCNN model: %s", tempo_cnn_model_path)
    else:
        tempo_cnn_model_path = os.path.join(MODELS_DIRECTORY, "tempo", "deeptemp-k16-3.pb")
        logging.info("Using default TempoCNN model: %s", tempo_cnn_model_path)
    models["tempo_cnn"] = es.TempoCNN(graphFilename=tempo_cnn_model_path)

    # Embedding models:
    discogs_embeddings_path = os.path.join(MODELS_DIRECTORY, "embeddings", "discogs-effnet-bs64-1.pb")
    models["discogs_model"] = TensorflowPredictEffnetDiscogs(
        graphFilename=discogs_embeddings_path, output="PartitionedCall:1"
    )

    musicnn_embeddings_path = os.path.join(MODELS_DIRECTORY, "embeddings", "msd-musicnn-1.pb")
    models["musicnn_model"] = TensorflowPredictMusiCNN(
        graphFilename=musicnn_embeddings_path, output="model/dense/BiasAdd"
    )

    # Danceability classifier (expects discogs embeddings)
    danceability_weights_path = os.path.join(MODELS_DIRECTORY, "danceability", "danceability-discogs-effnet-1.pb")
    models["danceability_classifier"] = TensorflowPredict2D(
        graphFilename=danceability_weights_path, output="model/Softmax"
    )

    # Danceability algorithm (signal processing approach)
    models["danceability_algo"] = Danceability()

    # Valence & Arousal model (using msd-musicnn activations)
    va_weights_path = os.path.join(MODELS_DIRECTORY, "valence-arousal", "emomusic-msd-musicnn-2.pb")
    models["va_model"] = TensorflowPredict2D(
        graphFilename=va_weights_path, output="model/Identity"
    )

    # Voice/Instrumental classifier (using discogs embeddings)
    voice_instrumental_weights_path = os.path.join(
        MODELS_DIRECTORY, "voice-instrumental", "voice_instrumental-discogs-effnet-1.pb"
    )
    models["voice_instrumental_model"] = TensorflowPredict2D(
        graphFilename=voice_instrumental_weights_path, output="model/Softmax"
    )

    # Genre / Style classifier (using discogs embeddings)
    genre_weights_path = os.path.join(MODELS_DIRECTORY, "genre", "genre_discogs400-discogs-effnet-1.pb")
    models["genre_model"] = TensorflowPredict2D(
        graphFilename=genre_weights_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    # Load genre metadata containing class names
    genre_metadata_path = os.path.join(MODELS_DIRECTORY, "genre", "genre_discogs400-discogs-effnet-1.json")
    try:
        with open(genre_metadata_path, "r") as f:
            models["genre_metadata"] = json.load(f)
    except Exception as e:
        logging.error("Error loading genre metadata: %s", e)
        models["genre_metadata"] = {"classes": []}

    # Traditional Essentia algorithms:
    models["rhythm_extractor"] = RhythmExtractor2013(method="multifeature")
    models["bpm_histogram"] = BpmHistogramDescriptors()

    # Instantiate three KeyExtractor objects (one per profile)
    models["key_extractors"] = {
        "temperley": KeyExtractor(profileType="temperley"),
        "krumhansl": KeyExtractor(profileType="krumhansl"),
        "edma": KeyExtractor(profileType="edma"),
    }

    # Loudness estimation algorithm (requires stereo input at 44100 Hz)
    models["loudness"] = LoudnessEBUR128(hopSize=1024 / 44100, startAtZero=True)
    models["stereo_trimmer"] = StereoTrimmer(startTime=0, endTime=10)  # Use first 10 sec

    return models


def process_track(file_path, models):
    """
    Process a single audio file and return a dictionary of descriptors.
    Any errors encountered will be caught and logged.
    """
    result = {}
    try:
        # Load the audio file once as stereo at 44100 Hz.
        audio_loader = es.AudioLoader(filename=file_path, sampleRate=44100)
        audio, sr, num_channels, md5, bit_rate, codec = audio_loader()
        # For loudness we need stereo audio.
        audio_stereo = audio

        # Create a mono version (for most other algorithms)
        mono_mixer = MonoMixer(type="mix")
        audio_mono = mono_mixer(audio, num_channels)
        # (audio_mono is now 44100 Hz mono)

        # ----- Tempo estimation with RhythmExtractor2013 -----
        rhythm_extractor = models["rhythm_extractor"]
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio_mono)
        result["rhythm_extractor"] = {
            "bpm": bpm,
            "beats_confidence": beats_confidence,
            "beats": beats.tolist() if isinstance(beats, np.ndarray) else beats,
        }

        # BPM histogram descriptors
        peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = models[
            "bpm_histogram"
        ](beats_intervals)
        result["bpm_histogram"] = {
            "peak1_bpm": peak1_bpm,
            "peak1_weight": peak1_weight,
            "peak1_spread": peak1_spread,
            "peak2_bpm": peak2_bpm,
            "peak2_weight": peak2_weight,
            "peak2_spread": peak2_spread,
            "histogram": histogram.tolist() if isinstance(histogram, np.ndarray) else histogram,
        }

        # ----- TempoCNN estimation -----
        # Resample mono audio to 11025 Hz for TempoCNN.
        resample_11k = Resample(inputSampleRate=44100, outputSampleRate=11025)
        audio_11khz = resample_11k(audio_mono)
        tempo_cnn = models["tempo_cnn"]
        global_bpm, local_bpm, local_probs = tempo_cnn(audio_11khz)
        result["tempo_cnn"] = {
            "global_bpm": global_bpm,
            "local_bpm": local_bpm.tolist() if isinstance(local_bpm, np.ndarray) else local_bpm,
            "local_probs": local_probs.tolist() if isinstance(local_probs, np.ndarray) else local_probs,
        }

        # ----- Key detection using three profiles -----
        key_results = {}
        for profile, key_extractor in models["key_extractors"].items():
            key, scale, strength = key_extractor(audio_mono)
            key_results[profile] = {"key": key, "scale": scale, "strength": strength}
        result["key"] = key_results

        # ----- Loudness estimation (requires stereo) -----
        stereo_trimmer = models["stereo_trimmer"]
        audio_st_trim = stereo_trimmer(audio_stereo)
        ebu_momentary, ebu_shortterm, ebu_integrated, dr = models["loudness"](audio_st_trim)
        result["loudness"] = {
            "integrated_loudness": ebu_integrated,
            "dynamic_range": dr,
            "ebu_momentary": ebu_momentary.tolist() if isinstance(ebu_momentary, np.ndarray) else ebu_momentary,
            "ebu_shortterm": ebu_shortterm.tolist() if isinstance(ebu_shortterm, np.ndarray) else ebu_shortterm,
        }

        # ----- Prepare audio for ML models (resample mono to 16kHz) -----
        resample_16k = Resample(inputSampleRate=44100, outputSampleRate=16000, quality=4)
        audio_16khz = resample_16k(audio_mono)

        # ----- Embeddings extraction -----
        # Discogs-Effnet embeddings
        discogs_model = models["discogs_model"]
        discogs_embeddings = discogs_model(audio_16khz)
        discogs_embedding_mean = np.mean(discogs_embeddings, axis=0)
        result["discogs_embedding_mean"] = discogs_embedding_mean.tolist()

        # MSD-MusicCNN activations
        musicnn_model = models["musicnn_model"]
        musicnn_activations = musicnn_model(audio_16khz)
        musicnn_embedding_mean = np.mean(musicnn_activations, axis=0)
        result["musicnn_embedding_mean"] = musicnn_embedding_mean.tolist()

        # ----- Danceability -----
        # Classifier (using discogs embeddings)
        danceability_classifier = models["danceability_classifier"]
        danceability_preds = danceability_classifier(discogs_embeddings)
        danceability_classifier_avg = np.mean(danceability_preds, axis=0)
        result["danceability_classifier"] = danceability_classifier_avg.tolist()

        # Signal processing Danceability algorithm (operates on mono audio)
        danceability_algo = models["danceability_algo"]
        danceability_value, dfa = danceability_algo(audio_mono)
        normalized_danceability = danceability_value / 3.0  # normalization (0-1)
        result["danceability_algorithm"] = {
            "danceability": danceability_value,
            "normalized": normalized_danceability,
            "dfa": dfa.tolist() if isinstance(dfa, np.ndarray) else dfa,
        }

        # ----- Valence & Arousal (emotion) -----
        va_model = models["va_model"]
        va_preds = va_model(musicnn_activations)
        va_avg = np.mean(va_preds, axis=0)
        # Scale predictions from [1,9] to [0,1]; assume first value is valence, second is arousal.
        valence = (va_avg[0] - 1) / 8.0
        arousal = (va_avg[1] - 1) / 8.0
        result["valence_arousal"] = {"valence": valence, "arousal": arousal}

        # ----- Voice/Instrumental classification -----
        voice_instrumental_model = models["voice_instrumental_model"]
        vi_preds = voice_instrumental_model(discogs_embeddings)
        vi_avg = np.mean(vi_preds, axis=0)
        # Assuming two classes: voice and instrumental.
        result["voice_instrumental"] = {"voice": vi_avg[0], "instrumental": vi_avg[1]}

        # ----- Genre / Style classification -----
        genre_model = models["genre_model"]
        genre_preds = genre_model(discogs_embeddings)
        genre_avg = np.mean(genre_preds, axis=0)
        # Get the genre class names from metadata
        genre_classes = models["genre_metadata"].get("classes", [])
        # Pair each genre name with its probability and sort by probability descending
        genre_prob_pairs = sorted(zip(genre_classes, genre_avg.tolist()), key=lambda x: x[1], reverse=True)
        # Get the top two genres
        top_two = genre_prob_pairs[:2]
        result["genre"] = [{"genre": name, "probability": prob} for name, prob in top_two]

    except Exception as e:
        logging.error("Error processing file %s: %s", file_path, str(e))
        traceback.print_exc()
        result["error"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser(description="Audio analysis with Essentia")
    parser.add_argument("audio_directory", help="Path to the root directory of your audio collection")
    parser.add_argument(
        "--output", default="analysis_results.json", help="Path to the output JSON file for analysis results"
    )
    args = parser.parse_args()

    # Find all MP3 files recursively in the provided directory.
    audio_files = glob.glob(os.path.join(args.audio_directory, "**", "*.mp3"), recursive=True)
    logging.info("Found %d MP3 files in %s", len(audio_files), args.audio_directory)

    # If an output file already exists, load previous results to support resumability.
    results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            logging.warning("Existing output file %s is not valid JSON. Starting with an empty result.", args.output)
            results = {}

    # Instantiate all required models/algorithms (only once)
    models = load_models()

    # Process each file with a progress bar. Skip files already processed.
    for file_path in tqdm(audio_files, desc="Processing audio files"):
        if file_path in results:
            continue  # skip files that have already been processed
        logging.info("Processing %s", file_path)
        res = process_track(file_path, models)
        results[file_path] = res
        # Write results to disk after each file to enable resuming if interrupted.
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

    logging.info("Processing complete. Results saved to %s", args.output)


if __name__ == "__main__":
    main()