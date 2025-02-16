import argparse
import json
import logging
import os
from multiprocessing import Pool, current_process
from pathlib import Path

import essentia.standard as es
import numpy as np
from tqdm import tqdm

# Global variable to hold models in each worker
models_global = None


def load_models(models_dir):
    """Load all required Essentia models and metadata."""
    models = {}

    # Load Discogs-Effnet embeddings model
    discogs_emb_path = os.path.join(models_dir, "embeddings/discogs-effnet-bs64-1.pb")
    models["discogs_model"] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=discogs_emb_path, output="PartitionedCall:1"
    )

    # Load MusicCNN embeddings model
    musicnn_emb_path = os.path.join(models_dir, "embeddings/msd-musicnn-1.pb")
    models["musicnn_model"] = es.TensorflowPredictMusiCNN(
        graphFilename=musicnn_emb_path, output="model/dense/BiasAdd"
    )

    # Load genre classification model
    genre_model_path = os.path.join(
        models_dir, "genre/genre_discogs400-discogs-effnet-1.pb"
    )
    models["genre_model"] = es.TensorflowPredict2D(
        graphFilename=genre_model_path,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    with open(
        os.path.join(models_dir, "genre/genre_discogs400-discogs-effnet-1.json"), "r"
    ) as f:
        models["genre_metadata"] = json.load(f)

    # Load voice/instrumental model
    voice_model_path = os.path.join(
        models_dir, "voice-instrumental/voice_instrumental-discogs-effnet-1.pb"
    )
    models["voice_model"] = es.TensorflowPredict2D(
        graphFilename=voice_model_path, output="model/Softmax"
    )
    with open(
        os.path.join(
            models_dir, "voice-instrumental/voice_instrumental-discogs-effnet-1.json"
        ),
        "r",
    ) as f:
        models["voice_metadata"] = json.load(f)

    # Load danceability model
    dance_model_path = os.path.join(
        models_dir, "danceability/danceability-discogs-effnet-1.pb"
    )
    models["dance_model"] = es.TensorflowPredict2D(
        graphFilename=dance_model_path, output="model/Softmax"
    )
    with open(
        os.path.join(models_dir, "danceability/danceability-discogs-effnet-1.json"), "r"
    ) as f:
        models["dance_metadata"] = json.load(f)

    # Load valence/arousal model
    va_model_path = os.path.join(
        models_dir, "valence-arousal/emomusic-msd-musicnn-2.pb"
    )
    models["va_model"] = es.TensorflowPredict2D(
        graphFilename=va_model_path, output="model/Identity"
    )
    with open(
        os.path.join(models_dir, "valence-arousal/emomusic-msd-musicnn-2.json"), "r"
    ) as f:
        models["va_metadata"] = json.load(f)

    return models


def process_file(file_path, output_path, models):
    """Process a single audio file and save results to output_path."""
    try:
        # Load audio
        audio_loader = es.AudioLoader(filename=file_path)
        audio_stereo, sr, num_channels, _, _, _ = audio_loader()

        # Resample stereo to 44.1kHz for loudness calculation
        if sr != 44100:
            resampler = es.Resample(inputSampleRate=sr, outputSampleRate=44100)
            audio_stereo = resampler(audio_stereo)

        # Compute loudness
        loudness_alg = es.LoudnessEBUR128(sampleRate=44100)
        _, _, integrated_loudness, _ = loudness_alg(audio_stereo)

        # Downmix to mono and resample to 44.1kHz for other features
        mono_mixer = es.MonoMixer()
        audio_mono = mono_mixer(audio_stereo, num_channels)
        if sr != 44100:
            resampler_mono = es.Resample(inputSampleRate=sr, outputSampleRate=44100)
            audio_mono = resampler_mono(audio_mono)

        # Compute tempo
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(audio_mono)

        # Compute keys with different profiles
        keys = {}
        for profile in ["temperley", "krumhansl", "edma"]:
            key_extractor = es.KeyExtractor(profileType=profile)
            key, scale, strength = key_extractor(audio_mono)
            keys[profile] = {"key": key, "scale": scale, "strength": float(strength)}

        # Resample to 16kHz for embedding models
        resampler_16k = es.Resample(inputSampleRate=44100, outputSampleRate=16000)
        audio_16k = resampler_16k(audio_mono)

        # Compute embeddings
        discogs_emb = models["discogs_model"](audio_16k)
        musicnn_emb = models["musicnn_model"](audio_16k)

        # Calculate mean embeddings
        discogs_mean = discogs_emb.mean(axis=0).tolist()
        musicnn_mean = musicnn_emb.mean(axis=0).tolist()

        # Music style classification

        genre_pred = models["genre_model"](discogs_emb)
        genre_probs = genre_pred.mean(axis=0)
        styles = {
            models["genre_metadata"]["classes"][i]: float(p)
            for i, p in enumerate(genre_probs)
        }

        sorted_indices = np.argsort(genre_probs)[::-1][:2]
        top_genres = [
            {
                "genre": models["genre_metadata"]["classes"][i],
                "probability": float(genre_probs[i]),
            }
            for i in sorted_indices
        ]

        # Voice/instrumental classification
        voice_pred = models["voice_model"](discogs_emb)
        voice_probs = voice_pred.mean(axis=0)
        voice = {
            models["voice_metadata"]["classes"][i]: float(p)
            for i, p in enumerate(voice_probs)
        }

        # Danceability prediction
        dance_pred = models["dance_model"](discogs_emb)
        dance_probs = dance_pred.mean(axis=0)
        danceability = float(dance_probs[0])

        # Valence/Arousal prediction
        va_pred = models["va_model"](musicnn_emb)
        va_means = va_pred.mean(axis=0)
        valence = float(va_means[0])
        arousal = float(va_means[1])

        result = {
            "tempo": float(bpm),
            "loudness": float(integrated_loudness),
            "key": keys,
            "embeddings": {"discogs-effnet": discogs_mean, "msd-musicnn": musicnn_mean},
            "music_styles": styles,
            "voice_instrumental": voice,
            "danceability": danceability,
            "valence": valence,
            "arousal": arousal,
        }

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return True
    except Exception as e:
        logging.error(
            f"Error processing {file_path} in {current_process().name}: {str(e)}"
        )
        return False


def worker_initializer(models_dir):
    global models_global
    models_global = load_models(models_dir)
    logging.info(f"{current_process().name} initialized models.")


def process_task(args):
    file_path, output_path = args
    return process_file(file_path, output_path, models_global)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel audio file analysis with Essentia"
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to save analysis results")
    parser.add_argument(
        "--models_dir", default="models", help="Directory containing models"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Collect all MP3 files recursively
    mp3_files = list(Path(input_dir).rglob("*.mp3"))
    tasks = []
    for file_path in mp3_files:
        rel_path = os.path.relpath(file_path, input_dir)
        out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".json")
        if not os.path.exists(out_path):
            tasks.append((str(file_path), out_path))

    with Pool(
        processes=args.workers,
        initializer=worker_initializer,
        initargs=(args.models_dir,),
    ) as pool:
        for success in tqdm(
            pool.imap_unordered(process_task, tasks),
            total=len(tasks),
            desc="Processing files",
        ):
            if not success:
                logging.warning("A file was skipped due to an error.")


if __name__ == "__main__":
    main()
