import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
import streamlit as st
from utils import parse_music_style


@st.cache_data(show_spinner=False)
def load_tracks_data(analysis_dir: str, audio_dir: str) -> pd.DataFrame:
    """
    Load analyzed tracks from JSON files.

    Args:
        analysis_dir (str): Directory containing analysis JSON files.
        audio_dir (str): Directory containing audio files.

    Returns:
        pd.DataFrame: DataFrame with track data.
    """
    data = []
    for json_file in Path(analysis_dir).glob("**/*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                track_data = json.load(f)
            track_data["analysis_path"] = str(json_file)
            relative_path = json_file.relative_to(analysis_dir)
            audio_path = Path(audio_dir) / relative_path.with_suffix(".mp3")
            track_data["audio_path"] = str(audio_path)
            track_data["track_id"] = json_file.stem
            data.append(track_data)
        except Exception as e:
            st.error(f"Error loading {json_file}: {str(e)}")
    return pd.DataFrame(data)


class MusicLibrary:
    def __init__(self, analysis_directory: str, audio_directory: str) -> None:
        """
        Initialize the MusicLibrary with directories for analysis and audio files.

        Args:
            analysis_directory (str): Path to the analysis directory.
            audio_directory (str): Path to the audio directory.
        """
        self.analysis_directory = analysis_directory
        self.audio_directory = audio_directory
        self.tracks_df: pd.DataFrame = load_tracks_data(
            analysis_directory, audio_directory
        )
        self.genre_stats: Dict[str, float] = {}
        self.style_stats: Dict[str, float] = {}
        self.genre_style_map: Dict[str, set] = {}
        self.compute_genre_style_stats()

    def compute_genre_style_stats(self) -> None:
        """
        Compute genre and style statistics from the track data.
        """
        genres: List[Tuple[str, float]] = []
        styles: List[Tuple[str, float]] = []
        genre_style_map: Dict[str, set] = {}

        for _, track in self.tracks_df.iterrows():
            if "music_styles" not in track:
                continue
            for genre_str, prob in track["music_styles"].items():
                parent, style = parse_music_style(genre_str)
                genres.append((parent, prob))
                styles.append((style, prob))
                if parent not in genre_style_map:
                    genre_style_map[parent] = set()
                genre_style_map[parent].add(style)

        genre_probs: Dict[str, float] = {}
        for genre, prob in genres:
            genre_probs[genre] = genre_probs.get(genre, 0) + prob

        style_probs: Dict[str, float] = {}
        for style, prob in styles:
            style_probs[style] = style_probs.get(style, 0) + prob

        self.genre_stats = genre_probs
        self.style_stats = style_probs
        self.genre_style_map = genre_style_map

    def get_track_distribution(
        self, feature: str, bins: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Get distribution statistics for a track feature.

        Args:
            feature (str): Feature (column) to analyze.
            bins (int): Number of histogram bins.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with distribution stats, or None if feature not found.
        """
        if feature not in self.tracks_df.columns:
            return None

        series = self.tracks_df[feature].dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75]).to_dict()
        stats = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "quartiles": {str(q): float(val) for q, val in quantiles.items()},
        }
        hist_values, bin_edges = np.histogram(series, bins=bins, density=True)
        stats["histogram"] = {
            "values": [float(x) for x in hist_values],
            "bin_edges": [float(x) for x in bin_edges],
        }
        return stats

    def filter_tracks(
        self,
        tempo_range: Optional[Tuple[float, float]] = None,
        has_vocals: Optional[bool] = None,
        danceability_range: Optional[Tuple[float, float]] = None,
        arousal_range: Optional[Tuple[float, float]] = None,
        valence_range: Optional[Tuple[float, float]] = None,
        key: Optional[List[str]] = None,
        scale: Optional[str] = None,
        genres: Optional[List[str]] = None,
        styles: Optional[List[str]] = None,
        profile: str = "temperley",
        loudness_range: Optional[Tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Filter tracks based on provided musical criteria.

        Returns:
            pd.DataFrame: Filtered tracks.
        """
        filtered_df = self.tracks_df.copy()

        if genres or styles:

            def match_criteria(track: pd.Series) -> bool:
                if "music_styles" not in track:
                    return False
                threshold = 0.1
                for genre_str, prob in track["music_styles"].items():
                    if prob < threshold:
                        continue
                    parent, style = parse_music_style(genre_str)
                    if genres and styles:
                        if parent in genres and style in styles:
                            return True
                    elif genres:
                        if parent in genres:
                            return True
                    elif styles:
                        if style in styles:
                            return True
                return False

            filtered_df = filtered_df[filtered_df.apply(match_criteria, axis=1)]

        if tempo_range:
            filtered_df = filtered_df[
                (filtered_df["tempo"] >= tempo_range[0])
                & (filtered_df["tempo"] <= tempo_range[1])
            ]
        if has_vocals is not None:
            threshold = 0.8
            is_vocal = filtered_df["voice_instrumental"].apply(
                lambda x: x["voice"] > threshold
            )
            filtered_df = (
                filtered_df[is_vocal] if has_vocals else filtered_df[~is_vocal]
            )
        if danceability_range:
            filtered_df = filtered_df[
                (filtered_df["danceability"] >= danceability_range[0])
                & (filtered_df["danceability"] <= danceability_range[1])
            ]
        if arousal_range:
            db_arousal_range = (
                (arousal_range[0] + 1) * 4 + 1,
                (arousal_range[1] + 1) * 4 + 1,
            )
            filtered_df = filtered_df[
                (filtered_df["arousal"] >= db_arousal_range[0])
                & (filtered_df["arousal"] <= db_arousal_range[1])
            ]
        if valence_range:
            db_valence_range = (
                (valence_range[0] + 1) * 4 + 1,
                (valence_range[1] + 1) * 4 + 1,
            )
            filtered_df = filtered_df[
                (filtered_df["valence"] >= db_valence_range[0])
                & (filtered_df["valence"] <= db_valence_range[1])
            ]
        if key is not None or scale is not None:
            filtered_df = filtered_df[
                filtered_df["key"].apply(
                    lambda x: (x[profile]["key"] in key if key is not None else True)
                    and (
                        x[profile]["scale"].lower() == scale.lower()
                        if scale is not None
                        else True
                    )
                )
            ]
        if loudness_range:
            filtered_df = filtered_df[
                (filtered_df["loudness"] >= loudness_range[0])
                & (filtered_df["loudness"] <= loudness_range[1])
            ]
        return filtered_df

    def find_similar_tracks(
        self,
        query_track_id: str,
        embedding_type: str = "discogs-effnet",
        n_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find tracks similar to a query track using embeddings.

        Args:
            query_track_id (str): ID of the reference track.
            embedding_type (str): Which embedding to use.
            n_results (int): Maximum number of similar tracks to return.

        Returns:
            List[Dict[str, Any]]: List of similar track dictionaries.
        """
        query_track = self.tracks_df[self.tracks_df["track_id"] == query_track_id].iloc[
            0
        ]
        query_embedding = np.array(query_track["embeddings"][embedding_type])
        similarities = []
        for _, track in self.tracks_df.iterrows():
            if track["track_id"] != query_track_id:
                track_embedding = np.array(track["embeddings"][embedding_type])
                similarity = 1 - distance.cosine(query_embedding, track_embedding)
                similarities.append(
                    {
                        "track_id": track["track_id"],
                        "audio_path": track["audio_path"],
                        "similarity": similarity,
                    }
                )
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)[
            :n_results
        ]

    def get_genre_style_recommendations(
        self,
        tracks: pd.DataFrame,
        selected_genres: Optional[List[str]] = None,
        selected_styles: Optional[List[str]] = None,
        n_recommendations: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Recommend tracks based on genre and style similarity.

        Returns:
            List[Dict[str, Any]]: Recommended tracks.
        """
        candidates = self.filter_tracks(genres=selected_genres, styles=selected_styles)
        input_genres, input_styles = set(), set()
        for _, track in tracks.iterrows():
            if "music_styles" in track:
                for genre_str, prob in track["music_styles"].items():
                    if prob < 0.1:
                        continue
                    parent, style = parse_music_style(genre_str)
                    if not selected_genres or parent in selected_genres:
                        input_genres.add(parent)
                    if not selected_styles or style in selected_styles:
                        input_styles.add(style)
        recommendations = []
        for _, track in candidates.iterrows():
            if "music_styles" not in track:
                continue
            track_genres, track_styles = set(), set()
            for genre_str, prob in track["music_styles"].items():
                if prob < 0.1:
                    continue
                parent, style = parse_music_style(genre_str)
                if not selected_genres or parent in selected_genres:
                    track_genres.add(parent)
                if not selected_styles or style in selected_styles:
                    track_styles.add(style)
            genre_sim = (
                (len(input_genres & track_genres) / len(input_genres | track_genres))
                if input_genres
                else 0
            )
            style_sim = (
                (len(input_styles & track_styles) / len(input_styles | track_styles))
                if input_styles
                else 0
            )
            similarity = (genre_sim + style_sim) / 2
            if similarity > 0:
                recommendations.append(
                    {
                        "track_id": track["track_id"],
                        "audio_path": track["audio_path"],
                        "similarity": similarity,
                        "matched_genres": list(track_genres),
                        "matched_styles": list(track_styles),
                    }
                )
        return sorted(recommendations, key=lambda x: x["similarity"], reverse=True)[
            :n_recommendations
        ]
