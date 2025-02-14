import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.spatial.distance as distance
import streamlit as st


class MusicLibrary:
    def __init__(self, analysis_dir: str, audio_dir: str):
        """Initialize the music library with analyzed tracks"""
        self.analysis_dir = analysis_dir
        self.audio_dir = audio_dir
        self.tracks_df = None
        self.genre_stats = None
        self.style_stats = None
        self.genre_style_map = {}
        self.load_tracks()
        self.compute_genre_style_stats()

    def load_tracks(self):
        """Load all analyzed tracks from JSON files"""
        data = []
        for json_file in Path(self.analysis_dir).glob("**/*.json"):
            try:
                with open(json_file) as f:
                    track_data = json.load(f)
                    track_data["analysis_path"] = str(json_file)
                    relative_path = Path(str(json_file)).relative_to(self.analysis_dir)
                    audio_path = Path(self.audio_dir) / relative_path.with_suffix(
                        ".mp3"
                    )
                    track_data["audio_path"] = str(audio_path)
                    track_data["track_id"] = json_file.stem
                    data.append(track_data)
            except Exception as e:
                st.error(f"Error loading {json_file}: {str(e)}")

        self.tracks_df = pd.DataFrame(data)

    def get_track_distribution(self, feature: str, bins: int = 10) -> Dict:
        """Get detailed distribution statistics for a musical feature"""
        if feature not in self.tracks_df.columns:
            return None

        stats = {
            "mean": float(self.tracks_df[feature].mean()),
            "std": float(self.tracks_df[feature].std()),
            "min": float(self.tracks_df[feature].min()),
            "max": float(self.tracks_df[feature].max()),
            "quartiles": {
                str(q): float(val)
                for q, val in self.tracks_df[feature]
                .quantile([0.25, 0.5, 0.75])
                .items()
            },
        }

        hist_values, bin_edges = np.histogram(
            self.tracks_df[feature].dropna(), bins=bins, density=True
        )

        stats["histogram"] = {
            "values": [float(x) for x in hist_values],
            "bin_edges": [float(x) for x in bin_edges],
        }

        return stats

    def compute_genre_style_stats(self):
        """Compute genre and style distribution statistics from embeddings"""
        genres = []
        styles = []
        genre_style_map = {}

        for _, track in self.tracks_df.iterrows():
            if "music_styles" not in track:
                continue

            for genre_str, prob in track["music_styles"].items():
                if "---" in genre_str:
                    parts = genre_str.split("---", 1)
                else:
                    parts = [genre_str, "unknown-style"]

                parent = parts[0].strip()
                style = parts[1].strip() if len(parts) > 1 else "unknown-style"

                genres.append((parent, prob))
                styles.append((style, prob))

                if parent not in genre_style_map:
                    genre_style_map[parent] = set()
                genre_style_map[parent].add(style)

        # Sum probabilities for each genre and style
        genre_probs = {}
        for genre, prob in genres:
            genre_probs[genre] = genre_probs.get(genre, 0) + prob

        style_probs = {}
        for style, prob in styles:
            style_probs[style] = style_probs.get(style, 0) + prob

        self.genre_stats = genre_probs
        self.style_stats = style_probs
        self.genre_style_map = genre_style_map

    def filter_tracks(
        self,
        tempo_range: Tuple[float, float] = None,
        has_vocals: bool = None,
        danceability_range: Tuple[float, float] = None,
        arousal_range: Tuple[float, float] = None,
        valence_range: Tuple[float, float] = None,
        key: List[str] = None,
        scale: str = None,
        genres: List[str] = None,
        styles: List[str] = None,
        profile: str = "temperley",
        loudness_range: Tuple[float, float] = None,
    ) -> pd.DataFrame:
        """Filter tracks based on musical criteria and genre/style"""
        filtered_df = self.tracks_df.copy()

        # Apply genre and style filtering first with a minimum probability threshold
        if genres or styles:

            def match_criteria(track):
                if "music_styles" not in track:
                    return False

                matches = False
                threshold = 0.1  # Minimum probability threshold

                for genre_str, prob in track["music_styles"].items():
                    if prob < threshold:
                        continue

                    if "---" in genre_str:
                        parts = genre_str.split("---", 1)
                    elif "—" in genre_str:
                        parts = genre_str.split("—", 1)
                    else:
                        parts = [genre_str, "unknown-style"]

                    parent = parts[0].strip()
                    style = parts[1].strip() if len(parts) > 1 else "unknown-style"

                    if genres and styles:
                        # Both genre and style must match
                        if parent in genres and style in styles:
                            matches = True
                            break
                    elif genres:
                        # Only genre needs to match
                        if parent in genres:
                            matches = True
                            break
                    elif styles:
                        # Only style needs to match
                        if style in styles:
                            matches = True
                            break

                return matches

            filtered_df = filtered_df[filtered_df.apply(match_criteria, axis=1)]

        # Apply other filters after genre/style filtering
        if tempo_range:
            filtered_df = filtered_df[
                (filtered_df["tempo"] >= tempo_range[0])
                & (filtered_df["tempo"] <= tempo_range[1])
            ]

        if has_vocals is not None:
            threshold = 0.5
            is_vocal = filtered_df["voice_instrumental"].apply(
                lambda x: x["voice"] > threshold
            )
            if has_vocals:
                filtered_df = filtered_df[is_vocal]
            else:
                filtered_df = filtered_df[~is_vocal]

        if danceability_range:
            filtered_df = filtered_df[
                (filtered_df["danceability"] >= danceability_range[0])
                & (filtered_df["danceability"] <= danceability_range[1])
            ]

        if arousal_range:
            # Convert from [-1,1] to [1,9] range for filtering
            db_arousal_range = (
                (arousal_range[0] + 1) * 4 + 1,
                (arousal_range[1] + 1) * 4 + 1,
            )
            filtered_df = filtered_df[
                (filtered_df["arousal"] >= db_arousal_range[0])
                & (filtered_df["arousal"] <= db_arousal_range[1])
            ]

        if valence_range:
            # Convert from [-1,1] to [1,9] range for filtering
            db_valence_range = (
                (valence_range[0] + 1) * 4 + 1,
                (valence_range[1] + 1) * 4 + 1,
            )
            filtered_df = filtered_df[
                (filtered_df["valence"] >= db_valence_range[0])
                & (filtered_df["valence"] <= db_valence_range[1])
            ]

        # Updated key and scale filtering to handle multiple keys
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
    ) -> List[Dict]:
        """Find similar tracks using embeddings"""
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
        selected_genres: List[str] = None,
        selected_styles: List[str] = None,
        n_recommendations: int = 10,
    ) -> List[Dict]:
        """Get track recommendations based on genre and style similarity"""
        # First filter the candidate tracks by the selected genres/styles
        filtered_candidates = self.filter_tracks(
            genres=selected_genres, styles=selected_styles
        )

        # Get input track characteristics
        input_genres = set()
        input_styles = set()

        for _, track in tracks.iterrows():
            if "music_styles" in track:
                for genre_str, prob in track["music_styles"].items():
                    if prob < 0.1:  # Skip low probability matches
                        continue

                    if "---" in genre_str:
                        parts = genre_str.split("---", 1)
                    elif "—" in genre_str:
                        parts = genre_str.split("—", 1)
                    else:
                        parts = [genre_str, "unknown-style"]

                    parent = parts[0].strip()
                    style = parts[1].strip() if len(parts) > 1 else "unknown-style"

                    if not selected_genres or parent in selected_genres:
                        input_genres.add(parent)
                    if not selected_styles or style in selected_styles:
                        input_styles.add(style)

        recommendations = []
        for _, track in filtered_candidates.iterrows():
            if "music_styles" not in track:
                continue

            track_genres = set()
            track_styles = set()

            # Only consider high-probability genres/styles
            for genre_str, prob in track["music_styles"].items():
                if prob < 0.1:
                    continue

                if "---" in genre_str:
                    parts = genre_str.split("---", 1)
                elif "—" in genre_str:
                    parts = genre_str.split("—", 1)
                else:
                    parts = [genre_str, "unknown-style"]

                parent = parts[0].strip()
                style = parts[1].strip() if len(parts) > 1 else "unknown-style"

                if not selected_genres or parent in selected_genres:
                    track_genres.add(parent)
                if not selected_styles or style in selected_styles:
                    track_styles.add(style)

            # Calculate similarity based on matching genres and styles
            genre_sim = (
                len(input_genres & track_genres) / len(input_genres | track_genres)
                if input_genres
                else 0
            )
            style_sim = (
                len(input_styles & track_styles) / len(input_styles | track_styles)
                if input_styles
                else 0
            )

            similarity = (genre_sim + style_sim) / 2

            if similarity > 0:  # Only include tracks with some similarity
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


def file_exists(path: str) -> bool:
    """Check if a file exists and is accessible"""
    return Path(path).is_file()


def safe_audio_player(audio_path: str) -> None:
    """Safely display audio player with error handling"""
    try:
        if file_exists(audio_path):
            st.audio(audio_path)
        else:
            st.error(f"Audio file not found: {audio_path}")
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")


# Define the playlists directory constant
PLAYLISTS_DIR = "./playlists"


def create_m3u8_playlist(tracks: List[Dict], output_path: str):
    """Create M3U8 playlist file with file existence check"""
    valid_tracks = []
    for track in tracks:
        if file_exists(track["audio_path"]):
            valid_tracks.append(track)
        else:
            st.warning(f"Skipping unavailable track: {track['track_id']}")

    if not valid_tracks:
        st.error("No valid tracks found for playlist")
        return False

    # If output_path is not absolute, join with PLAYLISTS_DIR and ensure the folder exists
    if not os.path.isabs(output_path):
        output_path = os.path.join(PLAYLISTS_DIR, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for track in valid_tracks:
            f.write(f'#EXTINF:-1,{track["track_id"]}\n')
            f.write(f'{track["audio_path"]}\n')
    return True


def paginate_tracks(tracks_data: List[Dict], page_key: str, tracks_per_page: int = 10):
    """Helper function to handle pagination for tracks"""
    if f"current_page_{page_key}" not in st.session_state:
        st.session_state[f"current_page_{page_key}"] = 0

    total_pages = (len(tracks_data) - 1) // tracks_per_page + 1
    current_page = st.session_state[f"current_page_{page_key}"]

    # Simplified pagination controls
    st.write(f"Page {current_page + 1} of {total_pages}")

    cols = st.columns(4)
    if cols[0].button("◀◀", key=f"first_{page_key}", disabled=current_page == 0):
        st.session_state[f"current_page_{page_key}"] = 0

    if cols[1].button("◀", key=f"prev_{page_key}", disabled=current_page == 0):
        st.session_state[f"current_page_{page_key}"] = max(0, current_page - 1)

    if cols[2].button(
        "▶", key=f"next_{page_key}", disabled=current_page >= total_pages - 1
    ):
        st.session_state[f"current_page_{page_key}"] = min(
            total_pages - 1, current_page + 1
        )

    if cols[3].button(
        "▶▶", key=f"last_{page_key}", disabled=current_page >= total_pages - 1
    ):
        st.session_state[f"current_page_{page_key}"] = total_pages - 1

    start_idx = current_page * tracks_per_page
    end_idx = start_idx + tracks_per_page
    return tracks_data[start_idx:end_idx]


def render_genre_style_section(library: MusicLibrary):
    """Render the genre and style analysis section"""
    st.header("Genre & Style Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Genre distribution
        st.subheader("Genre Distribution")
        genre_data = pd.DataFrame.from_dict(
            library.genre_stats, orient="index", columns=["count"]
        ).sort_values("count", ascending=False)

        fig_genres = px.bar(
            genre_data.head(20),
            title="Top 20 Genres",
            labels={"index": "Genre", "count": "Cumulative Probability"},
        )
        st.plotly_chart(fig_genres)

        selected_genres = st.multiselect(
            "Select Genres", options=sorted(library.genre_stats.keys())
        )

    with col2:
        # Style distribution - show only styles for selected genres
        st.subheader("Style Distribution")
        available_styles = set()
        if selected_genres:
            for genre in selected_genres:
                if genre in library.genre_style_map:
                    available_styles.update(library.genre_style_map[genre])
        else:
            available_styles = set(library.style_stats.keys())

        style_data = pd.DataFrame.from_dict(
            {k: v for k, v in library.style_stats.items() if k in available_styles},
            orient="index",
            columns=["count"],
        ).sort_values("count", ascending=False)

        fig_styles = px.bar(
            style_data.head(20),
            title="Top 20 Styles",
            labels={"index": "Style", "count": "Cumulative Probability"},
        )
        st.plotly_chart(fig_styles)

        selected_styles = st.multiselect(
            "Select Styles", options=sorted(available_styles)
        )

    if selected_genres or selected_styles:
        filtered_tracks = library.filter_tracks(
            genres=selected_genres, styles=selected_styles
        )

        st.subheader(f"Found {len(filtered_tracks)} matching tracks")

        if not filtered_tracks.empty:
            st.write("Sample tracks:")
            # Update to use pagination
            paginated_tracks = paginate_tracks(
                filtered_tracks.to_dict("records"), "genre_style"
            )
            for track in paginated_tracks:
                col1, col2 = st.columns([3, 1])
                with col1:
                    safe_audio_player(track["audio_path"])
                with col2:
                    # Show only the matching genres/styles
                    matched_info = []
                    if "music_styles" in track:
                        for genre_str in track["music_styles"].keys():
                            if "---" in genre_str:
                                parts = genre_str.split("---", 1)
                            elif "—" in genre_str:
                                parts = genre_str.split("—", 1)
                            else:
                                parts = [genre_str, "unknown-style"]

                            parent = parts[0].strip()
                            style = (
                                parts[1].strip() if len(parts) > 1 else "unknown-style"
                            )

                            if (not selected_genres or parent in selected_genres) and (
                                not selected_styles or style in selected_styles
                            ):
                                matched_info.append(f"{parent} - {style}")

                        if matched_info:
                            st.write("Matched categories:", ", ".join(matched_info))

            if st.button("Get Similar Tracks"):
                recommendations = library.get_genre_style_recommendations(
                    filtered_tracks,
                    selected_genres=selected_genres,
                    selected_styles=selected_styles,
                )

                if recommendations:
                    st.subheader("Recommended Tracks")
                    # Update to use pagination
                    paginated_recommendations = paginate_tracks(
                        recommendations, "genre_recommendations"
                    )
                    for rec in paginated_recommendations:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            safe_audio_player(rec["audio_path"])
                        with col2:
                            st.write(f"Similarity: {rec['similarity']:.2f}")
                            if rec["matched_genres"]:
                                st.write("Genres:", ", ".join(rec["matched_genres"]))
                            if rec["matched_styles"]:
                                st.write("Styles:", ", ".join(rec["matched_styles"]))
                else:
                    st.warning(
                        "No similar tracks found matching the selected criteria."
                    )

                if st.button("Export Recommendations as M3U8"):
                    with st.spinner("Exporting recommendations playlist..."):
                        playlist_path = "genre_recommendations.m3u8"
                        if create_m3u8_playlist(recommendations, playlist_path):
                            st.success(f"Playlist exported to {playlist_path}")
                        else:
                            st.error("Failed to export recommendations playlist.")


def display_descriptor_statistics(library: MusicLibrary):
    """Display statistics for various musical descriptors"""
    # First, create derived voice probability column
    library.tracks_df["voice_prob"] = library.tracks_df["voice_instrumental"].apply(
        lambda x: x["voice"]
    )

    descriptors = [
        ("tempo", None),
        ("danceability", None),
        ("arousal", lambda x: (x - 1) / 4 - 1),
        ("valence", lambda x: (x - 1) / 4 - 1),
        ("loudness", None),
        ("voice_prob", None),  # Use the derived column instead
    ]

    for i in range(0, len(descriptors), 3):  # Changed from 2 to 3
        cols = st.columns(3)  # Changed from 2 to 3
        for j in range(3):  # Changed from 2 to 3
            if i + j < len(descriptors):
                descriptor, normalizer = descriptors[i + j]
                with cols[j]:
                    # Special case for voice probability to show more readable title
                    title = (
                        "Voice Probability"
                        if descriptor == "voice_prob"
                        else descriptor.title()
                    )
                    st.subheader(title, divider="gray")

                    stats = library.get_track_distribution(descriptor)

                    if stats:
                        mean = stats["mean"]
                        std = stats["std"]
                        min_val = stats["min"]
                        max_val = stats["max"]

                        if normalizer:
                            mean = normalizer(mean)
                            std = normalizer(std * 4) / 4
                            min_val = normalizer(min_val)
                            max_val = normalizer(max_val)

                        # More compact metrics display
                        metric_cols = st.columns(4)
                        for idx, (label, value) in enumerate(
                            [
                                ("Mean", mean),
                                ("Std", std),
                                ("Min", min_val),
                                ("Max", max_val),
                            ]
                        ):
                            with metric_cols[idx]:
                                st.caption(label)
                                st.markdown(f"##### {value:.2f}")

                        # Create histogram
                        df = library.tracks_df.copy()
                        if normalizer:
                            df[descriptor] = df[descriptor].apply(normalizer)

                        fig = px.histogram(
                            df,
                            x=descriptor,
                            nbins=20,
                            title=None,
                        )

                        fig.update_layout(
                            height=150,
                            margin=dict(l=20, r=20, t=20, b=20),
                            bargap=0.05,
                            showlegend=False,
                            xaxis_title=None,
                            yaxis_title=None,
                        )

                        st.plotly_chart(fig, use_container_width=True)


def compute_track_statistics(tracks_df: pd.DataFrame) -> Dict:
    """Compute statistics for musical descriptors across tracks"""
    # First create voice probability series
    voice_prob = tracks_df["voice_instrumental"].apply(lambda x: x["voice"])

    descriptors = {
        "tempo": {"label": "Tempo (BPM)", "normalize": None},
        "danceability": {"label": "Danceability", "normalize": None},
        "arousal": {"label": "Arousal", "normalize": lambda x: (x - 1) / 4 - 1},
        "valence": {"label": "Valence", "normalize": lambda x: (x - 1) / 4 - 1},
        "loudness": {"label": "Loudness (dB)", "normalize": None},
        "voice_prob": {
            "label": "Voice Probability",
            "normalize": None,
            "values": voice_prob,
        },
    }

    stats = {}
    for desc, config in descriptors.items():
        if "values" in config:  # Handle pre-computed values
            values = config["values"]
        elif desc in tracks_df.columns:  # Handle direct columns
            values = tracks_df[desc]
        else:
            continue

        if config["normalize"]:
            values = values.apply(config["normalize"])

        stats[desc] = {
            "label": config["label"],
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "values": values.tolist(),
        }

    # Add key statistics
    if "key" in tracks_df.columns:
        key_profile = "temperley"  # Use default profile
        key_counts = (
            tracks_df["key"].apply(lambda x: x[key_profile]["key"]).value_counts()
        )
        stats["key"] = {
            "label": "Key Distribution",
            "values": key_counts.to_dict(),
        }

    return stats


def display_summary_stats(tracks_df: pd.DataFrame):
    """Display summary statistics, distributions, and key distribution for matching tracks"""
    if tracks_df.empty:
        return

    stats = compute_track_statistics(tracks_df)

    # Summary statistics
    st.subheader("Summary Statistics", divider="gray")
    stats_data = []
    for desc, data in stats.items():
        if desc != "key":
            row = {"Descriptor": data["label"]}
            if "mean" in data:
                row.update(
                    {
                        "Mean": f"{data['mean']:.2f}",
                        "Std": f"{data['std']:.2f}",
                        "Min": f"{data['min']:.2f}",
                        "Max": f"{data['max']:.2f}",
                        "Median": f"{data['median']:.2f}",
                    }
                )
                stats_data.append(row)
    if stats_data:
        st.dataframe(
            pd.DataFrame(stats_data).set_index("Descriptor"),
            hide_index=False,
            use_container_width=True,
        )

    # Distribution plots
    st.subheader("Distributions", divider="gray")
    numeric_stats = {k: v for k, v in stats.items() if k != "key"}
    if numeric_stats:
        cols = st.columns(len(numeric_stats))
        for idx, (desc, data) in enumerate(numeric_stats.items()):
            with cols[idx]:
                if "values" in data:
                    fig = px.histogram(
                        x=data["values"],
                        title=data["label"],
                        labels={"x": desc, "y": "Count"},
                    )
                    fig.update_layout(
                        height=150,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=False,
                        title_y=0.95,
                        title_x=0.5,
                        xaxis_title=None,
                        yaxis_title=None,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Key distribution chart
    if "key" in stats:
        st.subheader("Key Distribution", divider="gray")
        key_data = pd.DataFrame.from_dict(
            stats["key"]["values"], orient="index", columns=["count"]
        ).sort_index()
        fig = px.bar(
            key_data,
            title="Key Distribution",
            labels={"index": "Key", "count": "Count"},
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_detailed_tracks(tracks_df: pd.DataFrame, tracks_per_page: int = 5):
    """Display individual track values and individual tracks for matching tracks"""
    if tracks_df.empty:
        return

    # Store the tracks in session state if not already there
    if "current_filtered_tracks" not in st.session_state:
        st.session_state.current_filtered_tracks = tracks_df

    # Use the stored tracks instead of the parameter
    current_tracks = st.session_state.current_filtered_tracks

    # Rest of the display_detailed_tracks function remains the same
    # Individual track values table
    st.subheader("Individual Track Values", divider="gray")
    track_data = []

    # Calculate statistics for all tracks
    stats = compute_track_statistics(current_tracks)

    for _, track in current_tracks.iterrows():
        row = {"Track ID": track["track_id"]}
        for desc, data in stats.items():
            if desc == "key":
                # Handle key differently
                key_info = track[desc]["temperley"]
                row[data["label"]] = f"{key_info['key']} {key_info['scale']}"
            elif desc == "voice_prob":
                # Handle voice probability
                voice_prob = track["voice_instrumental"]["voice"]
                row[data["label"]] = f"{voice_prob:.2f}"
            else:
                # Handle numeric descriptors
                value = track[desc]
                if desc in ["arousal", "valence"]:
                    value = (value - 1) / 4 - 1
                row[data["label"]] = f"{value:.2f}"
        track_data.append(row)

    if track_data:  # Only create dataframe if we have data
        st.dataframe(
            pd.DataFrame(track_data).set_index("Track ID"),
            hide_index=False,
            use_container_width=True,
        )

    # Individual tracks display with integrated audio players
    st.subheader("Individual Tracks", divider="gray")

    # Get tracks for current page
    tracks_records = current_tracks.to_dict("records")
    paginated_tracks = paginate_tracks(
        tracks_records, "descriptor_tracks", tracks_per_page
    )

    # Display tracks with audio players
    for track in paginated_tracks:
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col1:
            st.write(f"**{track['track_id']}**")
        with col2:
            st.audio(track["audio_path"], format="audio/mp3")
        with col3:
            metrics = []
            for desc, data in stats.items():
                if desc == "key":
                    key_info = track[desc]["temperley"]
                    metrics.append(
                        f"{data['label']}: {key_info['key']} {key_info['scale']}"
                    )
                elif desc == "voice_prob":
                    voice_prob = track["voice_instrumental"]["voice"]
                    metrics.append(f"{data['label']}: {voice_prob:.2f}")
                else:
                    value = track[desc]
                    if desc in ["arousal", "valence"]:
                        value = (value - 1) / 4 - 1
                    metrics.append(f"{data['label']}: {value:.2f}")
            st.write("  \n".join(metrics))
        st.markdown("---")


def display_track_statistics(tracks_df: pd.DataFrame):
    """Display summary statistics, distributions, and key distribution for matching tracks"""
    if tracks_df.empty:
        return

    stats = compute_track_statistics(tracks_df)

    # Summary statistics
    st.subheader("Summary Statistics", divider="gray")
    stats_data = []
    for desc, data in stats.items():
        if desc != "key":
            row = {"Descriptor": data["label"]}
            if "mean" in data:
                row.update(
                    {
                        "Mean": f"{data['mean']:.2f}",
                        "Std": f"{data['std']:.2f}",
                        "Min": f"{data['min']:.2f}",
                        "Max": f"{data['max']:.2f}",
                        "Median": f"{data['median']:.2f}",
                    }
                )
                stats_data.append(row)
    if stats_data:
        st.dataframe(
            pd.DataFrame(stats_data).set_index("Descriptor"),
            hide_index=False,
            use_container_width=True,
        )

    # Distribution plots
    st.subheader("Distributions", divider="gray")
    numeric_stats = {k: v for k, v in stats.items() if k != "key"}
    if numeric_stats:
        cols = st.columns(len(numeric_stats))
        for idx, (desc, data) in enumerate(numeric_stats.items()):
            with cols[idx]:
                if "values" in data:
                    fig = px.histogram(
                        x=data["values"],
                        title=data["label"],
                        labels={"x": desc, "y": "Count"},
                    )
                    fig.update_layout(
                        height=150,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=False,
                        title_y=0.95,
                        title_x=0.5,
                        xaxis_title=None,
                        yaxis_title=None,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Key distribution chart
    if "key" in stats:
        st.subheader("Key Distribution", divider="gray")
        key_data = pd.DataFrame.from_dict(
            stats["key"]["values"], orient="index", columns=["count"]
        ).sort_index()
        fig = px.bar(
            key_data,
            title="Key Distribution",
            labels={"index": "Key", "count": "Count"},
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_track_statistics(tracks_df: pd.DataFrame):
    """Original combined function kept for reference (not called anymore)"""
    # ...existing code...
    pass


def display_track_statistics(tracks_df: pd.DataFrame):
    """Display comprehensive statistics for matching tracks"""
    if tracks_df.empty:
        return

    stats = compute_track_statistics(tracks_df)

    # More compact summary statistics table
    st.subheader("Summary Statistics", divider="gray")
    stats_data = []
    for desc, data in stats.items():
        if desc != "key":  # Only include numeric descriptors in summary stats
            row = {"Descriptor": data["label"]}
            if "mean" in data:  # Check if numeric statistics are available
                row.update(
                    {
                        "Mean": f"{data['mean']:.2f}",
                        "Std": f"{data['std']:.2f}",
                        "Min": f"{data['min']:.2f}",
                        "Max": f"{data['max']:.2f}",
                        "Median": f"{data['median']:.2f}",
                    }
                )
                stats_data.append(row)

    if stats_data:  # Only display if we have numeric statistics
        st.dataframe(
            pd.DataFrame(stats_data).set_index("Descriptor"),
            hide_index=False,
            use_container_width=True,
        )

    # Compact distribution plots
    st.subheader("Distributions", divider="gray")
    numeric_stats = {k: v for k, v in stats.items() if k != "key"}
    if numeric_stats:
        cols = st.columns(len(numeric_stats))
        for idx, (desc, data) in enumerate(numeric_stats.items()):
            with cols[idx]:
                if "values" in data:  # Check if we have numeric values to plot
                    fig = px.histogram(
                        x=data["values"],
                        title=data["label"],
                        labels={"x": desc, "y": "Count"},
                    )
                    fig.update_layout(
                        height=150,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=False,
                        title_y=0.95,
                        title_x=0.5,
                        xaxis_title=None,
                        yaxis_title=None,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # key distribution chart
    if "key" in stats:
        st.subheader("Key Distribution", divider="gray")
        key_data = pd.DataFrame.from_dict(
            stats["key"]["values"], orient="index", columns=["count"]
        ).sort_index()

        fig = px.bar(
            key_data,
            title="Key Distribution",
            labels={"index": "Key", "count": "Count"},
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Compact track values table
    st.subheader("Individual Track Values", divider="gray")
    track_data = []
    for _, track in tracks_df.iterrows():
        row = {"Track ID": track["track_id"]}
        for desc, data in stats.items():
            if desc == "key":
                # Handle key differently
                key_info = track[desc]["temperley"]
                row[data["label"]] = f"{key_info['key']} {key_info['scale']}"
            elif desc == "voice_prob":
                # Handle voice probability
                voice_prob = track["voice_instrumental"]["voice"]
                row[data["label"]] = f"{voice_prob:.2f}"
            else:
                # Handle numeric descriptors
                value = track[desc]
                if desc in ["arousal", "valence"]:
                    value = (value - 1) / 4 - 1
                row[data["label"]] = f"{value:.2f}"
        track_data.append(row)

    st.dataframe(
        pd.DataFrame(track_data).set_index("Track ID"),
        hide_index=False,
        use_container_width=True,
    )

    # Reorganized track display with integrated audio players
    st.subheader("Individual Tracks", divider="gray")

    # Create a more compact table with audio players
    for _, track in tracks_df.iterrows():
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

        # Track ID and basic info in first column
        with col1:
            st.write(f"**{track['track_id']}**")

        # Audio player in second column
        with col2:
            st.audio(track["audio_path"], format="audio/mp3")

        # Track metrics in third column
        with col3:
            metrics = []
            for desc, data in stats.items():
                if desc == "key":
                    key_info = track[desc]["temperley"]
                    metrics.append(
                        f"{data['label']}: {key_info['key']} {key_info['scale']}"
                    )
                elif desc == "voice_prob":
                    voice_prob = track["voice_instrumental"]["voice"]
                    metrics.append(f"{data['label']}: {voice_prob:.2f}")
                else:
                    value = track[desc]
                    if desc in ["arousal", "valence"]:
                        value = (value - 1) / 4 - 1
                    metrics.append(f"{data['label']}: {value:.2f}")
            st.write("  \n".join(metrics))

        # Light divider between tracks
        st.markdown("---")


def analyze_similarity_overlap(
    effnet_tracks: List[Dict], musicnn_tracks: List[Dict]
) -> Dict:
    """Analyze overlap between two sets of similar tracks"""
    effnet_ids = {t["track_id"] for t in effnet_tracks}
    musicnn_ids = {t["track_id"] for t in musicnn_tracks}

    overlap_ids = effnet_ids & musicnn_ids

    # Get detailed info for overlapping tracks
    overlap_tracks = []
    for track_id in overlap_ids:
        effnet_track = next(t for t in effnet_tracks if t["track_id"] == track_id)
        musicnn_track = next(t for t in musicnn_tracks if t["track_id"] == track_id)
        overlap_tracks.append(
            {
                "track_id": track_id,
                "audio_path": effnet_track["audio_path"],
                "effnet_similarity": effnet_track["similarity"],
                "musicnn_similarity": musicnn_track["similarity"],
                "avg_similarity": (
                    effnet_track["similarity"] + musicnn_track["similarity"]
                )
                / 2,
            }
        )

    # Sort by average similarity
    overlap_tracks.sort(key=lambda x: x["avg_similarity"], reverse=True)

    return {
        "overlap_count": len(overlap_ids),
        "effnet_unique": len(effnet_ids - musicnn_ids),
        "musicnn_unique": len(musicnn_ids - effnet_ids),
        "overlap_tracks": overlap_tracks,
    }


def get_valid_directories() -> List[Tuple[str, str]]:
    """Get list of valid directories in the project root and their first-level subdirectories"""
    project_root = Path(__file__).parent
    dirs = []

    # Get all root directories and sort them first
    root_dirs = sorted(
        [d for d in project_root.iterdir() if d.is_dir()], key=lambda x: x.name.lower()
    )

    # Process each root directory and its subdirectories
    for root_dir in root_dirs:
        # Add the root directory
        dirs.append((root_dir.name, str(root_dir)))

        # Get and sort subdirectories
        try:
            subdirs = sorted(
                [d for d in root_dir.iterdir() if d.is_dir()],
                key=lambda x: x.name.lower(),
            )
            for subdir in subdirs:
                dirs.append((f"  └─ {subdir.name}", str(subdir)))
        except Exception:
            pass

    return dirs


def validate_directory(path: str) -> Tuple[bool, str]:
    """Validate if a directory exists and is accessible
    Returns: (is_valid, error_message)
    """
    try:
        path = Path(path)
        if not path.exists():
            return False, "Directory does not exist"
        if not path.is_dir():
            return False, "Path is not a directory"
        # Try to list directory contents to check permissions
        next(path.iterdir(), None)
        return True, ""
    except PermissionError:
        return False, "Permission denied"
    except Exception as e:
        return False, str(e)


def main():
    st.set_page_config(page_title="Music Playlist Generator", layout="wide")
    st.title("Music Playlist Generator")

    # Directory configuration section
    st.header("Library Configuration", divider=True)

    # Get list of valid project directories
    project_dirs = get_valid_directories()
    dir_options = [(display, path) for display, path in project_dirs]

    # Analysis directory selection
    st.subheader("Analysis Directory")
    analysis_col1, analysis_col2, analysis_col3 = st.columns([1, 2, 1])

    with analysis_col1:
        use_project_dir_analysis = st.checkbox(
            "Use Project Directory", value=True, key="use_proj_analysis"
        )

    with analysis_col2:
        if use_project_dir_analysis:
            # Find index of default "results" directory
            default_index = next(
                (
                    i
                    for i, (_, path) in enumerate(dir_options)
                    if path.endswith("results")
                ),
                0,
            )
            selected_analysis = st.selectbox(
                "Select analysis directory",
                options=dir_options,
                format_func=lambda x: x[0],  # Display the indented name
                index=default_index,
                key="analysis_select",
            )
            analysis_dir = selected_analysis[1]  # Use the full path
        else:
            analysis_dir = st.text_input(
                "Enter path to analysis directory:",
                value="results",
                key="analysis_input",
                help="Enter the full path to your analysis directory",
            )

    with analysis_col3:
        is_valid_analysis, error_msg = validate_directory(analysis_dir)
        if not is_valid_analysis:
            st.error(f"⚠️ {error_msg}")
        else:
            st.success("✓ Valid")

    # Audio directory selection
    st.subheader("Audio Files Directory")
    audio_col1, audio_col2, audio_col3 = st.columns([1, 2, 1])

    with audio_col1:
        use_project_dir_audio = st.checkbox(
            "Use Project Directory", value=True, key="use_proj_audio"
        )

    with audio_col2:
        if use_project_dir_audio:
            # Find index of default "data/MusAV" directory
            default_index = next(
                (i for i, (_, path) in enumerate(dir_options) if "data/MusAV" in path),
                0,
            )
            selected_audio = st.selectbox(
                "Select audio directory",
                options=dir_options,
                format_func=lambda x: x[0],  # Display the indented name
                index=default_index,
                key="audio_select",
            )
            audio_dir = selected_audio[1]  # Use the full path
        else:
            audio_dir = st.text_input(
                "Enter path to audio files directory:",
                value="data/MusAV",
                key="audio_input",
                help="Enter the full path to your audio files directory",
            )

    with audio_col3:
        is_valid_audio, error_msg = validate_directory(audio_dir)
        if not is_valid_audio:
            st.error(f"⚠️ {error_msg}")
        else:
            st.success("✓ Valid")

    # Only enable the load button if both directories are valid
    load_disabled = not (is_valid_analysis and is_valid_audio)

    load_col1, load_col2 = st.columns([3, 1])
    with load_col1:
        if load_disabled:
            st.warning(
                "Please ensure both directories are valid before loading the library"
            )
    with load_col2:
        if st.button(
            "Load/Reload Library", disabled=load_disabled, use_container_width=True
        ):
            with st.spinner("Loading library..."):
                try:
                    st.session_state.library = MusicLibrary(analysis_dir, audio_dir)
                    st.success("✅ Library loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading library: {str(e)}")

    if "library" not in st.session_state or st.session_state.library is None:
        st.info(
            "ℹ️ Library not loaded yet. Please configure and load the library to continue."
        )
        return

    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Descriptor-based", "Similarity-based", "Genre & Style"])

    with tabs[0]:
        st.header("Library Statistics")
        with st.expander("✨ Click to view Library Statistics ✨", expanded=False):
            st.caption("Statistics are computed across all tracks in the library")
            display_descriptor_statistics(st.session_state.library)

        st.header("Generate Playlist by Musical Characteristics")

        # Initialize session state for pagination
        if "filtered_tracks" not in st.session_state:
            st.session_state.filtered_tracks = None
        if "current_page" not in st.session_state:
            st.session_state.current_page = 0

        # Create three columns for better organization
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("Rhythm & Movement")
            tempo_range = st.slider(
                "Tempo Range (BPM)", min_value=0, max_value=200, value=(60, 180)
            )
            danceability_range = st.slider(
                "Danceability", min_value=0.0, max_value=1.0, value=(0.0, 1.0)
            )

        with col2:
            st.subheader("Mood & Energy")
            arousal_range = st.slider(
                "Arousal (Energy)", min_value=-1.0, max_value=1.0, value=(-1.0, 1.0)
            )
            valence_range = st.slider(
                "Valence (Mood)", min_value=-1.0, max_value=1.0, value=(-1.0, 1.0)
            )

        with col3:
            st.subheader("Sound & Voice")
            loudness_range = st.slider(
                "Loudness Range (dB)",
                min_value=-60.0,
                max_value=0.0,
                value=(-30.0, -10.0),
            )
            has_vocals = st.radio(
                "Vocal Content",
                [None, True, False],
                format_func=lambda x: "Any"
                if x is None
                else "With Vocals"
                if x
                else "Instrumental",
            )

        # Musical key section in its own row
        st.subheader("Musical Key")
        key_col1, key_col2, _ = st.columns([1, 1, 2])
        with key_col1:
            key_options = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
            selected_keys = st.multiselect("Key", options=key_options)
            key_filter = selected_keys if selected_keys else None

        with key_col2:
            scale_options = ["major", "minor"]
            selected_scale = st.selectbox("Scale", ["Any"] + scale_options)
            scale_filter = selected_scale if selected_scale != "Any" else None

        # Button layout with proper spacing
        button_cols = st.columns([1.5, 0.5, 2])
        with button_cols[0]:
            col1, col2 = st.columns([2, 1])
            with col1:
                generate_button = st.button(
                    "Generate Playlist", use_container_width=True
                )
            with col2:
                shuffle_results = st.checkbox("Shuffle", value=False)

        with button_cols[2]:
            if st.session_state.filtered_tracks is not None:
                if st.button(
                    "Export as M3U8", key="export_descriptor", use_container_width=True
                ):
                    with st.spinner("Creating playlist file..."):
                        try:
                            track_list = [
                                {
                                    "track_id": row["track_id"],
                                    "audio_path": row["audio_path"],
                                }
                                for _, row in st.session_state.filtered_tracks.iterrows()
                            ]
                            if track_list:
                                success = create_m3u8_playlist(
                                    track_list,
                                    "descriptor_based_playlist.m3u8",
                                )
                                if success:
                                    st.success("✅ Playlist exported successfully!")
                                else:
                                    st.error(
                                        "❌ No valid tracks found for playlist export."
                                    )
                            else:
                                st.error("❌ No tracks to export.")
                        except Exception as e:
                            st.error(f"❌ Error exporting playlist: {str(e)}")

        # Rest of the code remains the same
        # Create a container for the results
        results_container = st.container()

        if generate_button:
            filtered_tracks = st.session_state.library.filter_tracks(
                tempo_range=tempo_range,
                has_vocals=has_vocals,
                danceability_range=danceability_range,
                arousal_range=arousal_range,
                valence_range=valence_range,
                key=key_filter,
                scale=scale_filter,
                loudness_range=loudness_range,
            )

            if shuffle_results:
                filtered_tracks = filtered_tracks.sample(frac=1)

            # Store filtered tracks in session state
            st.session_state.filtered_tracks = filtered_tracks
            # Reset pagination when generating new results
            if "current_page_descriptor_tracks" in st.session_state:
                del st.session_state.current_page_descriptor_tracks
            # Clear the current filtered tracks to allow new ones
            if "current_filtered_tracks" in st.session_state:
                del st.session_state.current_filtered_tracks

            # Display the count of matching tracks
            st.subheader(f"Found {len(filtered_tracks)} matching tracks")

            # Wrap summary stats into an expander
            with st.expander(
                "✨ Click here to View Summary Statistics, Distributions & Key Distribution ✨"
            ):
                display_summary_stats(filtered_tracks)

        # Always show tracks if we have them in session state
        if (
            "filtered_tracks" in st.session_state
            and st.session_state.filtered_tracks is not None
        ):
            display_detailed_tracks(st.session_state.filtered_tracks)

    with tabs[1]:
        st.header("Generate Playlist by Track Similarity")
        # New search input for reference track filtering
        search_query = st.text_input("Search for reference track", value="")

        # Helper to build a display label for each track
        def get_track_label(track):
            label = track["track_id"][:8]
            if "music_styles" in track:
                # Use the top genre (if available)
                top_styles = sorted(
                    track["music_styles"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:1]
                if top_styles:
                    style_key, _ = top_styles[0]
                    label += f" ({style_key.split('---')[0].strip()})"
            return label

        # Gather all tracks from the library
        all_tracks = st.session_state.library.tracks_df.to_dict("records")
        # Filter based on the search query (if provided)
        filtered_tracks = [
            track
            for track in all_tracks
            if search_query.lower() in track["track_id"].lower()
            or search_query.lower() in get_track_label(track).lower()
        ]
        # Fallback to all tracks if search is empty or no results found
        if not search_query or not filtered_tracks:
            filtered_tracks = all_tracks

        # Build options tuple: (display label, track_id)
        options = [
            (get_track_label(track), track["track_id"]) for track in filtered_tracks
        ]
        options.sort(key=lambda x: x[0])

        selected_label, selected_track = st.selectbox(
            "Select a reference track", options, format_func=lambda opt: opt[0]
        )

        # Display reference track info and player
        selected_track_info = st.session_state.library.tracks_df[
            st.session_state.library.tracks_df["track_id"] == selected_track
        ].iloc[0]

        st.subheader("Reference Track")
        safe_audio_player(selected_track_info["audio_path"])

        # Add descriptor stats for reference track
        descriptor_stats = compute_track_statistics(pd.DataFrame([selected_track_info]))
        cols = st.columns(4)
        for idx, (desc, data) in enumerate(descriptor_stats.items()):
            with cols[idx % 4]:
                if desc == "key":
                    key_info = selected_track_info["key"]["temperley"]
                    st.caption(data["label"])
                    st.subheader(f"{key_info['key']} {key_info['scale']}")
                elif desc == "voice_prob":
                    st.caption(data["label"])
                    st.subheader(
                        f"{selected_track_info['voice_instrumental']['voice']:.2f}"
                    )
                else:
                    st.caption(data["label"])
                    st.subheader(f"{data['mean']:.2f}")

        # Common controls for both similarity methods
        similarity_threshold = st.slider(
            "Minimum Similarity (%)", min_value=0, max_value=100, value=50
        )
        max_results = st.slider(
            "Number of similar tracks to show",
            min_value=10,  # Changed from 1 to 10
            max_value=20,
            value=10,
        )

        if st.button("Find Similar Tracks"):
            # Get results for both embedding types (always get at least 10)
            effnet_tracks = st.session_state.library.find_similar_tracks(
                selected_track, "discogs-effnet", n_results=max(10, max_results)
            )
            musicnn_tracks = st.session_state.library.find_similar_tracks(
                selected_track, "msd-musicnn", max(10, max_results)
            )

            threshold = similarity_threshold / 100.0

            # Store all tracks but mark them as below threshold
            effnet_matched = [
                {**t, "below_threshold": t["similarity"] < threshold}
                for t in effnet_tracks[:max_results]
            ]
            musicnn_matched = [
                {**t, "below_threshold": t["similarity"] < threshold}
                for t in musicnn_tracks[:max_results]
            ]

            # Store in session state
            st.session_state.effnet_tracks = effnet_matched
            st.session_state.musicnn_tracks = musicnn_matched

        # Display results analysis if we have matches
        if hasattr(st.session_state, "effnet_tracks"):
            st.markdown("## Results Analysis")

            # Analyze overlap
            overlap_analysis = analyze_similarity_overlap(
                st.session_state.effnet_tracks, st.session_state.musicnn_tracks
            )

            # Display overlap statistics
            stats_cols = st.columns(4)
            stats_cols[0].metric(
                "Total Effnet Results", len(st.session_state.effnet_tracks)
            )
            stats_cols[1].metric(
                "Total Musicnn Results", len(st.session_state.musicnn_tracks)
            )
            stats_cols[2].metric(
                "Overlapping Results", overlap_analysis["overlap_count"]
            )
            stats_cols[3].metric(
                "Overlap Percentage",
                f"{(overlap_analysis['overlap_count'] * 100 / max(len(st.session_state.effnet_tracks), 1)):.1f}%",
            )

            # Create tabs for different views
            result_tabs = st.tabs(
                ["Side-by-Side Comparison", "Overlapping Tracks", "All Results"]
            )

            with result_tabs[0]:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Discogs-Effnet Results")
                    if st.session_state.effnet_tracks:
                        paginated_effnet = paginate_tracks(
                            st.session_state.effnet_tracks, "effnet"
                        )
                        for track in paginated_effnet:
                            similarity_text = f"{track['similarity']:.0%}"
                            if track.get("below_threshold", False):
                                st.write(
                                    f"**{track['track_id']}** ({similarity_text}) :warning: *Below threshold*"
                                )
                            else:
                                st.write(f"**{track['track_id']}** ({similarity_text})")
                            safe_audio_player(track["audio_path"])
                    else:
                        st.write("No matches found.")

                with col2:
                    st.subheader("MSD-Musicnn Results")
                    if st.session_state.musicnn_tracks:
                        paginated_musicnn = paginate_tracks(
                            st.session_state.musicnn_tracks, "musicnn"
                        )
                        for track in paginated_musicnn:
                            similarity_text = f"{track['similarity']:.0%}"
                            if track.get("below_threshold", False):
                                st.write(
                                    f"**{track['track_id']}** ({similarity_text}) :warning: *Below threshold*"
                                )
                            else:
                                st.write(f"**{track['track_id']}** ({similarity_text})")
                            safe_audio_player(track["audio_path"])
                    else:
                        st.write("No matches found.")

            with result_tabs[1]:
                if overlap_analysis["overlap_tracks"]:
                    st.subheader(
                        f"Tracks Found by Both Methods ({overlap_analysis['overlap_count']})"
                    )
                    paginated_overlap = paginate_tracks(
                        overlap_analysis["overlap_tracks"], "overlap"
                    )
                    for track in paginated_overlap:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            safe_audio_player(track["audio_path"])
                        with col2:
                            st.write(f"**{track['track_id']}**")
                            st.write(
                                f"Effnet similarity: {track['effnet_similarity']:.0%}"
                            )
                            st.write(
                                f"Musicnn similarity: {track['musicnn_similarity']:.0%}"
                            )
                            st.write(
                                f"Average similarity: {track['avg_similarity']:.0%}"
                            )
                            if track.get("below_threshold", False):
                                st.write(":warning: *Below threshold*")
                else:
                    st.write("No overlapping tracks found.")

            with result_tabs[2]:
                st.subheader("All Results")
                # Combine and deduplicate results
                all_tracks = []
                seen_ids = set()

                for track in (
                    st.session_state.effnet_tracks + st.session_state.musicnn_tracks
                ):
                    if track["track_id"] not in seen_ids:
                        seen_ids.add(track["track_id"])
                        all_tracks.append(track)

                paginated_all = paginate_tracks(all_tracks, "all")
                for track in paginated_all:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        safe_audio_player(track["audio_path"])
                    with col2:
                        st.write(f"**{track['track_id']}**")
                        found_in = []
                        if any(
                            t["track_id"] == track["track_id"]
                            for t in st.session_state.effnet_tracks
                        ):
                            found_in.append("Effnet")
                        if any(
                            t["track_id"] == track["track_id"]
                            for t in st.session_state.musicnn_tracks
                        ):
                            found_in.append("Musicnn")
                        st.write(f"Found by: {', '.join(found_in)}")

    with tabs[2]:
        # Genre & Style content
        render_genre_style_section(st.session_state.library)


if __name__ == "__main__":
    main()
