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
        key: str = None,
        scale: str = None,
        genres: List[str] = None,
        styles: List[str] = None,
        profile: str = "temperley",
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
            filtered_df = filtered_df[
                (filtered_df["arousal"] >= arousal_range[0])
                & (filtered_df["arousal"] <= arousal_range[1])
            ]

        if valence_range:
            filtered_df = filtered_df[
                (filtered_df["valence"] >= valence_range[0])
                & (filtered_df["valence"] <= valence_range[1])
            ]

        # Updated key and scale filtering
        if key is not None or scale is not None:
            filtered_df = filtered_df[
                filtered_df["key"].apply(
                    lambda x: (x[profile]["key"] == key if key is not None else True)
                    and (
                        x[profile]["scale"].lower() == scale.lower()
                        if scale is not None
                        else True
                    )
                )
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

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pagination_cols = st.columns(5)

        if pagination_cols[0].button(
            "◀◀", key=f"first_{page_key}", disabled=current_page == 0
        ):
            st.session_state[f"current_page_{page_key}"] = 0

        if pagination_cols[1].button(
            "◀", key=f"prev_{page_key}", disabled=current_page == 0
        ):
            st.session_state[f"current_page_{page_key}"] = max(0, current_page - 1)

        pagination_cols[2].write(f"Page {current_page + 1} of {total_pages}")

        if pagination_cols[3].button(
            "▶", key=f"next_{page_key}", disabled=current_page >= total_pages - 1
        ):
            st.session_state[f"current_page_{page_key}"] = min(
                total_pages - 1, current_page + 1
            )

        if pagination_cols[4].button(
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


def main():
    st.set_page_config(page_title="Music Playlist Generator", layout="wide")
    st.title("Music Playlist Generator")

    # Ensure the library is always freshly loaded
    analysis_dir = st.text_input("Enter path to analysis directory:", "results")
    audio_dir = st.text_input("Enter path to audio files directory:", "data/MusAV")

    if st.button("Load/Reload Library"):
        st.session_state.library = MusicLibrary(analysis_dir, audio_dir)
        st.success(
            "Library loaded/reloaded. Please reload the app by pressing R if needed."
        )

    if "library" not in st.session_state or st.session_state.library is None:
        st.info("Library not loaded yet.")
        return

    # Replace sidebar navigation with tabs
    tabs = st.tabs(["Descriptor-based", "Similarity-based", "Genre & Style"])

    with tabs[0]:
        # Descriptor-based content
        st.header("Generate Playlist by Musical Characteristics")

        # Initialize session state for pagination
        if "filtered_tracks" not in st.session_state:
            st.session_state.filtered_tracks = None
        if "current_page" not in st.session_state:
            st.session_state.current_page = 0

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            tempo_range = st.slider(
                "Tempo Range (BPM)", min_value=0, max_value=200, value=(60, 180)
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

            danceability_range = st.slider(
                "Danceability", min_value=0.0, max_value=1.0, value=(0.0, 1.0)
            )

        with col2:
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
            scale_options = ["major", "minor"]

            selected_key = st.selectbox("Key", ["Any"] + key_options)
            selected_scale = st.selectbox("Scale", ["Any"] + scale_options)

            key_filter = selected_key if selected_key != "Any" else None
            scale_filter = selected_scale if selected_scale != "Any" else None

            arousal_range = st.slider(
                "Arousal (Energy)", min_value=-1.0, max_value=1.0, value=(-1.0, 1.0)
            )
            arousal_range = (
                (arousal_range[0] + 1) * 4 + 1,
                (arousal_range[1] + 1) * 4 + 1,
            )

            valence_range = st.slider(
                "Valence (Mood)", min_value=-1.0, max_value=1.0, value=(-1.0, 1.0)
            )
            valence_range = (
                (valence_range[0] + 1) * 4 + 1,
                (valence_range[1] + 1) * 4 + 1,
            )

        with col3:
            shuffle_results = st.checkbox("Shuffle Results", value=False)

        generate_col1, generate_col2 = st.columns([3, 1])
        with generate_col1:
            generate_button = st.button("Generate Playlist")
        with generate_col2:
            if st.session_state.filtered_tracks is not None:
                if st.button("Export as M3U8", key="export_descriptor"):
                    with st.spinner("Exporting descriptor-based playlist..."):
                        tracks_list = [
                            {
                                "track_id": row["track_id"],
                                "audio_path": row["audio_path"],
                            }
                            for _, row in st.session_state.filtered_tracks.iterrows()
                        ]
                        playlist_path = "descriptor_based_playlist.m3u8"
                        if create_m3u8_playlist(tracks_list, playlist_path):
                            st.success(f"Playlist exported to {playlist_path}")

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
            )

            if shuffle_results:
                filtered_tracks = filtered_tracks.sample(frac=1)

            # Store filtered tracks in session state
            st.session_state.filtered_tracks = filtered_tracks
            st.session_state.current_page = 0

        # Display results if we have filtered tracks
        if st.session_state.filtered_tracks is not None:
            with results_container:
                filtered_tracks = st.session_state.filtered_tracks
                tracks_per_page = 10
                total_pages = (len(filtered_tracks) - 1) // tracks_per_page + 1

                # Header with track count
                st.subheader(f"Found {len(filtered_tracks)} matching tracks")

                if not filtered_tracks.empty:
                    # Use pagination helper
                    page_tracks = paginate_tracks(
                        filtered_tracks.to_dict("records"), "descriptor"
                    )
                    for track in page_tracks:
                        safe_audio_player(track["audio_path"])
                else:
                    st.warning("No tracks found matching the criteria")

    with tabs[1]:
        # Similarity-based content
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
        # Always display the reference track player below the selectbox
        selected_track_info = st.session_state.library.tracks_df[
            st.session_state.library.tracks_df["track_id"] == selected_track
        ].iloc[0]
        safe_audio_player(selected_track_info["audio_path"])

        embedding_type = st.selectbox(
            "Select similarity method", ["discogs-effnet", "msd-musicnn"]
        )

        n_results = st.slider(
            "Number of similar tracks", min_value=1, max_value=50, value=10
        )

        if st.button("Find Similar Tracks"):
            similar_tracks = st.session_state.library.find_similar_tracks(
                selected_track, embedding_type, n_results
            )

            # Store in session state for pagination
            st.session_state.similar_tracks = similar_tracks

        if "similar_tracks" in st.session_state and st.session_state.similar_tracks:
            # Display table header
            header_cols = st.columns([2, 1, 3])
            header_cols[0].write("Name")
            header_cols[1].write("Similarity")
            header_cols[2].write("Music Style")

            # Use pagination helper
            paginated_similar = paginate_tracks(
                st.session_state.similar_tracks, "similarity"
            )

            for track in paginated_similar:
                track_info = st.session_state.library.tracks_df[
                    st.session_state.library.tracks_df["track_id"] == track["track_id"]
                ].iloc[0]
                music_styles = track_info.get("music_styles", {})
                top_styles = sorted(
                    music_styles.items(), key=lambda item: item[1], reverse=True
                )[:3]
                top_styles_split = []
                for key, prob in top_styles:
                    if "---" in key:
                        genre, style = key.split("---", 1)
                    else:
                        genre = key
                        style = "unknown-style"
                    top_styles_split.append(
                        f"{genre.strip()}: {style.strip()} ({prob:.2f})"
                    )
                music_style = ", ".join(top_styles_split) if top_styles_split else "N/A"
                row_cols = st.columns([2, 1, 3])
                row_cols[0].write(track["track_id"])
                row_cols[1].write(f"{track['similarity'] * 100:.0f}%")
                row_cols[2].write(music_style)
                safe_audio_player(track["audio_path"])

    with tabs[2]:
        # Genre & Style content
        render_genre_style_section(st.session_state.library)


if __name__ == "__main__":
    main()
