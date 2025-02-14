import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
import streamlit as st


class MusicLibrary:
    def __init__(self, analysis_dir: str, audio_dir: str):
        """Initialize the music library with analyzed tracks"""
        self.analysis_dir = analysis_dir
        self.audio_dir = audio_dir
        self.tracks_df = None
        self.load_tracks()

    def load_tracks(self):
        """Load all analyzed tracks from JSON files"""
        data = []
        for json_file in Path(self.analysis_dir).glob("**/*.json"):
            try:
                with open(json_file) as f:
                    track_data = json.load(f)
                    # Store both analysis and audio paths
                    track_data["analysis_path"] = str(json_file)
                    # Convert analysis path to audio path by replacing directory and extension
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

    def filter_tracks(
        self,
        tempo_range: Tuple[float, float] = None,
        has_vocals: bool = None,
        danceability_range: Tuple[float, float] = None,
        arousal_range: Tuple[float, float] = None,
        valence_range: Tuple[float, float] = None,
        key_scale: Tuple[str, str] = None,
        profile: str = "temperley",
    ) -> pd.DataFrame:
        """Filter tracks based on musical criteria"""
        filtered_df = self.tracks_df.copy()

        if tempo_range:
            filtered_df = filtered_df[
                (filtered_df["tempo"] >= tempo_range[0])
                & (filtered_df["tempo"] <= tempo_range[1])
            ]

        if has_vocals is not None:
            threshold = 0.5  # Threshold for vocal detection
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

        if key_scale:
            key, scale = key_scale
            filtered_df = filtered_df[
                filtered_df["key"].apply(
                    lambda x: x[profile]["key"] == key
                    and x[profile]["scale"].lower() == scale.lower()
                )
            ]

        return filtered_df

    def get_track_distribution(self, feature: str) -> Dict:
        """Get distribution statistics for a musical feature"""
        if feature not in self.tracks_df.columns:
            return None
        return {
            "mean": self.tracks_df[feature].mean(),
            "std": self.tracks_df[feature].std(),
            "min": self.tracks_df[feature].min(),
            "max": self.tracks_df[feature].max(),
            "quartiles": self.tracks_df[feature].quantile([0.25, 0.5, 0.75]).to_dict(),
        }

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

        # Calculate similarities
        similarities = []
        for _, track in self.tracks_df.iterrows():
            if track["track_id"] != query_track_id:
                track_embedding = np.array(track["embeddings"][embedding_type])
                similarity = 1 - distance.cosine(query_embedding, track_embedding)
                similarities.append(
                    {
                        "track_id": track["track_id"],
                        "similarity": similarity,
                        "audio_path": track[
                            "audio_path"
                        ],  # Changed from file_path to audio_path
                    }
                )

        # Sort by similarity and return top N
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)[
            :n_results
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

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for track in valid_tracks:
            f.write(f'#EXTINF:-1,{track["track_id"]}\n')
            f.write(f'{track["audio_path"]}\n')
    return True


def main():
    st.set_page_config(page_title="Music Playlist Generator", layout="wide")
    st.title("Music Playlist Generator")

    # Initialize session state
    if "library" not in st.session_state:
        analysis_dir = st.text_input("Enter path to analysis directory:", "results")
        audio_dir = st.text_input("Enter path to audio files directory:", "data/MusAV")
        if st.button("Load Library"):
            st.session_state.library = MusicLibrary(analysis_dir, audio_dir)
            st.success("Library loaded. Please press R to reload the app to continue.")
        return

    # Sidebar navigation
    page = st.sidebar.radio("Select Mode", ["Descriptor-based", "Similarity-based"])

    if page == "Descriptor-based":
        st.header("Generate Playlist by Musical Characteristics")

        col1, col2 = st.columns(2)

        with col1:
            # Tempo range
            tempo_range = st.slider(
                "Tempo Range (BPM)", min_value=0, max_value=200, value=(60, 180)
            )

            # Voice/Instrumental
            has_vocals = st.radio(
                "Vocal Presence", options=["All", "Vocal Only", "Instrumental Only"]
            )

            # Danceability
            dance_range = st.slider(
                "Danceability Range", min_value=0.0, max_value=1.0, value=(0.0, 1.0)
            )

        with col2:
            # Arousal and Valence
            arousal_range = st.slider(
                "Arousal Range", min_value=0.0, max_value=1.0, value=(0.0, 1.0)
            )

            valence_range = st.slider(
                "Valence Range", min_value=0.0, max_value=1.0, value=(0.0, 1.0)
            )

            # Key and Scale
            key = st.selectbox(
                "Key",
                options=[
                    "C",
                    "C#",
                    "D",
                    "Eb",
                    "E",
                    "F",
                    "F#",
                    "G",
                    "Ab",
                    "A",
                    "Bb",
                    "B",
                ],
            )
            scale = st.selectbox("Scale", options=["Major", "Minor"])

        # Process vocal selection
        vocals_filter = None
        if has_vocals == "Vocal Only":
            vocals_filter = True
        elif has_vocals == "Instrumental Only":
            vocals_filter = False

        # Filter tracks
        filtered_tracks = st.session_state.library.filter_tracks(
            tempo_range=tempo_range,
            has_vocals=vocals_filter,
            danceability_range=dance_range,
            arousal_range=arousal_range,
            valence_range=valence_range,
            key_scale=(key, scale) if key and scale else None,
        )

        # Display results
        st.subheader(f"Found {len(filtered_tracks)} matching tracks")
        if not filtered_tracks.empty:
            st.write("Top 10 tracks:")
            for _, track in filtered_tracks.head(10).iterrows():
                safe_audio_player(track["audio_path"])

            # Export playlist
            if st.button("Export as M3U8"):
                playlist_path = "playlist.m3u8"
                if create_m3u8_playlist(
                    filtered_tracks.to_dict("records"), playlist_path
                ):
                    st.success(f"Playlist exported to {playlist_path}")

    else:  # Similarity-based
        st.header("Find Similar Tracks")

        # Track selection
        track_id = st.selectbox(
            "Select query track",
            options=st.session_state.library.tracks_df["track_id"].tolist(),
        )

        # Embedding selection
        embedding_type = st.radio(
            "Select embedding type", options=["discogs-effnet", "msd-musicnn"]
        )

        if st.button("Find Similar Tracks"):
            # Get query track details
            query_track = st.session_state.library.tracks_df[
                st.session_state.library.tracks_df["track_id"] == track_id
            ].iloc[0]

            st.subheader("Query Track")
            safe_audio_player(query_track["audio_path"])

            # Find and display similar tracks
            similar_tracks = st.session_state.library.find_similar_tracks(
                track_id, embedding_type
            )

            st.subheader(f"Similar Tracks ({embedding_type})")
            for track in similar_tracks:
                st.write(f"Similarity: {track['similarity']:.3f}")
                # Get audio path from tracks_df
                track_data = st.session_state.library.tracks_df[
                    st.session_state.library.tracks_df["track_id"] == track["track_id"]
                ].iloc[0]
                safe_audio_player(track_data["audio_path"])

            # Export playlist
            if st.button("Export Similar Tracks as M3U8"):
                playlist_path = f"similar_tracks_{embedding_type}.m3u8"
                if create_m3u8_playlist(similar_tracks, playlist_path):
                    st.success(f"Playlist exported to {playlist_path}")


if __name__ == "__main__":
    main()
