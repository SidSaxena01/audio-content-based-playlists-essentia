# utils.py
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def get_valid_directories() -> List[Tuple[str, str]]:
    """
    Retrieve directories in the project root and their first-level subdirectories.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing a display name (with indentation) and the full path.
    """
    project_root = Path(__file__).parent.parent
    dirs = []
    root_dirs = sorted(
        [d for d in project_root.iterdir() if d.is_dir()],
        key=lambda x: x.name.lower(),
    )
    for root_dir in root_dirs:
        dirs.append((root_dir.name, str(root_dir)))
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
    """
    Validate if a directory exists, is accessible, and contains required file types.

    Args:
        path (str): The directory path to validate.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        path = Path(path)
        if not path.exists():
            return False, "Directory does not exist"
        if not path.is_dir():
            return False, "Path is not a directory"

        # Try to list directory contents to check permissions
        try:
            files = list(path.rglob("*"))
            if not files:
                return False, "Directory is empty"
        except Exception as e:
            return False, f"Cannot read directory contents: {str(e)}"

        # Check for specific file types
        json_files = [f for f in files if f.suffix.lower() == ".json"]
        mp3_files = [f for f in files if f.suffix.lower() == ".mp3"]

        # Determine directory type and validate accordingly
        is_analysis_dir = (
            "analysis" in str(path).lower() or "results" in str(path).lower()
        )
        is_audio_dir = "audio" in str(path).lower() or "MusAV" in str(path).lower()

        if is_analysis_dir:
            if not json_files:
                return False, (
                    "❌ No JSON files found in analysis directory.\n"
                    "This directory should contain analysis results in JSON format.\n"
                    "Have you run the audio analysis script on your music library?"
                )
            return True, f"✓ Found {len(json_files)} JSON files"

        elif is_audio_dir:
            if not mp3_files:
                return False, (
                    "❌ No MP3 files found in audio directory.\n"
                    "This directory should contain your music files in MP3 format."
                )
            return True, f"✓ Found {len(mp3_files)} MP3 files"

        else:
            # If directory type can't be determined, check for both types
            if not (json_files or mp3_files):
                return False, (
                    "❌ No JSON or MP3 files found.\n"
                    "Analysis directory should contain JSON files.\n"
                    "Audio directory should contain MP3 files."
                )
            return (
                True,
                f"✓ Found {len(json_files)} JSON and {len(mp3_files)} MP3 files",
            )

    except PermissionError:
        return False, "❌ Permission denied accessing directory"
    except Exception as e:
        return False, f"❌ Error validating directory: {str(e)}"


def file_exists(path: str) -> bool:
    """
    Check if a file exists.

    Args:
        path (str): The file path.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(path).is_file()


def parse_music_style(genre_str: str) -> Tuple[str, str]:
    """
    Parse a music style string into a (genre, style) tuple.

    Args:
        genre_str (str): A string representing genre and optionally style, separated by '---' or '—'.

    Returns:
        Tuple[str, str]: Parsed (genre, style). If no separator, style is "unknown-style".
    """
    if "---" in genre_str:
        parts = genre_str.split("---", 1)
    elif "—" in genre_str:
        parts = genre_str.split("—", 1)
    else:
        parts = [genre_str, "unknown-style"]
    parent = parts[0].strip()
    style = parts[1].strip() if len(parts) > 1 else "unknown-style"
    return parent, style


def safe_audio_player(audio_path: str) -> None:
    """
    Safely display an audio player with error handling.

    Args:
        audio_path (str): The path to the audio file.
    """
    try:
        if file_exists(audio_path):
            st.audio(audio_path, format="audio/mp3")
        else:
            st.error(f"Audio file not found: {audio_path}")
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")


def create_m3u8_playlist(
    tracks: List[Dict[str, Any]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Create an M3U8 playlist from a list of track dictionaries.

    Args:
        tracks (List[Dict[str, Any]]): List of track dictionaries.
        output_path (str): Where to save the playlist file. If relative, placed under "./playlists".
        metadata (Optional[Dict[str, Any]]): Additional metadata to include in playlist header.

    Returns:
        bool: True if successful, False otherwise.
    """
    valid_tracks = []
    for track in tracks:
        if file_exists(track["audio_path"]):
            valid_tracks.append(track)
        else:
            st.warning(f"Skipping unavailable track: {track['track_id']}")
    if not valid_tracks:
        st.error("No valid tracks found for playlist")
        return False

    PLAYLISTS_DIR = "./playlists"
    if not os.path.isabs(output_path):
        output_path = os.path.join(PLAYLISTS_DIR, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")

            # Write metadata as comments
            f.write(
                f"#PLAYLIST-TITLE: {os.path.splitext(os.path.basename(output_path))[0]}\n"
            )
            f.write(f"#PLAYLIST-CREATED: {pd.Timestamp.now().isoformat()}\n")
            f.write(f"#TRACK-COUNT: {len(valid_tracks)}\n")

            if metadata:
                # Write generation method
                if "method" in metadata:
                    f.write(f"#GENERATION-METHOD: {metadata['method']}\n")

                # Write reference track if exists
                if "reference_track" in metadata:
                    f.write(f"#REFERENCE-TRACK: {metadata['reference_track']}\n")

                # Write genre/style filters
                if "genres" in metadata and metadata["genres"]:
                    f.write(f"#GENRES: {', '.join(metadata['genres'])}\n")
                if "styles" in metadata and metadata["styles"]:
                    f.write(f"#STYLES: {', '.join(metadata['styles'])}\n")

                # Write descriptor filters
                if "descriptors" in metadata:
                    for desc, value in metadata["descriptors"].items():
                        if isinstance(value, tuple):
                            f.write(
                                f"#DESCRIPTOR-{desc.upper()}: {value[0]} - {value[1]}\n"
                            )
                        else:
                            f.write(f"#DESCRIPTOR-{desc.upper()}: {value}\n")

                # Write similarity thresholds
                if "similarity_threshold" in metadata:
                    f.write(
                        f"#SIMILARITY-THRESHOLD: {metadata['similarity_threshold']}\n"
                    )

                # Add a blank line after metadata
                f.write("\n")

            # Write track entries
            for track in valid_tracks:
                track_info = []
                if "similarity" in track:
                    track_info.append(f"similarity={track['similarity']:.2%}")
                if "matched_genres" in track:
                    track_info.append(f"genres={','.join(track['matched_genres'])}")
                if "matched_styles" in track:
                    track_info.append(f"styles={','.join(track['matched_styles'])}")

                info_str = " - ".join([track["track_id"]] + track_info)
                f.write(f"#EXTINF:-1,{info_str}\n")
                f.write(f'{track["audio_path"]}\n')

        return True
    except Exception as e:
        st.error(f"Error writing playlist: {str(e)}")
        return False


def normalize_arousal_valence(x: float) -> float:
    """
    Normalize an arousal/valence value that is ideally in the range [1, 9] to the range [-1, 1].
    If x falls outside [1, 9], it is first clipped to that range.

    Args:
        x (float): The raw predicted value.

    Returns:
        float: The normalized value in [-1, 1].
    """
    clipped = max(min(x, 9), 1)
    return (clipped - 1) / 4 - 1


def paginate_tracks(
    tracks_data: List[Dict[str, Any]], page_key: str, tracks_per_page: int = 10
) -> List[Dict[str, Any]]:
    """
    Return a subset of tracks for the current pagination page.

    Args:
        tracks_data (List[Dict[str, Any]]): List of track dictionaries.
        page_key (str): Unique key for pagination state.
        tracks_per_page (int): Number of tracks per page.

    Returns:
        List[Dict[str, Any]]: List of tracks for the current page.
    """
    current_page_key = f"current_page_{page_key}"
    if current_page_key not in st.session_state:
        st.session_state[current_page_key] = 0

    total_pages = (len(tracks_data) - 1) // tracks_per_page + 1
    current_page = st.session_state[current_page_key]

    st.info(
        "ℹ️ Note: Page navigation buttons might require double-clicking to work properly."
    )
    st.write(f"Page {current_page + 1} of {total_pages}")
    cols = st.columns(4)
    if cols[0].button("◀◀", key=f"first_{page_key}", disabled=current_page == 0):
        st.session_state[current_page_key] = 0
    if cols[1].button("◀", key=f"prev_{page_key}", disabled=current_page == 0):
        st.session_state[current_page_key] = max(0, current_page - 1)
    if cols[2].button(
        "▶", key=f"next_{page_key}", disabled=current_page >= total_pages - 1
    ):
        st.session_state[current_page_key] = min(total_pages - 1, current_page + 1)
    if cols[3].button(
        "▶▶", key=f"last_{page_key}", disabled=current_page >= total_pages - 1
    ):
        st.session_state[current_page_key] = total_pages - 1

    start_idx = st.session_state[current_page_key] * tracks_per_page
    end_idx = start_idx + tracks_per_page
    return tracks_data[start_idx:end_idx]
