# utils.py
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def get_valid_directories() -> List[Tuple[str, str]]:
    """
    Retrieve directories in the project root and their first-level subdirectories.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing a display name (with indentation) and the full path.
    """
    project_root = Path(__file__).parent
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
    Validate that a directory exists and is accessible.

    Args:
        path (str): The directory path to validate.

    Returns:
        Tuple[bool, str]: (True, "") if valid; otherwise (False, error_message).
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return False, "Directory does not exist"
        if not path_obj.is_dir():
            return False, "Path is not a directory"
        next(path_obj.iterdir(), None)
        return True, ""
    except PermissionError:
        return False, "Permission denied"
    except Exception as e:
        return False, str(e)


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


def create_m3u8_playlist(tracks: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Create an M3U8 playlist from a list of track dictionaries.

    Args:
        tracks (List[Dict[str, Any]]): List of track dictionaries.
        output_path (str): Where to save the playlist file. If relative, placed under "./playlists".

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
            for track in valid_tracks:
                f.write(f'#EXTINF:-1,{track["track_id"]}\n')
                f.write(f'{track["audio_path"]}\n')
        return True
    except Exception as e:
        st.error(f"Error writing playlist: {str(e)}")
        return False


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

    st.write(f"Page {current_page + 1} of {total_pages}")
    cols = st.columns(4)
    if cols[0].button("◀◀", key=f"first_{page_key}", disabled=current_page == 0):
        st.session_state[current_page_key] = 0
    if cols[1].button("◀", key=f"prev_{page_key}", disabled=current_page == 0):
        st.session_state[current_page_key] = max(0, current_page - 1)
    if cols[2].button("▶", key=f"next_{page_key}", disabled=current_page >= total_pages - 1):
        st.session_state[current_page_key] = min(total_pages - 1, current_page + 1)
    if cols[3].button("▶▶", key=f"last_{page_key}", disabled=current_page >= total_pages - 1):
        st.session_state[current_page_key] = total_pages - 1

    start_idx = st.session_state[current_page_key] * tracks_per_page
    end_idx = start_idx + tracks_per_page
    return tracks_data[start_idx:end_idx]