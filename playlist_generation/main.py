# main.py
import streamlit as st
from music_library import MusicLibrary
from ui_components import (
    render_descriptor_tab,
    render_genre_style_tab,
    render_similarity_tab,
)
from utils import get_valid_directories, validate_directory


def configure_library() -> tuple[str, str, bool]:
    """
    Render the library configuration section in the sidebar.

    Returns:
        tuple[str, str, bool]: (analysis_dir, audio_dir, is_valid)
    """
    st.sidebar.header("Library Configuration")
    st.sidebar.markdown("---")
    project_dirs = get_valid_directories()
    dir_options = [(display, path) for display, path in project_dirs]

    # Analysis Directory Configuration
    st.sidebar.subheader("Analysis Directory")
    st.sidebar.info("""
        Select the directory containing JSON analysis files.
        This should be the output directory from running the audio analysis script.
        Files should have the .json extension and contain audio analysis results.
    """)
    use_proj_analysis = st.sidebar.checkbox(
        "Use Project Directory", value=True, key="use_proj_analysis"
    )
    if use_proj_analysis:
        default_index = next(
            (i for i, (_, path) in enumerate(dir_options) if path.endswith("results")),
            0,
        )
        selected_analysis = st.sidebar.selectbox(
            "Select analysis directory",
            options=dir_options,
            format_func=lambda x: x[0],
            index=default_index,
            key="analysis_select",
        )
        analysis_dir = selected_analysis[1]
    else:
        analysis_dir = st.sidebar.text_input(
            "Enter path to analysis directory:",
            value="results",
            key="analysis_input",
        )

    # Audio Directory Configuration
    st.sidebar.subheader("Audio Files Directory")
    st.sidebar.info("""
        Select the directory containing your audio files.
        This should be your music library folder containing MP3 files
        that were analyzed by the audio analysis script.
    """)
    use_proj_audio = st.sidebar.checkbox(
        "Use Project Directory", value=True, key="use_proj_audio"
    )
    if use_proj_audio:
        default_index = next(
            (i for i, (_, path) in enumerate(dir_options) if "data/MusAV" in path), 0
        )
        selected_audio = st.sidebar.selectbox(
            "Select audio directory",
            options=dir_options,
            format_func=lambda x: x[0],
            index=default_index,
            key="audio_select",
        )
        audio_dir = selected_audio[1]
    else:
        audio_dir = st.sidebar.text_input(
            "Enter path to audio files directory:",
            value="data/MusAV",
            key="audio_input",
        )

    # Validate Directories and provide feedback in the sidebar
    valid_analysis, analysis_msg = validate_directory(analysis_dir)
    valid_audio, audio_msg = validate_directory(audio_dir)

    st.sidebar.subheader("Directory Validation", divider="gray")

    # Analysis directory validation feedback
    st.sidebar.markdown("##### Analysis Directory Status:")
    if valid_analysis:
        st.sidebar.success(analysis_msg)
    else:
        st.sidebar.error(analysis_msg)
        if "No JSON files found" in analysis_msg:
            st.sidebar.warning("""
                ⚠️ The analysis directory should contain JSON files with audio analysis results.
                If you haven't analyzed your music library yet:
                1. Run the audio analysis script first
                2. Point to the directory containing the analysis results
            """)

    # Audio directory validation feedback
    st.sidebar.markdown("##### Audio Directory Status:")
    if valid_audio:
        st.sidebar.success(audio_msg)
    else:
        st.sidebar.error(audio_msg)
        if "No MP3 files found" in audio_msg:
            st.sidebar.warning("""
                ⚠️ The audio directory should contain your music files in MP3 format.
                Make sure you're pointing to the correct music library folder.
            """)

    return analysis_dir, audio_dir, (valid_analysis and valid_audio)


def main() -> None:
    st.set_page_config(page_title="Music Playlist Generator", layout="wide")
    st.title("Music Playlist Generator")

    # Render library configuration in the sidebar
    analysis_dir, audio_dir, is_valid = configure_library()

    # Prompt the user if configuration is not complete
    if not is_valid:
        st.info(
            "Please configure your library directories in the sidebar and ensure they are valid."
        )
        return

    # Provide a sidebar button to load/reload the library
    if st.sidebar.button("Load/Reload Library"):
        with st.spinner("Loading library..."):
            try:
                st.session_state.library = MusicLibrary(analysis_dir, audio_dir)
                st.sidebar.success(
                    "Library loaded successfully! Feel free to close the sidebar now."
                )
            except Exception as e:
                st.sidebar.error(f"Error loading library: {str(e)}")

    if "library" not in st.session_state or st.session_state.library is None:
        st.info(
            "Library not loaded. Use the sidebar configuration to load your library."
        )
        return

    # Setup tabs for different functionalities
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] { justify-content: center; }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 24px; }
        hr { margin: 1em 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Descriptor-based", "Similarity-based", "Genre & Style"])

    with tabs[0]:
        render_descriptor_tab(st.session_state.library)
    with tabs[1]:
        render_similarity_tab(st.session_state.library)
    with tabs[2]:
        render_genre_style_tab(st.session_state.library)


if __name__ == "__main__":
    main()
