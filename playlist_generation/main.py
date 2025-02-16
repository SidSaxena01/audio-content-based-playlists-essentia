# main.py
import streamlit as st
from music_library import MusicLibrary
from ui_components import render_descriptor_tab, render_similarity_tab, render_genre_style_tab
from utils import get_valid_directories, validate_directory

# main.py
import streamlit as st
from music_library import MusicLibrary
from ui_components import render_descriptor_tab, render_similarity_tab, render_genre_style_tab
from utils import get_valid_directories, validate_directory

def configure_library() -> tuple[str, str, bool]:
    """
    Render the library configuration section in the sidebar.
    
    Returns:
        tuple[str, str, bool]: (analysis_dir, audio_dir, is_valid)
    """
    st.sidebar.header("Library Configuration")
    project_dirs = get_valid_directories()
    dir_options = [(display, path) for display, path in project_dirs]

    # Analysis Directory Configuration
    st.sidebar.subheader("Analysis Directory")
    use_proj_analysis = st.sidebar.checkbox("Use Project Directory", value=True, key="use_proj_analysis")
    if use_proj_analysis:
        default_index = next(
            (i for i, (_, path) in enumerate(dir_options) if path.endswith("results")),
            0
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
    use_proj_audio = st.sidebar.checkbox("Use Project Directory", value=True, key="use_proj_audio")
    if use_proj_audio:
        default_index = next(
            (i for i, (_, path) in enumerate(dir_options) if "data/MusAV" in path),
            0
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
    valid_analysis, analysis_err = validate_directory(analysis_dir)
    valid_audio, audio_err = validate_directory(audio_dir)

    if not valid_analysis:
        st.sidebar.error(f"Analysis Directory Error: {analysis_err}")
    else:
        st.sidebar.success("Analysis Directory Valid")
    if not valid_audio:
        st.sidebar.error(f"Audio Directory Error: {audio_err}")
    else:
        st.sidebar.success("Audio Directory Valid")

    return analysis_dir, audio_dir, (valid_analysis and valid_audio)

def main() -> None:
    st.set_page_config(page_title="Music Playlist Generator", layout="wide")
    st.title("Music Playlist Generator")

    # Render library configuration in the sidebar
    analysis_dir, audio_dir, is_valid = configure_library()

    # Prompt the user if configuration is not complete
    if not is_valid:
        st.info("Please configure your library directories in the sidebar and ensure they are valid.")
        return

    # Provide a sidebar button to load/reload the library
    if st.sidebar.button("Load/Reload Library"):
        with st.spinner("Loading library..."):
            try:
                st.session_state.library = MusicLibrary(analysis_dir, audio_dir)
                st.sidebar.success("Library loaded successfully! Feel free to close the sidebar now.")
            except Exception as e:
                st.sidebar.error(f"Error loading library: {str(e)}")

    if "library" not in st.session_state or st.session_state.library is None:
        st.info("Library not loaded. Use the sidebar configuration to load your library.")
        return

    # Setup tabs for different functionalities
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] { justify-content: center; }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 24px; }
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
