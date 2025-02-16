from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from music_library import MusicLibrary
from utils import (
    create_m3u8_playlist,
    normalize_arousal_valence,
    paginate_tracks,
    parse_music_style,
    safe_audio_player,
)


def plot_three_key_distributions(filtered_df: pd.DataFrame) -> px.bar:
    """
    Create a grouped bar chart comparing key distributions from three algorithms:
    Temperley, EDMA, and Krumhansl.

    Args:
        filtered_df (pd.DataFrame): DataFrame of filtered tracks. Each row should have a "key"
                                    column which is a dictionary with entries for
                                    "temperley", "edma", and "krumhansl".

    Returns:
        A Plotly Express figure with grouped bars.
    """
    algorithms = ["temperley", "edma", "krumhansl"]
    data = []

    for algo in algorithms:
        # Extract the key for each algorithm from the "key" column.
        # If a track does not have a value for the algorithm, it will be ignored.
        counts = (
            filtered_df["key"]
            .dropna()
            .apply(
                lambda x: x.get(algo, {}).get("key") if isinstance(x, dict) else None
            )
            .dropna()
            .value_counts()
        )
        for k, count in counts.items():
            data.append(
                {
                    "Algorithm": algo.capitalize(),  # Capitalize for nicer labels
                    "Key": k,
                    "Count": count,
                }
            )

    if not data:
        st.warning("No key distribution data available.")
        return px.bar(title="Key Distribution Comparison Across Algorithms")

    df_plot = pd.DataFrame(data)
    fig = px.bar(
        df_plot,
        x="Key",
        y="Count",
        color="Algorithm",
        barmode="group",
        title="Key Distribution Comparison Across Algorithms",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def reset_descriptor_filters() -> None:
    """
    Reset all descriptor filters to their default values by updating st.session_state.
    """
    st.session_state.tempo_range = (60, 180)
    st.session_state.danceability_range = (0.0, 1.0)
    st.session_state.arousal_range = (-1.0, 1.0)
    st.session_state.valence_range = (-1.0, 1.0)
    st.session_state.loudness_range = (-30.0, -10.0)
    st.session_state.has_vocals = None
    st.session_state.selected_keys = []
    st.session_state.selected_scale = "Any"


def plot_valence_arousal_range(
    valence_range: Tuple[float, float], arousal_range: Tuple[float, float]
) -> go.Figure:
    """
    Create an XY plot for valence (x-axis) and arousal (y-axis) with four quadrants.
    A dashed line is drawn at x=0 and y=0 to indicate quadrant boundaries.
    The currently selected range is highlighted as a semi-transparent green rectangle.

    Args:
        valence_range (Tuple[float, float]): The selected (min, max) valence values.
        arousal_range (Tuple[float, float]): The selected (min, max) arousal values.

    Returns:
        go.Figure: A Plotly figure with the plotted quadrants and highlighted selection.
    """
    fig = go.Figure()

    # Draw quadrant dividing lines at 0
    fig.add_shape(
        type="line", x0=-1, x1=1, y0=0, y1=0, line=dict(color="black", dash="dash")
    )
    fig.add_shape(
        type="line", x0=0, x1=0, y0=-1, y1=1, line=dict(color="black", dash="dash")
    )

    # Add rectangle highlighting the selected range
    fig.add_shape(
        type="rect",
        x0=valence_range[0],
        y0=arousal_range[0],
        x1=valence_range[1],
        y1=arousal_range[1],
        fillcolor="rgba(0,128,0,0.3)",
        line=dict(color="green"),
    )

    # Set layout and axes properties
    fig.update_layout(
        xaxis=dict(range=[-1, 1], title="Valence"),
        yaxis=dict(range=[-1, 1], title="Arousal"),
        margin=dict(l=20, r=20, t=20, b=20),
        width=300,
        height=300,
        title="Valence-Arousal Range",
    )
    return fig


def display_descriptor_statistics(library: MusicLibrary) -> None:
    """
    Display statistics and distribution plots for musical descriptors.

    Args:
        library (MusicLibrary): The music library instance.
    """
    library.tracks_df["voice_prob"] = library.tracks_df["voice_instrumental"].apply(
        lambda x: x["voice"]
    )
    descriptors = [
        ("tempo", None),
        ("danceability", None),
        ("arousal", normalize_arousal_valence),
        ("valence", normalize_arousal_valence),
        ("loudness", None),
        ("voice_prob", None),
    ]
    for i in range(0, len(descriptors), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(descriptors):
                descriptor, normalizer = descriptors[i + j]
                with cols[j]:
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
                        df = library.tracks_df.copy()
                        if normalizer:
                            df[descriptor] = df[descriptor].apply(normalizer)
                        fig = px.histogram(df, x=descriptor, nbins=20)
                        fig.update_layout(
                            height=150,
                            margin=dict(l=20, r=20, t=20, b=20),
                            bargap=0.05,
                            showlegend=False,
                            xaxis_title=None,
                            yaxis_title=None,
                        )
                        st.plotly_chart(fig, use_container_width=True)


def compute_track_statistics(tracks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics for musical descriptors across tracks.

    Args:
        tracks_df (pd.DataFrame): DataFrame of tracks.

    Returns:
        Dict[str, Any]: Dictionary of computed statistics.
    """
    voice_prob = tracks_df["voice_instrumental"].apply(lambda x: x["voice"])
    descriptors = {
        "tempo": {"label": "Tempo (BPM)", "normalize": None},
        "danceability": {"label": "Danceability", "normalize": None},
        "arousal": {"label": "Arousal", "normalize": normalize_arousal_valence},
        "valence": {"label": "Valence", "normalize": normalize_arousal_valence},
        "loudness": {"label": "Loudness (dB)", "normalize": None},
        "voice_prob": {
            "label": "Voice Probability",
            "normalize": None,
            "values": voice_prob,
        },
    }
    stats = {}
    for desc, config in descriptors.items():
        if "values" in config:
            values = config["values"]
        elif desc in tracks_df.columns:
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
    if "key" in tracks_df.columns:
        key_profile = "temperley"
        key_counts = (
            tracks_df["key"].apply(lambda x: x[key_profile]["key"]).value_counts()
        )
        stats["key"] = {"label": "Key Distribution", "values": key_counts.to_dict()}
    return stats


def display_summary_stats(tracks_df: pd.DataFrame) -> None:
    """
    Display summary statistics, distributions, and a combined key distribution plot
    for the matched tracks (filtered by your descriptor-based filters).
    """
    if tracks_df.empty:
        return

    stats = compute_track_statistics(tracks_df)
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

    # Combined Key Distribution Plot Across Algorithms for matched tracks only
    st.subheader("Key Distribution Comparison", divider="gray")
    # Use only the filtered (matched) tracks.
    matched_tracks = (
        tracks_df  # Here, tracks_df should be st.session_state.filtered_tracks
    )
    fig_keys = plot_three_key_distributions(matched_tracks)
    st.plotly_chart(fig_keys, use_container_width=True)


def display_detailed_tracks(tracks_df: pd.DataFrame, tracks_per_page: int = 5) -> None:
    """
    Display detailed track statistics and audio players with pagination.

    Args:
        tracks_df (pd.DataFrame): DataFrame of tracks.
        tracks_per_page (int): Number of tracks per page.
    """
    if tracks_df.empty:
        return

    if "current_filtered_tracks" not in st.session_state:
        st.session_state.current_filtered_tracks = tracks_df
    current_tracks = st.session_state.current_filtered_tracks

    st.subheader("Individual Track Values", divider="gray")
    track_data = []
    stats = compute_track_statistics(current_tracks)
    for _, track in current_tracks.iterrows():
        row = {"Track ID": track["track_id"]}
        for desc, data in stats.items():
            if desc == "key":
                key_info = track[desc]["temperley"]
                row[data["label"]] = f"{key_info['key']} {key_info['scale']}"
            elif desc == "voice_prob":
                voice_prob = track["voice_instrumental"]["voice"]
                row[data["label"]] = f"{voice_prob:.2f}"
            else:
                value = track[desc]
                if desc in ["arousal", "valence"]:
                    value = (value - 1) / 4 - 1
                row[data["label"]] = f"{value:.2f}"
        track_data.append(row)

    if track_data:
        st.dataframe(
            pd.DataFrame(track_data).set_index("Track ID"),
            hide_index=False,
            use_container_width=True,
        )

    st.subheader("Individual Tracks", divider="gray")
    tracks_records = current_tracks.to_dict("records")
    paginated_tracks = paginate_tracks(
        tracks_records, "descriptor_tracks", tracks_per_page
    )
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


def analyze_similarity_overlap(
    effnet_tracks: List[Dict[str, Any]], musicnn_tracks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze overlap between two sets of similar tracks.

    Args:
        effnet_tracks (List[Dict[str, Any]]): List from first similarity method.
        musicnn_tracks (List[Dict[str, Any]]): List from second similarity method.

    Returns:
        Dict[str, Any]: Dictionary with overlap analysis.
    """
    effnet_ids = {t["track_id"] for t in effnet_tracks}
    musicnn_ids = {t["track_id"] for t in musicnn_tracks}
    overlap_ids = effnet_ids & musicnn_ids
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
    overlap_tracks.sort(key=lambda x: x["avg_similarity"], reverse=True)
    return {
        "overlap_count": len(overlap_ids),
        "effnet_unique": len(effnet_ids - musicnn_ids),
        "musicnn_unique": len(musicnn_ids - effnet_ids),
        "overlap_tracks": overlap_tracks,
    }


def render_descriptor_tab(library: MusicLibrary) -> None:
    """
    Render the Descriptor-based tab which displays library statistics and allows playlist generation
    based on musical descriptors.
    """
    st.header("Library Statistics")
    with st.expander("✨ Click to view Library Statistics ✨", expanded=False):
        st.caption("Statistics are computed across all tracks in the library")
        display_descriptor_statistics(library)

    st.header("Generate Playlist by Musical Characteristics")
    if "filtered_tracks" not in st.session_state:
        st.session_state.filtered_tracks = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0

    # Ensure session state values exist so that the reset button can update them
    if "tempo_range" not in st.session_state:
        st.session_state.tempo_range = (60, 180)
    if "danceability_range" not in st.session_state:
        st.session_state.danceability_range = (0.0, 1.0)
    if "arousal_range" not in st.session_state:
        st.session_state.arousal_range = (-1.0, 1.0)
    if "valence_range" not in st.session_state:
        st.session_state.valence_range = (-1.0, 1.0)
    if "loudness_range" not in st.session_state:
        st.session_state.loudness_range = (-30.0, -10.0)
    if "has_vocals" not in st.session_state:
        st.session_state.has_vocals = None
    if "selected_keys" not in st.session_state:
        st.session_state.selected_keys = []
    if "selected_scale" not in st.session_state:
        st.session_state.selected_scale = "Any"

        # Row 1: Rhythm & Movement
    st.subheader("Rhythm & Movement")
    col_rm1, col_rm2 = st.columns(2)
    with col_rm1:
        tempo_range = st.slider(
            "Tempo Range (BPM)", 0, 200, st.session_state.tempo_range, key="tempo_range"
        )
    with col_rm2:
        danceability_range = st.slider(
            "Danceability",
            0.0,
            1.0,
            st.session_state.danceability_range,
            key="danceability_range",
        )

    # Row 2: Mood & Energy with sliders and quadrant plot side by side
    st.subheader("Mood & Energy")
    col_me_left, col_me_right = st.columns([1, 1])
    with col_me_left:
        arousal_range = st.slider(
            "Arousal (Energy)",
            -1.0,
            1.0,
            st.session_state.arousal_range,
            key="arousal_range",
        )
        valence_range = st.slider(
            "Valence (Mood)",
            -1.0,
            1.0,
            st.session_state.valence_range,
            key="valence_range",
        )
    with col_me_right:
        # Display the valence-arousal quadrant plot to the right of the sliders.
        from ui_components import (
            plot_valence_arousal_range,  # Ensure this function is imported
        )

        fig = plot_valence_arousal_range(valence_range, arousal_range)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Additional Filters - Musical Key, Sound & Voice, and Reset Filters
    st.subheader("Additional Filters")
    col_key, col_sv, col_reset = st.columns([1, 1, 0.5])
    with col_key:
        st.subheader("Musical Key")
        key_options = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        selected_keys = st.multiselect("Key", options=key_options, key="selected_keys")
        key_filter = selected_keys if selected_keys else None
        scale_options = ["major", "minor"]
        selected_scale = st.selectbox(
            "Scale", ["Any"] + scale_options, key="selected_scale"
        )
        scale_filter = selected_scale if selected_scale != "Any" else None

    with col_sv:
        st.subheader("Sound & Voice")
        has_vocals = st.radio(
            "Vocal Content",
            [None, True, False],
            format_func=lambda x: "Any"
            if x is None
            else "With Vocals"
            if x
            else "Instrumental",
            key="has_vocals",
        )
        loudness_range = st.slider(
            "Loudness Range (dB)",
            -60.0,
            0.0,
            st.session_state.loudness_range,
            key="loudness_range",
        )

    with col_reset:
        if st.button(
            "Reset Filters", key="reset_filters", on_click=reset_descriptor_filters
        ):
            st.rerun()

    button_cols = st.columns([1.5, 0.5, 2])
    with button_cols[0]:
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            generate_button = st.button("Generate Playlist", use_container_width=True)
        with col_btn2:
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
                                track_list, "descriptor_based_playlist.m3u8"
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
    if generate_button:
        filtered_tracks = library.filter_tracks(
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
        st.session_state.filtered_tracks = filtered_tracks
        if "current_page_descriptor_tracks" in st.session_state:
            del st.session_state.current_page_descriptor_tracks
        if "current_filtered_tracks" in st.session_state:
            del st.session_state.current_filtered_tracks
        st.subheader(f"Found {len(filtered_tracks)} matching tracks")
        from ui_components import (
            display_summary_stats,  # Local import to avoid circular dependency
        )

        with st.expander(
            "✨ Click here to View Summary Statistics, Distributions & Key Distribution ✨"
        ):
            display_summary_stats(filtered_tracks)
    if st.session_state.filtered_tracks is not None:
        from ui_components import display_detailed_tracks  # Local import

        display_detailed_tracks(st.session_state.filtered_tracks)


def render_similarity_tab(library: MusicLibrary) -> None:
    """
    Render the Similarity-based tab which allows the user to search for a reference track,
    display similar tracks using two algorithms, and view results in three sub-tabs.
    Each sub-tab includes an export button that exports only the tracks shown in that tab.
    """
    st.header("Generate Playlist by Track Similarity")
    search_query = st.text_input("Search for reference track", value="")

    def get_track_label(track: dict) -> str:
        label = track["track_id"][:8]
        if "music_styles" in track:
            top_styles = sorted(
                track["music_styles"].items(), key=lambda item: item[1], reverse=True
            )[:1]
            if top_styles:
                style_key, _ = top_styles[0]
                label += f" ({parse_music_style(style_key)[0]})"
        return label

    # Filter tracks based on search query.
    all_tracks = library.tracks_df.to_dict("records")
    filtered_tracks = [
        track
        for track in all_tracks
        if search_query.lower() in track["track_id"].lower()
        or search_query.lower() in get_track_label(track).lower()
    ]
    if not search_query or not filtered_tracks:
        filtered_tracks = all_tracks
    options = [(get_track_label(track), track["track_id"]) for track in filtered_tracks]
    options.sort(key=lambda x: x[0])
    selected_label, selected_track = st.selectbox(
        "Select a reference track", options, format_func=lambda opt: opt[0]
    )
    selected_track_info = library.tracks_df[
        library.tracks_df["track_id"] == selected_track
    ].iloc[0]

    st.subheader("Reference Track")
    safe_audio_player(selected_track_info["audio_path"])

    # Display some basic descriptor stats for the reference track.
    try:
        descriptor_stats = compute_track_statistics(
            library.tracks_df[library.tracks_df["track_id"] == selected_track]
        )
    except Exception:
        descriptor_stats = None
    cols = st.columns(4)
    if descriptor_stats:
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

    col_sim, col_results = st.columns([2, 1])
    with col_sim:
        similarity_threshold = st.slider(
            "Minimum Similarity (%)", min_value=0, max_value=100, value=50
        )
    with col_results:
        max_results = st.number_input(
            "Number of similar tracks to show", min_value=10, step=1, value=10
        )

    if st.button("Find Similar Tracks"):
        effnet_tracks = library.find_similar_tracks(
            selected_track, "discogs-effnet", n_results=max(10, max_results)
        )
        musicnn_tracks = library.find_similar_tracks(
            selected_track, "msd-musicnn", n_results=max(10, max_results)
        )
        threshold = similarity_threshold / 100.0
        effnet_matched = [
            {**t, "below_threshold": t["similarity"] < threshold}
            for t in effnet_tracks[:max_results]
        ]
        musicnn_matched = [
            {**t, "below_threshold": t["similarity"] < threshold}
            for t in musicnn_tracks[:max_results]
        ]
        st.session_state.effnet_tracks = effnet_matched
        st.session_state.musicnn_tracks = musicnn_matched

    if not st.session_state.get("effnet_tracks") or not st.session_state.get(
        "musicnn_tracks"
    ):
        st.info(
            "No similar tracks found yet. Please click 'Find Similar Tracks' to generate results."
        )
        return

    if hasattr(st.session_state, "effnet_tracks"):
        st.markdown("## Results Analysis")
        overlap_analysis = analyze_similarity_overlap(
            st.session_state.effnet_tracks, st.session_state.musicnn_tracks
        )
        result_tabs = st.tabs(
            ["Side-by-Side Comparison", "Overlapping Tracks", "All Results"]
        )

    # In the Similarity-based tab's Side-by-Side sub-tab:
    with result_tabs[0]:
        # Two export buttons in separate columns:
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("Export Discogs-Effnet Playlist", key="export_effnet"):
                if create_m3u8_playlist(
                    st.session_state.effnet_tracks, "similarity_effnet_playlist.m3u8"
                ):
                    st.success("Discogs-Effnet playlist exported successfully!")
                else:
                    st.error("Export failed for Discogs-Effnet playlist.")
        with col_export2:
            if st.button("Export MSD-Musicnn Playlist", key="export_musicnn"):
                if create_m3u8_playlist(
                    st.session_state.musicnn_tracks, "similarity_musicnn_playlist.m3u8"
                ):
                    st.success("MSD-Musicnn playlist exported successfully!")
                else:
                    st.error("Export failed for MSD-Musicnn playlist.")

        # Now display the two sets side-by-side
        col_left, col_right = st.columns(2)
        with col_left:
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
        with col_right:
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

        # Sub-tab 2: Overlapping Tracks
        with result_tabs[1]:
            if st.button("Export Overlapping Playlist", key="export_overlap"):
                if create_m3u8_playlist(
                    overlap_analysis["overlap_tracks"],
                    "similarity_overlap_playlist.m3u8",
                ):
                    st.success("Playlist exported successfully!")
                else:
                    st.error("Playlist export failed.")
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
                        st.write(f"Effnet similarity: {track['effnet_similarity']:.0%}")
                        st.write(
                            f"Musicnn similarity: {track['musicnn_similarity']:.0%}"
                        )
                        st.write(f"Average similarity: {track['avg_similarity']:.0%}")
                        if track.get("below_threshold", False):
                            st.write(":warning: *Below threshold*")
            else:
                st.write("No overlapping tracks found.")

        # Sub-tab 3: All Results
        with result_tabs[2]:
            if st.button("Export All Results Playlist", key="export_all"):
                all_tracks_combined = []
                seen_ids = set()
                for track in (
                    st.session_state.effnet_tracks + st.session_state.musicnn_tracks
                ):
                    if track["track_id"] not in seen_ids:
                        seen_ids.add(track["track_id"])
                        all_tracks_combined.append(track)
                if create_m3u8_playlist(
                    all_tracks_combined, "similarity_all_results_playlist.m3u8"
                ):
                    st.success("Playlist exported successfully!")
                else:
                    st.error("Playlist export failed.")
            st.subheader("All Results")
            all_tracks_combined = []
            seen_ids = set()
            for track in (
                st.session_state.effnet_tracks + st.session_state.musicnn_tracks
            ):
                if track["track_id"] not in seen_ids:
                    seen_ids.add(track["track_id"])
                    all_tracks_combined.append(track)
            paginated_all = paginate_tracks(all_tracks_combined, "all")
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


# def render_similarity_tab(library: MusicLibrary) -> None:
#     """
#     Render the Similarity-based tab which allows the user to find tracks similar to a reference track.
#     """
#     st.header("Generate Playlist by Track Similarity")
#     search_query = st.text_input("Search for reference track", value="")

#     def get_track_label(track: dict) -> str:
#         label = track["track_id"][:8]
#         if "music_styles" in track:
#             top_styles = sorted(
#                 track["music_styles"].items(), key=lambda item: item[1], reverse=True
#             )[:1]
#             if top_styles:
#                 style_key, _ = top_styles[0]
#                 label += f" ({parse_music_style(style_key)[0]})"
#         return label

#     all_tracks = library.tracks_df.to_dict("records")
#     filtered_tracks = [
#         track
#         for track in all_tracks
#         if search_query.lower() in track["track_id"].lower()
#         or search_query.lower() in get_track_label(track).lower()
#     ]
#     if not search_query or not filtered_tracks:
#         filtered_tracks = all_tracks
#     options = [(get_track_label(track), track["track_id"]) for track in filtered_tracks]
#     options.sort(key=lambda x: x[0])
#     selected_label, selected_track = st.selectbox(
#         "Select a reference track", options, format_func=lambda opt: opt[0]
#     )
#     selected_track_info = library.tracks_df[
#         library.tracks_df["track_id"] == selected_track
#     ].iloc[0]
#     st.subheader("Reference Track")
#     safe_audio_player(selected_track_info["audio_path"])

#     try:
#         descriptor_stats = compute_track_statistics(
#             library.tracks_df[library.tracks_df["track_id"] == selected_track]
#         )
#     except Exception:
#         descriptor_stats = None
#     cols = st.columns(4)
#     if descriptor_stats:
#         for idx, (desc, data) in enumerate(descriptor_stats.items()):
#             with cols[idx % 4]:
#                 if desc == "key":
#                     key_info = selected_track_info["key"]["temperley"]
#                     st.caption(data["label"])
#                     st.subheader(f"{key_info['key']} {key_info['scale']}")
#                 elif desc == "voice_prob":
#                     st.caption(data["label"])
#                     st.subheader(
#                         f"{selected_track_info['voice_instrumental']['voice']:.2f}"
#                     )
#                 else:
#                     st.caption(data["label"])
#                     st.subheader(f"{data['mean']:.2f}")

#     similarity_threshold = st.slider(
#         "Minimum Similarity (%)", min_value=0, max_value=100, value=50
#     )
#     max_results = st.slider(
#         "Number of similar tracks to show", min_value=10, max_value=20, value=10
#     )
#     if st.button("Find Similar Tracks"):
#         effnet_tracks = library.find_similar_tracks(
#             selected_track, "discogs-effnet", n_results=max(10, max_results)
#         )
#         musicnn_tracks = library.find_similar_tracks(
#             selected_track, "msd-musicnn", n_results=max(10, max_results)
#         )
#         threshold = similarity_threshold / 100.0
#         effnet_matched = [
#             {**t, "below_threshold": t["similarity"] < threshold}
#             for t in effnet_tracks[:max_results]
#         ]
#         musicnn_matched = [
#             {**t, "below_threshold": t["similarity"] < threshold}
#             for t in musicnn_tracks[:max_results]
#         ]
#         st.session_state.effnet_tracks = effnet_matched
#         st.session_state.musicnn_tracks = musicnn_matched

#     if hasattr(st.session_state, "effnet_tracks"):
#         st.markdown("## Results Analysis")
#         overlap_analysis = analyze_similarity_overlap(
#             st.session_state.effnet_tracks, st.session_state.musicnn_tracks
#         )
#         stats_cols = st.columns(4)
#         stats_cols[0].metric(
#             "Total Effnet Results", len(st.session_state.effnet_tracks)
#         )
#         stats_cols[1].metric(
#             "Total Musicnn Results", len(st.session_state.musicnn_tracks)
#         )
#         stats_cols[2].metric("Overlapping Results", overlap_analysis["overlap_count"])
#         stats_cols[3].metric(
#             "Overlap Percentage",
#             f"{(overlap_analysis['overlap_count'] * 100 / max(len(st.session_state.effnet_tracks), 1)):.1f}%",
#         )
#         result_tabs = st.tabs(
#             ["Side-by-Side Comparison", "Overlapping Tracks", "All Results"]
#         )
#         with result_tabs[0]:
#             col_left, col_right = st.columns(2)
#             with col_left:
#                 st.subheader("Discogs-Effnet Results")
#                 if st.session_state.effnet_tracks:
#                     paginated_effnet = paginate_tracks(
#                         st.session_state.effnet_tracks, "effnet"
#                     )
#                     for track in paginated_effnet:
#                         similarity_text = f"{track['similarity']:.0%}"
#                         if track.get("below_threshold", False):
#                             st.write(
#                                 f"**{track['track_id']}** ({similarity_text}) :warning: *Below threshold*"
#                             )
#                         else:
#                             st.write(f"**{track['track_id']}** ({similarity_text})")
#                         safe_audio_player(track["audio_path"])
#                 else:
#                     st.write("No matches found.")
#             with col_right:
#                 st.subheader("MSD-Musicnn Results")
#                 if st.session_state.musicnn_tracks:
#                     paginated_musicnn = paginate_tracks(
#                         st.session_state.musicnn_tracks, "musicnn"
#                     )
#                     for track in paginated_musicnn:
#                         similarity_text = f"{track['similarity']:.0%}"
#                         if track.get("below_threshold", False):
#                             st.write(
#                                 f"**{track['track_id']}** ({similarity_text}) :warning: *Below threshold*"
#                             )
#                         else:
#                             st.write(f"**{track['track_id']}** ({similarity_text})")
#                         safe_audio_player(track["audio_path"])
#                 else:
#                     st.write("No matches found.")
#         with result_tabs[1]:
#             if overlap_analysis["overlap_tracks"]:
#                 st.subheader(
#                     f"Tracks Found by Both Methods ({overlap_analysis['overlap_count']})"
#                 )
#                 paginated_overlap = paginate_tracks(
#                     overlap_analysis["overlap_tracks"], "overlap"
#                 )
#                 for track in paginated_overlap:
#                     col1, col2 = st.columns([2, 1])
#                     with col1:
#                         safe_audio_player(track["audio_path"])
#                     with col2:
#                         st.write(f"**{track['track_id']}**")
#                         st.write(f"Effnet similarity: {track['effnet_similarity']:.0%}")
#                         st.write(
#                             f"Musicnn similarity: {track['musicnn_similarity']:.0%}"
#                         )
#                         st.write(f"Average similarity: {track['avg_similarity']:.0%}")
#                         if track.get("below_threshold", False):
#                             st.write(":warning: *Below threshold*")
#             else:
#                 st.write("No overlapping tracks found.")
#         with result_tabs[2]:
#             st.subheader("All Results")
#             all_tracks_combined = []
#             seen_ids = set()
#             for track in (
#                 st.session_state.effnet_tracks + st.session_state.musicnn_tracks
#             ):
#                 if track["track_id"] not in seen_ids:
#                     seen_ids.add(track["track_id"])
#                     all_tracks_combined.append(track)
#             paginated_all = paginate_tracks(all_tracks_combined, "all")
#             for track in paginated_all:
#                 col1, col2 = st.columns([2, 1])
#                 with col1:
#                     safe_audio_player(track["audio_path"])
#                 with col2:
#                     st.write(f"**{track['track_id']}**")
#                     found_in = []
#                     if any(
#                         t["track_id"] == track["track_id"]
#                         for t in st.session_state.effnet_tracks
#                     ):
#                         found_in.append("Effnet")
#                     if any(
#                         t["track_id"] == track["track_id"]
#                         for t in st.session_state.musicnn_tracks
#                     ):
#                         found_in.append("Musicnn")
#                     st.write(f"Found by: {', '.join(found_in)}")


def render_genre_style_tab(library: MusicLibrary) -> None:
    """
    Render the Genre & Style Analysis tab.

    Displays genre/style statistics, plots, filtering options, and paginated track results.
    """
    st.header("Genre & Style Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Genre Analysis")
        genre_data = pd.DataFrame.from_dict(
            library.genre_stats, orient="index", columns=["activation"]
        ).sort_values("activation", ascending=False)
        st.caption("Genre Activation Statistics")
        genre_stats = {
            "Mean": genre_data["activation"].mean(),
            "Std": genre_data["activation"].std(),
            "Min": genre_data["activation"].min(),
            "Max": genre_data["activation"].max(),
            "Count": len(genre_data),
            "Total Activation": genre_data["activation"].sum(),
        }
        stats_cols = st.columns(3)
        for idx, (stat, value) in enumerate(genre_stats.items()):
            with stats_cols[idx % 3]:
                st.metric(stat, f"{value:.2f}" if isinstance(value, float) else value)
        fig_genres = px.bar(
            genre_data.head(20),
            title="Top 20 Genres",
            labels={"index": "Genre", "activation": "Activation"},
        )
        fig_genres.update_layout(height=300)
        st.plotly_chart(fig_genres, use_container_width=True)
        selected_genres = st.multiselect(
            "Select Genres", options=sorted(library.genre_stats.keys())
        )
    with col2:
        st.subheader("Style Analysis")
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
            columns=["activation"],
        ).sort_values("activation", ascending=False)
        st.caption("Style Activation Statistics")
        style_stats = {
            "Mean": style_data["activation"].mean(),
            "Std": style_data["activation"].std(),
            "Min": style_data["activation"].min(),
            "Max": style_data["activation"].max(),
            "Count": len(style_data),
            "Total Activation": style_data["activation"].sum(),
        }
        stats_cols = st.columns(3)
        for idx, (stat, value) in enumerate(style_stats.items()):
            with stats_cols[idx % 3]:
                st.metric(stat, f"{value:.2f}" if isinstance(value, float) else value)
        fig_styles = px.bar(
            style_data.head(20),
            title="Top 20 Styles",
            labels={"index": "Style", "activation": "Activation"},
        )
        fig_styles.update_layout(height=300)
        st.plotly_chart(fig_styles, use_container_width=True)
        selected_styles = st.multiselect(
            "Select Styles", options=sorted(available_styles)
        )

    st.subheader("Activation Range Filter")
    activation_range = st.slider(
        "Filter by activation probability",
        min_value=0.0,
        max_value=1.0,
        value=(0.1, 1.0),
        step=0.05,
    )
    if selected_genres or selected_styles:
        filtered_tracks = library.filter_tracks(
            genres=selected_genres, styles=selected_styles
        )

        def check_activation_range(track: pd.Series) -> bool:
            if "music_styles" not in track:
                return False
            for genre_str, prob in track["music_styles"].items():
                if prob < activation_range[0] or prob > activation_range[1]:
                    continue
                parent, style = parse_music_style(genre_str)
                if (not selected_genres or parent in selected_genres) and (
                    not selected_styles or style in selected_styles
                ):
                    return True
            return False

        filtered_tracks = filtered_tracks[
            filtered_tracks.apply(check_activation_range, axis=1)
        ]
        st.subheader(f"Found {len(filtered_tracks)} matching tracks")
        if st.button(
            "Export as M3U8", key="export_genre_style", use_container_width=True
        ):
            from utils import create_m3u8_playlist

            with st.spinner("Creating playlist file..."):
                try:
                    track_list = [
                        {"track_id": row["track_id"], "audio_path": row["audio_path"]}
                        for _, row in filtered_tracks.iterrows()
                    ]
                    if track_list:
                        success = create_m3u8_playlist(
                            track_list, "genre_style_playlist.m3u8"
                        )
                        if success:
                            st.success("✅ Playlist exported successfully!")
                        else:
                            st.error("❌ No valid tracks found for playlist export.")
                    else:
                        st.error("❌ No tracks to export.")
                except Exception as e:
                    st.error(f"❌ Error exporting playlist: {str(e)}")
        st.subheader("Matching Tracks:")
        header_cols = st.columns([1, 2, 1, 1, 1])
        with header_cols[0]:
            st.subheader("Track ID")
        with header_cols[1]:
            st.subheader("Player")
        with header_cols[2]:
            st.subheader("Genre")
        with header_cols[3]:
            st.subheader("Style")
        with header_cols[4]:
            st.subheader("Probability")
        paginated_tracks = paginate_tracks(
            filtered_tracks.to_dict("records"), "genre_style"
        )
        for track in paginated_tracks:
            row_cols = st.columns([1, 2, 1, 1, 1])
            with row_cols[0]:
                st.write(f"**{track['track_id']}**")
            with row_cols[1]:
                safe_audio_player(track["audio_path"])
            if "music_styles" in track:
                category_probs = []
                for genre_str, prob in track["music_styles"].items():
                    if prob <= 0:
                        continue
                    parent, style = parse_music_style(genre_str)
                    if (
                        (not selected_genres or parent in selected_genres)
                        and (not selected_styles or style in selected_styles)
                        and activation_range[0] <= prob <= activation_range[1]
                    ):
                        category_probs.append((parent, style, prob))
                category_probs.sort(key=lambda x: x[2], reverse=True)
                if category_probs:
                    with row_cols[2]:
                        st.write("\n".join([f"**{p[0]}**" for p in category_probs]))
                    with row_cols[3]:
                        st.write("\n".join([f"**{p[1]}**" for p in category_probs]))
                    with row_cols[4]:
                        st.write("\n".join([f"**{p[2]:.2f}**" for p in category_probs]))
