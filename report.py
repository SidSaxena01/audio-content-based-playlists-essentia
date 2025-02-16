import argparse
import json
import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Global plot settings
plt.style.use("seaborn-v0_8-deep")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "figure.figsize": (12, 8),
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


class MusicCollectionAnalyzer:
    def __init__(self, analysis_dir, output_dir):
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.df = None
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """Load analysis files with error handling"""
        json_files = []
        for root, _, files in os.walk(self.analysis_dir):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))

        data = []
        for json_file in tqdm(json_files, desc="Loading files"):
            try:
                with open(json_file) as f:
                    entry = json.load(f)
                    entry["path"] = os.path.relpath(json_file, self.analysis_dir)
                    data.append(entry)
            except Exception as e:
                print(f"Skipping {json_file}: {str(e)}")

        self.df = pd.DataFrame(data)
        return self

    def _process_genres(self, genres):
        """Process genre entries from dictionary with proper separator handling"""
        processed = []
        for genre_str, prob in genres.items():
            if "---" in genre_str:
                parts = genre_str.split("---", 1)
            else:
                parts = [genre_str, "unknown-style"]

            parent = parts[0].strip()
            style = parts[1].strip() if len(parts) > 1 else "unknown-style"
            processed.append({"parent": parent, "style": style, "probability": prob})
        return processed

    def preprocess_data(self):
        """Prepare data for analysis"""
        self.df["genres_processed"] = self.df["music_styles"].apply(
            self._process_genres
        )

        # Process keys
        key_profiles = ["temperley", "krumhansl", "edma"]
        for profile in key_profiles:
            self.df[f"key_{profile}"] = self.df["key"].apply(
                lambda x: f"{x[profile]['key']} {x[profile]['scale'].capitalize()}"
            )

        return self

    def analyze_styles(self):
        """Parent genre distribution (keep original)"""
        parent_data = []
        for track in self.df.genres_processed:
            total = sum(g["probability"] for g in track)
            for genre in track:
                parent_data.append(
                    {"parent": genre["parent"], "weight": genre["probability"] / total}
                )

        parent_df = pd.DataFrame(parent_data)
        top_parents = parent_df.groupby("parent")["weight"].sum().nlargest(20)

        # Create style TSV
        style_counts = (
            pd.Series(
                [
                    genre["style"]
                    for track in self.df.genres_processed
                    for genre in track
                ]
            )
            .value_counts()
            .reset_index()
        )
        style_counts.columns = ["Style", "Count"]
        style_counts.to_csv(
            os.path.join(self.output_dir, "style_distribution.tsv"),
            sep="\t",
            index=False,
        )

        plt.figure()
        top_parents.sort_values().plot(kind="barh")
        plt.title("Top 20 Parent Genre Distribution")
        plt.xlabel("Weighted Proportion")
        plt.ylabel("Genre Category")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "parent_genre_distribution.png"))
        plt.close()

        return self

    def analyze_tempo_danceability(self):
        """Revert to original tempo/danceability plots"""
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))

        # Tempo plot
        sns.histplot(self.df["tempo"], bins=50, ax=axs[0], kde=True)
        axs[0].set_title("Tempo Distribution")
        axs[0].set_xlabel("BPM")
        axs[0].set_xlim(60, 180)

        # Danceability plot
        sns.violinplot(y=self.df["danceability"], ax=axs[1], inner="quartile")
        axs[0].set_ylabel("Count")
        axs[1].set_title("Danceability Distribution")
        axs[1].set_ylabel("Danceability Score")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "tempo_danceability.png"))
        plt.close()
        return self

    def analyze_keys(self):
        """Keep improved key comparison plot"""
        # Create comparison DataFrame
        key_data = []
        profiles = ["temperley", "krumhansl", "edma"]
        for profile in profiles:
            counts = self.df[f"key_{profile}"].value_counts().reset_index()
            counts.columns = ["Key", "Count"]
            counts["Profile"] = profile.capitalize()
            key_data.append(counts)

        key_df = pd.concat(key_data)

        # Musical key order
        key_order = [
            "C Major",
            "C Minor",
            "C# Major",
            "C# Minor",
            "D Major",
            "D Minor",
            "Eb Major",
            "Eb Minor",
            "E Major",
            "E Minor",
            "F Major",
            "F Minor",
            "F# Major",
            "F# Minor",
            "G Major",
            "G Minor",
            "Ab Major",
            "Ab Minor",
            "A Major",
            "A Minor",
            "Bb Major",
            "Bb Minor",
            "B Major",
            "B Minor",
        ]

        # Plot comparison
        plt.figure(figsize=(18, 8))
        sns.barplot(
            x="Key",
            y="Count",
            hue="Profile",
            data=key_df,
            order=key_order,
            palette="Set2",
        )
        plt.title("Key Distribution Comparison")
        plt.xlabel("Musical Key")
        plt.ylabel("Number of Tracks")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Estimation Method")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "key_comparison.png"))
        plt.close()
        return self

    def analyze_loudness(self):
        """Keep original loudness plot"""
        plt.figure()
        sns.histplot(self.df["loudness"], bins=30, kde=True)
        plt.axvline(-14, color="r", linestyle="--", label="-14 LUFS (Music)")
        plt.axvline(-23, color="g", linestyle="--", label="-23 LUFS (Broadcast)")
        plt.title("Integrated Loudness Distribution")
        plt.xlabel("LUFS")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "loudness_distribution.png"))
        plt.close()
        return self

    def analyze_emotion(self):
        """Generate both emotion plots using normalized values"""
        # Normalize raw valence/arousal values from [1,9] to [-1,1]
        norm_valence = (self.df["valence"] - 5) / 4.0
        norm_arousal = (self.df["arousal"] - 5) / 4.0

        # joint plot with normalized values
        plt.figure()
        sns.jointplot(x=norm_valence, y=norm_arousal, kind="hex", cmap="viridis")
        plt.suptitle("Valence-Arousal Emotion Space")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "emotion_space_joint.png"))
        plt.close()

        # Another plot with quadrant labels using normalized values
        plt.figure(figsize=(12, 10))
        hexbin = plt.hexbin(
            norm_valence,
            norm_arousal,
            gridsize=25,
            cmap="viridis",
            mincnt=1,
        )
        plt.colorbar(hexbin, label="Track Density")
        plt.title("Musical Emotion Landscape")
        plt.xlabel("Valence (Positive ↔ Negative)")
        plt.ylabel("Arousal (Calm ↔ Energetic)")

        # Quadrant labels
        plt.text(
            0.8,
            0.9,
            "Exciting\nPositive",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            color="white",
            path_effects=[path_effects.withStroke(linewidth=3, foreground="black")],
        )
        plt.text(
            0.2,
            0.9,
            "Stressful\nNegative",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            color="black",
        )
        plt.text(
            0.8,
            0.1,
            "Content\nCalm",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.2,
            0.1,
            "Depressing\nSad",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "emotion_space_quadrants.png"))
        plt.close()
        return self

    def analyze_vocal_instrumental(self):
        """Enhanced vocal/instrumental visualizations"""
        vocal_mean = self.df["voice_instrumental"].apply(lambda x: x["voice"]).mean()

        # Donut Chart
        plt.figure(figsize=(10, 10))
        plt.pie(
            [vocal_mean, 1 - vocal_mean],
            labels=["Vocal", "Instrumental"],
            colors=["#FF6B6B", "#4ECDC4"],
            startangle=90,
            wedgeprops={"linewidth": 4, "edgecolor": "white"},
            textprops={"fontsize": 14},
        )

        # Draw center circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        plt.gca().add_artist(centre_circle)

        # Add percentage text with shadow
        plt.text(
            0,
            0,
            f"{vocal_mean:.1%} Vocal\nTracks",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color="#2d3436",
            path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
        )

        plt.title("Vocal Presence in Collection", fontsize=18, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "vocal_instrumental_donut.png"))
        plt.close()

        return self

    def generate_report(self):
        """Generate complete report"""
        self.load_data()
        if self.df.empty:
            raise ValueError("No valid analysis files found")

        self.preprocess_data()

        print("Generating analysis report...")
        (
            self.analyze_styles()
            .analyze_tempo_danceability()
            .analyze_keys()
            .analyze_loudness()
            .analyze_emotion()
            .analyze_vocal_instrumental()
        )

        print(f"Report successfully generated in {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music collection report")
    parser.add_argument("analysis_dir", help="Directory with JSON analysis files")
    parser.add_argument("output_dir", help="Output directory for report")
    args = parser.parse_args()

    analyzer = MusicCollectionAnalyzer(args.analysis_dir, args.output_dir)
    try:
        analyzer.generate_report()
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        exit(1)
