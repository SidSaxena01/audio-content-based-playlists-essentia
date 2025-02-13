import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
from tqdm import tqdm

# plt.style.available
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class MusicCollectionAnalyzer:
    def __init__(self, analysis_dir, output_dir):
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.df = None
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """Load all JSON analysis files into a DataFrame"""
        json_files = []
        for root, _, files in os.walk(self.analysis_dir):
            for f in files:
                if f.endswith('.json'):
                    json_files.append(os.path.join(root, f))

        data = []
        for json_file in tqdm(json_files, desc='Loading analysis files'):
            with open(json_file) as f:
                entry = json.load(f)
                entry['path'] = os.path.relpath(json_file, self.analysis_dir)
                data.append(entry)

        self.df = pd.DataFrame(data)
        return self

    def preprocess_data(self):
        """Preprocess data for analysis"""
        # Process genres
        def process_genres(genres):
            return [(g['genre'].split('—', 1), g['probability']) 
                    for g in genres]

        self.df['genres_processed'] = self.df['music_styles'].apply(process_genres)
        
        # Process keys
        key_profiles = ['temperley', 'krumhansl', 'edma']
        for profile in key_profiles:
            self.df[f'key_{profile}'] = self.df['key'].apply(
                lambda x: f"{x[profile]['key']} {x[profile]['scale'].capitalize()}")

        return self

    def analyze_styles(self):
        """Analyze music style distribution"""
        # Create style TSV
        style_counts = pd.Series(
            [style for track in self.df.genres_processed 
             for (_, style), _ in track]
        ).value_counts().reset_index()
        style_counts.columns = ['Style', 'Count']
        style_counts.to_csv(os.path.join(self.output_dir, 'style_distribution.tsv'), 
                          sep='\t', index=False)

        # Parent genre analysis
        genre_data = []
        for track_genres in self.df.genres_processed:
            total_prob = sum(prob for _, prob in track_genres)
            for (parent, _), prob in track_genres:
                genre_data.append({
                    'parent': parent,
                    'weight': prob / total_prob
                })

        parent_df = pd.DataFrame(genre_data)
        top_parents = parent_df.groupby('parent')['weight'].sum().nlargest(20)

        # Plot parent genre distribution
        plt.figure()
        top_parents.sort_values().plot(kind='barh')
        plt.title('Top 20 Parent Genre Distribution')
        plt.xlabel('Weighted Count')
        plt.ylabel('Parent Genre')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parent_genre_distribution.png'))
        plt.close()

        return self

    def analyze_tempo_danceability(self):
        """Analyze tempo and danceability distributions"""
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        
        # Tempo histogram
        sns.histplot(self.df['tempo'], bins=50, ax=axs[0], kde=True)
        axs[0].set_title('Tempo Distribution')
        axs[0].set_xlabel('BPM')
        axs[0].set_xlim(60, 180)

        # Danceability boxplot
        sns.violinplot(y=self.df['danceability'], ax=axs[1], inner='quartile')
        axs[1].set_title('Danceability Distribution')
        axs[1].set_ylabel('Danceability Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tempo_danceability.png'))
        plt.close()

        return self

    def analyze_keys(self):
        """Analyze key and scale distributions"""
        key_profiles = ['temperley', 'krumhansl', 'edma']
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))

        # Key agreement calculation
        self.df['key_match'] = self.df[[f'key_{p}' for p in key_profiles]].nunique(axis=1) == 1
        agreement_rate = self.df['key_match'].mean()

        # Plot key distributions
        for i, profile in enumerate(key_profiles):
            key_counts = self.df[f'key_{profile}'].value_counts().sort_index()
            key_order = sorted(key_counts.index, 
                              key=lambda x: (x.split()[0], x.split()[1]))
            
            sns.barplot(x=key_counts.values, y=key_counts.index, 
                        order=key_order, ax=axs[i])
            axs[i].set_title(f'Key Distribution ({profile.capitalize()})')
            axs[i].set_xlabel('Count')
            axs[i].set_ylabel('')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'key_distributions.png'))
        plt.close()

        print(f"Key agreement rate between all profiles: {agreement_rate:.1%}")
        return self

    def analyze_loudness(self):
        """Analyze loudness distribution"""
        plt.figure()
        sns.histplot(self.df['loudness'], bins=30, kde=True)
        plt.axvline(-14, color='r', linestyle='--', label='Standard Music Reference (-14 LUFS)')
        plt.axvline(-23, color='g', linestyle='--', label='Broadcast Standard (-23 LUFS)')
        plt.title('Integrated Loudness Distribution')
        plt.xlabel('LUFS')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'loudness_distribution.png'))
        plt.close()
        return self

    def analyze_emotion(self):
        """Analyze valence-arousal emotion space"""
        plt.figure()
        sns.jointplot(x=self.df['valence'], y=self.df['arousal'], 
                     kind='hex', cmap='viridis')
        plt.suptitle('Valence-Arousal Emotion Space')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'emotion_space.png'))
        plt.close()
        return self

    def analyze_vocal_instrumental(self):
        """Analyze vocal vs instrumental distribution"""
        vocal_ratio = self.df['voice_instrumental'].apply(
            lambda x: x['voice']).mean()

        plt.figure()
        plt.pie([vocal_ratio, 1-vocal_ratio], 
                labels=['Vocal', 'Instrumental'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Vocal vs Instrumental Distribution')
        plt.savefig(os.path.join(self.output_dir, 'vocal_instrumental.png'))
        plt.close()
        return self

    def generate_report(self):
        """Generate full analysis report"""
        self.load_data()
        self.preprocess_data()
        
        print("Starting analysis...")
        (self.analyze_styles()
           .analyze_tempo_danceability()
           .analyze_keys()
           .analyze_loudness()
           .analyze_emotion()
           .analyze_vocal_instrumental())
        
        print(f"Analysis complete! Results saved to {self.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate music collection report')
    parser.add_argument('analysis_dir', help='Directory with JSON analysis files')
    parser.add_argument('output_dir', help='Directory to save report results')
    args = parser.parse_args()

    analyzer = MusicCollectionAnalyzer(args.analysis_dir, args.output_dir)
    analyzer.generate_report()