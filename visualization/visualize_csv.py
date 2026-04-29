#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path('/Users/iamsergio/Desktop/NLP_Course-Evals')
RESULTS_DIR = PROJECT_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'visualizations'
FEEDBACK_COL = 'Feedback'
OTHER_LABEL = 'None of the above / Other'


OUTPUT_FILES = [
    'avg_topics_per_comment.png',
    'other_usage_by_model.png',
    'topic_count_distribution.png',
    'overclassification_examples.csv',
    'llama_vs_gemini_examples.png',
]


def prettify_model_name(name: str) -> str:
    lowered = name.lower()
    if 'gemini' in lowered:
        return 'Gemini-2.5'
    if 'llama' in lowered:
        return 'Llama3'
    if 'distil' in lowered:
        return 'DistilroBERTa'
    if 'roberta' in lowered and 'distil' not in lowered:
        return 'roBERTa'
    return name


def classification_csvs() -> list[Path]:
    return sorted(
        path
        for path in RESULTS_DIR.glob('*/*.csv')
        if 'sentiment' not in path.name.lower()
    )


def topic_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != FEEDBACK_COL]


def normalize_assignment(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str) and value.strip():
        return 1
    return 0


def melt_assignments(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    model_name = prettify_model_name(csv_path.parent.name)
    topics = topic_columns(df)

    binary = df.copy()
    for col in topics:
        binary[col] = binary[col].apply(normalize_assignment)
    binary['Model'] = model_name
    binary['Comment_ID'] = range(1, len(binary) + 1)
    binary['Num_Topics'] = binary[topics].sum(axis=1)
    binary['Assigned_Other'] = binary[OTHER_LABEL]

    long_df = binary.melt(
        id_vars=['Model', 'Comment_ID', FEEDBACK_COL, 'Num_Topics', 'Assigned_Other'],
        value_vars=topics,
        var_name='Topic',
        value_name='Assigned',
    )
    long_df = long_df[long_df['Assigned'] == 1].copy()
    return binary, long_df


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_outputs() -> None:
    for filename in OUTPUT_FILES:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_avg_topics_per_comment(binary_frames: list[pd.DataFrame]) -> None:
    combined = pd.concat(binary_frames, ignore_index=True)
    summary = (
        combined.groupby('Model')['Num_Topics']
        .mean()
        .reset_index(name='Average Topics')
        .sort_values('Average Topics', ascending=False)
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x='Model', y='Average Topics')
    plt.title('Average Number of Topics Assigned per Comment')
    plt.ylabel('Average Topics')
    plt.xlabel('Model')
    save_plot('avg_topics_per_comment.png')


def plot_topics_per_comment_distribution(binary_frames: list[pd.DataFrame]) -> None:
    combined = pd.concat(binary_frames, ignore_index=True)
    capped = combined.copy()
    capped['Topic Bucket'] = capped['Num_Topics'].apply(lambda x: str(int(x)) if x < 4 else '4+')
    counts = (
        capped.groupby(['Model', 'Topic Bucket'])
        .size()
        .reset_index(name='Count')
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts, x='Model', y='Count', hue='Topic Bucket', hue_order=['0', '1', '2', '3', '4+'])
    plt.title('Distribution of Topic Counts per Comment')
    plt.ylabel('Number of Comments')
    plt.xlabel('Model')
    save_plot('topic_count_distribution.png')


def plot_other_usage(binary_frames: list[pd.DataFrame]) -> None:
    combined = pd.concat(binary_frames, ignore_index=True)
    summary = (
        combined.groupby('Model')['Assigned_Other']
        .mean()
        .mul(100)
        .reset_index(name='Percent Other')
        .sort_values('Percent Other', ascending=False)
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary, x='Model', y='Percent Other')
    plt.title(f'Percent of Comments Assigned to "{OTHER_LABEL}"')
    plt.ylabel('Percent of Comments')
    plt.xlabel('Model')
    save_plot('other_usage_by_model.png')


def build_overclassification_examples(binary_frames: list[pd.DataFrame], long_frames: list[pd.DataFrame]) -> None:
    rows = []
    for binary_df, long_df in zip(binary_frames, long_frames):
        model_name = binary_df['Model'].iloc[0]
        examples = binary_df.sort_values('Num_Topics', ascending=False).head(10)
        for _, row in examples.iterrows():
            assigned_topics = long_df.loc[long_df['Comment_ID'] == row['Comment_ID'], 'Topic'].tolist()
            rows.append(
                {
                    'Model': model_name,
                    'Comment_ID': int(row['Comment_ID']),
                    'Num_Topics': int(row['Num_Topics']),
                    'Assigned_Other': int(row['Assigned_Other']),
                    'Assigned_Topics': ' | '.join(assigned_topics),
                    'Feedback': row[FEEDBACK_COL],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'overclassification_examples.csv', index=False)


def build_llama_vs_gemini_examples(binary_frames: list[pd.DataFrame], long_frames: list[pd.DataFrame]) -> None:
    binary_by_model = {df['Model'].iloc[0]: df.copy() for df in binary_frames}
    long_by_model = {df['Model'].iloc[0]: df.copy() for df in long_frames}

    if 'Llama3' not in binary_by_model or 'Gemini-2.5' not in binary_by_model:
        return

    llama_binary = binary_by_model['Llama3'][[FEEDBACK_COL, 'Comment_ID', 'Num_Topics', 'Assigned_Other']].copy()
    gemini_binary = binary_by_model['Gemini-2.5'][[FEEDBACK_COL, 'Comment_ID', 'Num_Topics', 'Assigned_Other']].copy()
    llama_binary = llama_binary.rename(columns={'Num_Topics': 'Llama_Num_Topics', 'Assigned_Other': 'Llama_Assigned_Other'})
    gemini_binary = gemini_binary.rename(columns={'Num_Topics': 'Gemini_Num_Topics', 'Assigned_Other': 'Gemini_Assigned_Other'})

    merged = llama_binary.merge(gemini_binary, on=FEEDBACK_COL, how='inner')

    llama_topics_map = long_by_model['Llama3'].groupby(FEEDBACK_COL)['Topic'].apply(lambda s: sorted(set(s.tolist()))).to_dict()
    gemini_topics_map = long_by_model['Gemini-2.5'].groupby(FEEDBACK_COL)['Topic'].apply(lambda s: sorted(set(s.tolist()))).to_dict()

    rows = []
    for _, row in merged.iterrows():
        feedback = row[FEEDBACK_COL]
        llama_topics = llama_topics_map.get(feedback, [])
        gemini_topics = gemini_topics_map.get(feedback, [])
        if llama_topics == gemini_topics:
            continue

        llama_substantive = [t for t in llama_topics if t != OTHER_LABEL]
        gemini_substantive = [t for t in gemini_topics if t != OTHER_LABEL]

        likely_better = ''
        reason = 'Models disagree on the topic set.'
        if llama_substantive and not gemini_substantive:
            likely_better = 'Llama3'
            reason = 'Llama3 assigns substantive rubric topics while Gemini falls back to Other.'
        elif row['Gemini_Assigned_Other'] and not row['Llama_Assigned_Other']:
            likely_better = 'Llama3'
            reason = 'Gemini-2.5 defaults to Other while Llama3 makes a specific assignment.'
        elif len(llama_substantive) > len(gemini_substantive):
            likely_better = 'Llama3'
            reason = 'Llama3 captures more substantive rubric topics.'
        elif len(gemini_substantive) > len(llama_substantive):
            likely_better = 'Gemini-2.5'
            reason = 'Gemini-2.5 captures more substantive rubric topics.'

        rows.append({
            'Feedback': feedback,
            'Llama_Topics': llama_topics,
            'Gemini_Topics': gemini_topics,
            'Llama_Num_Topics': int(row['Llama_Num_Topics']),
            'Gemini_Num_Topics': int(row['Gemini_Num_Topics']),
            'Likely_Better_Model': likely_better,
            'Reason': reason,
        })

    disagreements = pd.DataFrame(rows)
    if disagreements.empty:
        return

    preferred = disagreements[disagreements['Likely_Better_Model'] == 'Llama3'].copy()
    if preferred.empty:
        preferred = disagreements.copy()

    preferred['Llama_Topic_Advantage'] = preferred['Llama_Num_Topics'] - preferred['Gemini_Num_Topics']
    preferred = preferred.sort_values(['Llama_Topic_Advantage', 'Llama_Num_Topics'], ascending=[False, False]).head(2)

    fig, axes = plt.subplots(len(preferred), 1, figsize=(16, 4.8 * len(preferred)))
    if len(preferred) == 1:
        axes = [axes]

    for ax, (_, example) in zip(axes, preferred.iterrows()):
        ax.axis('off')
        feedback = example['Feedback'].strip().replace('\n', ' ')
        llama_topics = '\n'.join(f'• {t}' for t in example['Llama_Topics']) or '• None'
        gemini_topics = '\n'.join(f'• {t}' for t in example['Gemini_Topics']) or '• None'

        ax.text(0.00, 0.95, 'Comment', fontsize=16, fontweight='bold', va='top')
        ax.text(0.00, 0.80, feedback, fontsize=12.5, va='top', wrap=True)

        ax.text(0.02, 0.42, 'Llama3', fontsize=15, fontweight='bold', color='#2f6f4f', va='top')
        ax.text(0.02, 0.36, llama_topics, fontsize=12.5, va='top', linespacing=1.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#2f6f4f'))

        ax.text(0.52, 0.42, 'Gemini-2.5', fontsize=15, fontweight='bold', color='#8c4b1f', va='top')
        ax.text(0.52, 0.36, gemini_topics, fontsize=12.5, va='top', linespacing=1.5, bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#8c4b1f'))

        ax.text(0.00, 0.08, f"Why this favors {example['Likely_Better_Model']}: {example['Reason']}", fontsize=12, style='italic', va='top')
        ax.axhline(0.02, color='#cccccc', linewidth=1)

    fig.suptitle('Representative Topic Classification Differences: Llama3 vs Gemini-2.5', fontsize=20, fontweight='bold', y=0.995)
    save_plot('llama_vs_gemini_examples.png')

def main() -> None:
    ensure_output_dir()
    clear_old_outputs()
    sns.set_theme(style='whitegrid', context='talk')

    csv_files = classification_csvs()
    if not csv_files:
        print(f'No classification CSVs found in {RESULTS_DIR}')
        return

    binary_frames = []
    long_frames = []
    for csv_path in csv_files:
        binary_df, long_df = melt_assignments(csv_path)
        binary_frames.append(binary_df)
        long_frames.append(long_df)

    plot_avg_topics_per_comment(binary_frames)
    plot_other_usage(binary_frames)
    plot_topics_per_comment_distribution(binary_frames)
    build_overclassification_examples(binary_frames, long_frames)
    build_llama_vs_gemini_examples(binary_frames, long_frames)

    print('Saved 5 key poster outputs to', OUTPUT_DIR)
    for filename in OUTPUT_FILES:
        print('-', OUTPUT_DIR / filename)


if __name__ == '__main__':
    main()
