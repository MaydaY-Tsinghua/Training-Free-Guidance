import csv 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Set a new color palette for the plots
palette = sns.color_palette("husl")  # Use a different color map
sns.set_palette(palette)

Flux_results = {'color': 0.74, 'shape': 0.57, 'texture': 0.69, 'numeracy': 0.62, 'spatial': 0.29}

# datasets = ['texture', 'color', 'shape']
datasets = ['numeracy', 'spatial']
colors = {dataset: palette[i] for i, dataset in enumerate(datasets)}  # Assign consistent colors to datasets
threshold = 0.7
for dataset in datasets:
    # Plot Best-of-N (bon) results
    csv_file = f"results_{dataset}_SSD_bon.csv"
    df = pd.read_csv(csv_file)
    df = df.groupby('particles', as_index=False).mean()
    df = df.sort_values(by=['particles'])
    if dataset == 'numeracy' or 'spatial':
        df = df[(df['particles'] <= 8) & (df['particles'] != 6) & (df['particles'] >= 2) ]
    else:
        df = df[(df['particles'] <= 8) & (df['particles'] % 2 == 0) ]
    plt.plot(df['particles'] * 50, df[' score'], linewidth=2, marker='o', markersize=6, linestyle='-', color=colors[dataset])

    # Plot DFS results
    csv_file = f"results_{dataset}_dfs.csv"
    df = pd.read_csv(csv_file)
    if dataset == 'numeracy':
        df = df[((df['start']==25) & (df['recur_depth']==25) & (df['threshold'] == 0.4)) | (df['start']==35) & (df['recur_depth']==35) & (df['threshold']==0.7)]
    else:
        df = df[((df['start']==25) & (df['recur_depth']==25) & (df['threshold'] == 0.7)) | (df['start']==35) & (df['recur_depth']==35) & (df['threshold']==0.7)]
    df = df.sort_values(by=['compute'])
    plt.plot(df['compute'], df['score'], linewidth=2, marker='s', markersize=6, linestyle='--', color=colors[dataset])

    # # Plot Flux results
    # flux_score = Flux_results[dataset]
    # plt.scatter(500, flux_score, color=colors[dataset], marker='*', s=100, label=f'Flux ({dataset})')

plt.xlabel('Compute')
plt.ylabel('Score')
plt.xscale('log', base=2)
# plt.xscale('linear')
plt.yscale('linear')
plt.title('DFS results for Object Relationships')

# Create custom legend entries for datasets and line styles
dataset_legend = [Line2D([0], [0], color=colors[dataset], lw=2, label=dataset) for dataset in datasets]
style_legend = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Best-of-N'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='DFS')
]

# Combine legends
plt.legend(handles=dataset_legend + style_legend, loc='best')

plt.grid()
plt.savefig(f'DFS_obj.png')

