import pandas as pd
import matplotlib.pyplot as plt
import argparse

# CSVファイルを読み込む関数
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df[['project', 'percentage']] = df.iloc[:, 0].str.extract(r'([^@]+)@([0-9]+)%')
    df['percentage'] = df['percentage'].astype(int)
    return df

# プロジェクト名ごとにプロットする関数
def plot_combined_metrics(df_list, titles, selected_metric=None):
    combined_df = pd.concat(df_list, keys=titles, names=['Dataset', 'Index'])
    projects = combined_df['project'].unique()
    metrics = ["precision", "recall", "f1_score", "accuracy"]
    line_styles = {"Dataset A": "-", "Dataset B": "--"}
    colors = {"precision": "blue", "recall": "green", "f1_score": "red", "accuracy": "orange"}

    # 使用する評価指標を決定
    if selected_metric and selected_metric in metrics:
        metrics = [selected_metric]

    # プロジェクトごとに分割して表示
    n_projects = len(projects)
    projects_per_figure = 10  # 1つのウィンドウに表示するプロジェクト数
    total_figures = (n_projects + projects_per_figure - 1) // projects_per_figure

    for fig_idx in range(total_figures):
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))  # 横5列、縦2行
        axes = axes.flatten()

        start_idx = fig_idx * projects_per_figure
        end_idx = min(start_idx + projects_per_figure, n_projects)
        for i, project in enumerate(projects[start_idx:end_idx]):
            ax = axes[i]
            for metric in metrics:
                for dataset in titles:
                    dataset_data = combined_df.loc[dataset]
                    project_data = dataset_data[dataset_data['project'] == project]
                    ax.plot(
                        project_data['percentage'], 
                        project_data[metric], 
                        marker='o', 
                        linestyle=line_styles[dataset], 
                        color=colors[metric], 
                        label=f"{dataset} - {metric}"
                    )

            ax.set_title(f"Project: {project}", fontsize=12)
            ax.set_xlabel("Percentage", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.set_xticks(sorted(combined_df['percentage'].unique()))
            ax.set_yticks([i * 0.2 for i in range(6)])  # 縦軸の目盛りを0.0から1.0に設定
            ax.set_ylim(0.0, 1.0)
            ax.legend(fontsize=8)
            ax.grid()

        # 余分なプロットを非表示
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f'results/{fig_idx}.png')
        plt.show()

# メイン関数
def main():
    parser = argparse.ArgumentParser(description="Plot metrics from CSV files.")
    parser.add_argument("-p", "--precision", action="store_true", help="Plot precision only.")
    parser.add_argument("-r", "--recall", action="store_true", help="Plot recall only.")
    parser.add_argument("-f1", "--f1_score", action="store_true", help="Plot F1 score only.")
    parser.add_argument("-a", "--accuracy", action="store_true", help="Plot accuracy only.")

    args = parser.parse_args()

    # 指定された指標を確認
    selected_metric = None
    if args.precision:
        selected_metric = "precision"
    elif args.recall:
        selected_metric = "recall"
    elif args.f1_score:
        selected_metric = "f1_score"
    elif args.accuracy:
        selected_metric = "accuracy"

    # 既定のファイルパス
    file_a = "results/soloStepDecisionTree.csv"
    file_b = "results/stepBystepResultDecisionTree.csv"

    # データ読み込み
    df_a = load_csv(file_a)
    df_b = load_csv(file_b)

    # プロット
    plot_combined_metrics([df_a, df_b], ["Dataset A", "Dataset B"], selected_metric=selected_metric)

if __name__ == "__main__":
    main()
