import csv

import pandas as pd
from modules.get_cluster import get_cluster

start_num = 0
cluster_num = 10

# クラスタを計算し，ラベルのリストを取得
cluster_label, project_list = get_cluster(cluster_num)

for project_name in project_list:
    df_label = len(
        pd.read_csv(
            f"dataset/outputs/{project_name}_label.csv", header=None
        ).values.ravel()
    )

    output_file_path = f"dataset/outputs/{project_name}_cluster{cluster_num}.csv"

    with open(output_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in cluster_label[start_num : start_num + df_label]:
            writer.writerow([i])

    start_num += df_label
