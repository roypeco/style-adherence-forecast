import pandas as pd
import os
from modules.get_cluster import get_cluster

dir_path = "dataset/row_data/"
cluster_num_list= []
start_num = 0

project_list = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]

# クラスタを計算し，ラベルのリストを取得
# cluster_label, _ = get_cluster(10)

for project_name in project_list:
  df_label = len(pd.read_csv(f'dataset/outputs/{project_name}_label.csv', header=None).values.ravel())
  # ToDo
