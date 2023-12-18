import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering

# ToDo
# データセットができてからクラスタリングをするためのメソッド
def get_cluster(cluster_num: int):
  dir_path = "dataset"
  df_all = pd.DataFrame()

  project_list = [
      f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
  ]

  for priject_name in project_list:
    df = pd.read_csv(f"dataset/outputs/{priject_name}_value.csv")
    df_all = pd.concat([df_all, df], ignore_index=True)

  # ラベルを得る
  # モデル訓練
  model = AgglomerativeClustering(n_clusters = cluster_num, metric = 'euclidean', linkage = 'ward')
  y_hir_clus = model.fit_predict(df_all.iloc[:, 3:])
  for i in range(cluster_num):
    print(list(y_hir_clus).count(i))
  
  return y_hir_clus, df_all