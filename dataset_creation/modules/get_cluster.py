import os

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler


# 正規化メソッド
def normalize_dataframe(input_df: pd.DataFrame):
    # MinMaxスケーラーのインスタンスを作成
    scaler = MinMaxScaler()

    # 各列ごとに正規化を実行
    normalized_data = scaler.fit_transform(input_df)

    # 正規化されたデータを新しいDataFrameとして作成
    normalized_df = pd.DataFrame(normalized_data, columns=input_df.columns)

    return normalized_df


# データセットができてからクラスタリングをするためのメソッド
def get_cluster(cluster_num: int):
    df_all = pd.DataFrame()

    with open("dataset/white_list.txt") as f:
        project_list = f.read().splitlines()

    for priject_name in project_list:
        df = pd.read_csv(f"dataset/outputs/{priject_name}_value.csv")
        df_all = pd.concat([df_all, df], ignore_index=True)

    # ラベルを得る
    # モデル訓練
    # df_norm = normalize_dataframe(df_all.iloc[:, 2:])
    # print(df_norm)
    model = AgglomerativeClustering(
        n_clusters=cluster_num, metric="euclidean", linkage="ward"
    )
    y_hir_clus = model.fit_predict(df_all.iloc[:, 2:])
    for i in range(cluster_num):
        print(list(y_hir_clus).count(i))

    return y_hir_clus, project_list
