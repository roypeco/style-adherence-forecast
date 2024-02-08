import csv
import os
import warnings

import pandas as pd
from modules import label_sep

# warning文の無視
warnings.simplefilter("ignore")

## 宣言
# ディレクトリパス
dir_path = "dataset/row_data"

# dataset内のプロジェクト名一覧取得
with open("dataset/project_list.txt") as f:
    project_list = f.read().splitlines()


exit_flg = False

col_label = [
    [
        "Warning ID",
        "Project_name",
        "AvgCyclomatic",
        "AvgCyclomaticModified",
        "AvgCyclomaticStrict",
        "AvgEssential",
        "AvgLine",
        "AvgLineBlank",
        "AvgLineCode",
        "AvgLineComment",
        "CountClassDerived",
        "CountClassBase",
        "CountDeclClass",
        "CountDeclFile",
        "CountDeclFunction",
        "CountDeclInstanceMethod",
        "CountDeclInstanceVariable",
        "CountDeclMethod",
        "CountDeclMethodAll",
        "CountLine",
        "CountLineBlank",
        "CountLineCode",
        "CountLineCodeDecl",
        "CountLineCodeExe",
        "CountLineComment",
        "CountPath",
        "CountPathLog",
        "CountStmt",
        "CountStmtDecl",
        "CountStmtExe",
        "Cyclomatic",
        "CyclomaticModified",
        "CyclomaticStrict",
        "Essential",
        "MaxCyclomatic",
        "MaxCyclomaticModified",
        "MaxCyclomaticStrict",
        "MaxEssential",
        "MaxInheritanceTree",
        "MaxNesting",
        "RatioCommentToCode",
        "SumCyclomatic",
        "SumCyclomaticModified",
        "SumCyclomaticStrict",
        "SumEssential",
    ]
]

for project_name in project_list:
    # for project_name in project_list[70:]: # デバッグ用
    with open(f"{dir_path}/{project_name}/pylintc_commit_hash.csv") as f:
        lines = f.readlines()  # 行ごとにリストで読み込み
        lines_ = [line.strip() for line in lines]  # 改行コードの削除
        hash_list = list(reversed(lines_))[
            :-1
        ]  # 逆順にしてヘッダ(pylintrc_commit_hash)を消す

    label = []  # 正解ラベル格納
    value_data = []  # 説明変数データ格納
    df_history = pd.read_csv(
        f"{dir_path}/{project_name}/coding_fix_history.csv"
    )  # 違反修正履歴の読み込み
    file_path_list = df_history[
        "Filepath"
    ].to_list()  # 違反発生ファイルパスのリスト取得
    edge = (len(df_history.iloc[0]) - 3) // 2  # 計測期間内の最終リビジョン数
    lnumber = df_history[
        "Line Number_" + str(edge)
    ].to_list()  # 最終リビジョンにおける違反発生コード行数(-1だと削除されたか修正されたか)

    # status_?内を全探索
    for status_num in range(1, edge + 1):
        status_list = df_history[
            "Status_" + str(status_num)
        ].to_list()  # 探索中のステータスリスト
        for status_index in range(
            len(status_list)
        ):  # status_listを探索するためのインデックスを全探索
            try:
                if (
                    status_list[status_index] == "New Warning"
                    and "test" not in file_path_list[status_index].lower()
                ):  # テストコードの排除
                    # print(status_list[status_index], file_path_list[status_num], end="\t") # 確認用コード
                    # ------------- 目的変数ラベルの取得 -------------
                    if label_sep.label_catcher(
                        status_index, status_num, edge, df_history
                    ):
                        label.append(1)
                        # print("True") # 確認用コード
                    else:
                        label.append(0)
                        # print("False") # 確認用コード
                    # -----------------------------------------------

                    # ------------- 説明変数の取得 -------------
                    file_path = file_path_list[
                        status_index
                    ]  # 違反が発生していたファイル名を取得
                    sat_result_path = f"{dir_path}/{project_name}/und_result/{hash_list[status_num-1]}.csv"  # 読み込むファイルのパスの設定
                    df_sat = pd.read_csv(sat_result_path).fillna(
                        0
                    )  # 各リビジョンごとのsatの結果の読み込み
                    if f"{project_name}/{file_path}" in df_sat["File"].to_list():
                        file_path_index = (
                            df_sat["File"]
                            .to_list()
                            .index(f"{project_name}/{file_path}")
                        )
                    else:
                        file_path_index = df_sat["File"].to_list().index(file_path)

                    data_vector = df_sat.iloc[file_path_index].values.tolist()[3:]
                    data_vector.insert(0, str(project_name))
                    data_vector.insert(0, str(df_history["Warning ID"][status_index]))
                    value_data.append(data_vector)

            except ValueError as e:
                label.pop()
                if "yuta-m" in df_history["Module"][status_index]:
                    continue
                print(e)
                print(hash_list[status_num - 1])
                # exit_flg = True
                # break

            except FileNotFoundError as e:
                label.pop()
                print(e)
                exit_flg = True
                break

            except IndexError as e:
                label.pop()
                print(
                    f"status_num:{status_num}, status_index:{status_index}, file_path_list:{len(file_path_list)}, Error:{e}"
                )
                exit_flg = True
                break

        if exit_flg:
            break

    # ファイルへの書き出し
    f = open(f"dataset/outputs/{project_name}_value.csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerows(col_label)
    writer.writerows(value_data)
    f.close()

    f = open(f"dataset/outputs/{project_name}_label.csv", "w")
    writer = csv.writer(f, lineterminator="\n")
    for i in label:
        writer.writerow([i])
    f.close()

    # print(file_path_list)
    print(
        f"{project_name} : Done {project_list.index(project_name)+1}/{len(project_list)}"
    )

    # 1プロジェクト確認デバッグ用
    # break

# print(value_data)
# print(hash_list)
# print(file_path_list)
# print(edge)
