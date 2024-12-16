from analysis import calc_similarity
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  # 評価指標算出用
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

for jaccard_counter in range(1, 5):
    for distance_counter in range(15, 51, 5):
        project_pattern = calc_similarity.group_up(jaccard_counter/10, distance_counter/10)
        result_df = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])
        counter = 1

        for key, value in project_pattern.items():
            id_dict = {}
            train_df = pd.DataFrame()
            model = LogisticRegression(
                        penalty="l2",  # 正則化項(L1正則化 or L2正則化が選択可能)
                        class_weight="balanced",  # クラスに付与された重み
                        random_state=0,  # 乱数シード
                        solver="lbfgs",  # ハイパーパラメータ探索アルゴリズム
                        max_iter=10000,  # 最大イテレーション数
                        multi_class="auto",  # クラスラベルの分類問題（2値問題の場合'auto'を指定）
                        warm_start=False,  # Trueの場合、モデル学習の初期化に前の呼出情報を利用
                        n_jobs=None,  # 学習時に並列して動かすスレッドの数
                    )
            df_value = pd.read_csv(f"./dataset/outputs/{key}_value.csv")
            df_label = pd.read_csv(f"./dataset/outputs/{key}_label.csv", header=None)
            X_train, X_test, Y_train, Y_test = train_test_split(
                df_value, df_label, test_size=0.2, shuffle=False
            )
            Y_train = Y_train.values.ravel()
            X_train["AnsTF"] = copy.deepcopy(Y_train)
            train_df = pd.concat([train_df, X_train], axis=0)
            Y_test = Y_test.values.ravel()
            X_test["AnsTF"] = Y_test
            X_test = X_test.reset_index(drop=True)
            
            for project_name in value:
                df_value = pd.read_csv(f"./dataset/outputs/{project_name}_value.csv")
                df_label = pd.read_csv(f"./dataset/outputs/{project_name}_label.csv", header=None)
                X_train, _, Y_train, _ = train_test_split(
                    df_value, df_label, test_size=0.2, shuffle=False
                )
                Y_train = Y_train.values.ravel()
                X_train["AnsTF"] = copy.deepcopy(Y_train)
                train_df = pd.concat([train_df, X_train], axis=0)
            
            df_marge = pd.concat(
                [pd.get_dummies(train_df["Warning ID"]), train_df.drop(columns="Warning ID")],
                axis=1,
            )
            dummys = list(pd.get_dummies(train_df["Warning ID"]))
            model.fit(
                    df_marge.drop(["Project_name", "AnsTF"], axis=1), df_marge["AnsTF"]
                )
            
            # 予測
            id_dict.clear()
            for i in list(dummys):
                id_dict[i] = []
            for wid in X_test["Warning ID"]:
                if wid in id_dict:
                    id_dict[wid].append(1)
                    for i in id_dict:
                        id_dict[i].append(0)
                    id_dict[wid].pop(-1)
                else:
                    for i in id_dict:
                        id_dict[i].append(0)
                        
            id_df = pd.DataFrame(id_dict)
            test_df = pd.concat([id_df, X_test], axis=1)
            predict_result = model.predict(
                test_df.drop(["Warning ID", "Project_name", "AnsTF"], axis=1)
            )
            result = {
                "precision": format(
                    precision_score(Y_test, predict_result, zero_division=np.nan), ".2f"
                )
            }
            result["recall"] = format(
                recall_score(Y_test, predict_result, zero_division=np.nan), ".2f"
            )
            result["f1_score"] = format(
                f1_score(Y_test, predict_result, zero_division=np.nan), ".2f"
            )
            result["accuracy"] = format(accuracy_score(Y_test, predict_result), ".2f")

            result_df = pd.concat(
                [result_df, pd.DataFrame([result], index=[key])], axis=0
            )
            print(key, f"{counter} / {len(list(project_pattern.keys()))}")
            counter += 1

        result_df.to_csv(f"results/sample/{jaccard_counter}_{distance_counter}.csv")
        print(result_df)