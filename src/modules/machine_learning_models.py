import copy
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  # 評価指標算出用
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("always", category=UserWarning)


# 従来手法の予測結果の算出
# 入力（説明変数:df，目的変数:df，プロジェクト名:str, モデルの名前:str）
# 出力（プロジェクトごとの予測結果（適合率，再現率，F1値，正解率）:df）
def predict(explanatory_variable, label, project_name: str, model_name: str):
    # モデルの初期化
    model = select_model(model_name)

    # コーディング規約IDをダミー変数化
    df_marge = pd.concat(
        [
            pd.get_dummies(explanatory_variable["Warning ID"]),
            explanatory_variable.drop(columns="Warning ID"),
        ],
        axis=1,
    )

    # 説明変数，目的変数を学習用，テスト用に分割
    X_train, X_test, Y_train, Y_test = train_test_split(
        df_marge, label, test_size=0.2, shuffle=False
    )
    
    # 一旦追加行------
    X_train, _, Y_train, _ = train_test_split(
        X_train, Y_train, test_size=0.9, shuffle=False
    )
    # print(X_train)
    # -----------
    
    Y_train = Y_train.values.ravel()
    Y_test = Y_test.values.ravel()

    # モデルの学習
    try:
        model.fit(X_train.drop(["Project_name"], axis=1), Y_train)
        filename = f"src/models/conventional/{model_name}/{project_name}_model.sav"
        # joblib.dump(model, filename)
    except ValueError as e:
        print(e)
        result = {
            "precision": "NaN",
            "recall": "NaN",
            "f1_score": "NaN",
            "accuracy": "NaN",
        }
        return pd.DataFrame([result], index=[project_name]), 0

    # モデルを使った学習
    # predict_result = model.predict(X_test.drop(['Project_name', 'Cluster_num'], axis=1))
    predict_result = model.predict(X_test.drop(["Project_name"], axis=1))

    # 分析用DF
    # print(Y_test)
    return_df = copy.deepcopy(X_test)
    return_df["real_TF"] = copy.deepcopy(Y_test)
    return_df["predict_TF"] = copy.deepcopy(predict_result)

    # 結果の格納
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

    return pd.DataFrame([result], index=[project_name]), return_df.reset_index(
        drop=True
    )


# すべてのデータを結合したモデルの作成
# 入力（クラスタ数:int, モデルの名前:str）
# 出力（モデル:model，規約違反ダミー:list）
def create_all_model(cnum, model_name: str):
    # for文を回すファイル名を取得
    with open("dataset/white_list.txt") as f:
        project_list = f.read().splitlines()

    train_df = pd.DataFrame()

    model_all = select_model(model_name)

    path = "dataset/outputs/"

    for project_name in project_list:
        df_value = pd.read_csv(f"{path}{project_name}_value.csv")
        df_label = pd.read_csv(f"{path}{project_name}_label.csv", header=None)

        # 説明変数，目的変数を学習用，テスト用に分割
        # X_train, _, Y_train, _ = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
        X_train, _, Y_train, _ = train_test_split(
            df_value, df_label, test_size=0.2, shuffle=False
        )
        Y_train = Y_train.values.ravel()
        X_train["AnsTF"] = copy.deepcopy(Y_train)
        train_df = pd.concat([train_df, X_train], axis=0)

        # コーディング規約IDをダミー変数化
    df_marge = pd.concat(
        [pd.get_dummies(train_df["Warning ID"]), train_df.drop(columns="Warning ID")],
        axis=1,
    )
    dummys = list(pd.get_dummies(train_df["Warning ID"]))

    # print(train_df.head())
    try:
        # model_all.fit(df_marge.drop(['Project_name', 'Cluster_num', 'AnsTF'], axis=1), df_marge["AnsTF"])
        model_all.fit(
            df_marge.drop(["Project_name", "AnsTF"], axis=1), df_marge["AnsTF"]
        )
        filename = f"src/models/merge/{model_name}.sav"
        joblib.dump(model_all, filename)
    except ValueError as e:
        print(e)

    return model_all, dummys


# すべてのデータを結合しクラスタリング後に，クラスタごとの予測モデルの作成
# 入力（クラスタ数:int）
# 出力（モデル:dict(key:project name, value:model))，規約違反ダミー:list）
def create_model(cnum: int, model_name: str):
    with open("dataset/white_list.txt") as f:
        project_list = f.read().splitlines()

    df_all = pd.DataFrame()

    model_dict = {}
    for i in range(cnum):
        model_dict[f"cluster_{i}"] = select_model(model_name)

    path = "dataset/outputs"

    for project_name in project_list:
        df_value = pd.read_csv(f"{path}/{project_name}_value.csv")
        df_label = pd.read_csv(f"{path}/{project_name}_label.csv", header=None)
        df_cluster = pd.read_csv(f"{path}/{project_name}_cluster.csv", header=None)

        # 説明変数，目的変数を学習用，テスト用に分割
        X_train, _, Y_train, _, Z_train, _ = train_test_split(
            df_value, df_label, df_cluster, test_size=0.2, shuffle=False
        )
        Y_train = Y_train.values.ravel()
        Z_train = Z_train.values.ravel()
        X_train["Cluster_num"] = copy.deepcopy(Z_train)
        X_train["AnsTF"] = copy.deepcopy(Y_train)
        df_all = pd.concat([df_all, X_train], axis=0)

        # コーディング規約IDをダミー変数化
    df_marge = pd.concat(
        [pd.get_dummies(df_all["Warning ID"]), df_all.drop(columns="Warning ID")],
        axis=1,
    )
    dummys = list(pd.get_dummies(df_all["Warning ID"]))

    # print(df_all.head())
    for i in range(cnum):

        try:
            model_dict[f"cluster_{i}"].fit(
                df_marge[df_marge["Cluster_num"] == i].drop(
                    ["Project_name", "Cluster_num", "AnsTF"], axis=1
                ),
                df_marge[df_marge["Cluster_num"] == i]["AnsTF"],
            )
            filename = (
                f"src/models/cross/{cnum}clusters/{model_name}/cluster{i}_model.sav"
            )
            joblib.dump(model_dict[f"cluster_{i}"], filename)
        except ValueError as e:
            # model_dict["cluster_"+str(i)] = df_marge[df_marge['Cluster_num'] == i]["AnsTF"][0]
            print(f"cluster_{i} {e}")

    return model_dict, dummys

def step_predict(explanatory_variable, label, project_name: str, model_name: str):
    # for文を回すファイル名を取得
    print(project_name)
    with open("dataset/white_list.txt") as f:
        project_list = f.read().splitlines()
    project_list.remove(project_name)
    id_dict = {}
    train_df = pd.DataFrame()
    returnDF = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])

    path = "dataset/outputs/"

    for p_name in project_list:
        X_train = pd.read_csv(f"{path}{p_name}_value.csv")
        Y_train = pd.read_csv(f"{path}{p_name}_label.csv", header=None)
        Y_train = Y_train.values.ravel()
        X_train["AnsTF"] = copy.deepcopy(Y_train)
        train_df = pd.concat([train_df, X_train], axis=0)

    # 引数で受け取ったデータを結合
    label = label.values.ravel()
    explanatory_variable["AnsTF"] = copy.deepcopy(label)
    train_all, test_data = train_test_split(
        explanatory_variable, test_size=0.2, shuffle=False
    )
    test_data = test_data.reset_index(drop=True)
    
    for i in range(1, 11):
        model_all = select_model(model_name)
        if i < 10:
            train_part, _ = train_test_split(
                train_all, test_size=((10-i)/10), shuffle=False
            )
        else:
            train_part = train_all
        use_train_df = pd.concat([train_df, train_part], axis=0)
        # コーディング規約IDをダミー変数化
        df_marge = pd.concat(
            [pd.get_dummies(use_train_df["Warning ID"]), use_train_df.drop(columns="Warning ID")],
            axis=1,
        )
        dummys = list(pd.get_dummies(use_train_df["Warning ID"]))
        # print(dummys)

        try:
            print(len(df_marge["AnsTF"].to_list()))
            model_all.fit(
                df_marge.drop(["Project_name", "AnsTF"], axis=1), df_marge["AnsTF"]
            )
            # filename = f"src/models/merge/{model_name}.sav"
            # joblib.dump(model_all, filename)
        except ValueError as e:
            print(e)
            
        id_dict.clear()
        for j in list(dummys):
            id_dict[j] = []
        for wid in test_data["Warning ID"]:
            if wid in id_dict:
                id_dict[wid].append(1)
                for j in id_dict:
                    id_dict[j].append(0)
                id_dict[wid].pop(-1)
            else:
                for j in id_dict:
                    id_dict[j].append(0)
        
        id_df = pd.DataFrame(id_dict)
        test_data_2nd = pd.concat([id_df, test_data], axis=1)
        # print(len(list(df_marge.columns)))
        # print(len(list(test_data_2nd.columns))-1)
        # print(list(test_data.columns))
        
        predict_result = model_all.predict(
            test_data_2nd.drop(["Warning ID", "Project_name", "AnsTF"], axis=1)
        )
        result = {
            "precision": format(
                precision_score(test_data_2nd["AnsTF"].to_list(), predict_result, zero_division=np.nan), ".2f"
            )
        }
        result["recall"] = format(
            recall_score(test_data_2nd["AnsTF"].to_list(), predict_result, zero_division=np.nan), ".2f"
        )
        result["f1_score"] = format(
            f1_score(test_data_2nd["AnsTF"].to_list(), predict_result, zero_division=np.nan), ".2f"
        )
        result["accuracy"] = format(accuracy_score(test_data_2nd["AnsTF"].to_list(), predict_result), ".2f")
        tmp = pd.DataFrame([result], index=[f"{project_name}@{i*10}%"])
        returnDF = pd.concat([returnDF, tmp], axis=0)
        # print(predict_result)

    return returnDF


def select_model(model_name: str):
    match model_name:
        case "Logistic":
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

            return model

        case "RandomForest":
            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=0,
            )
            return model

        case "SVM":
            model = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)
            return model
        
        case "DecisionTree":
            model = DecisionTreeClassifier(
                criterion="gini",  # または "entropy"
                splitter="best",  # または "random"
                max_depth=None,   # 最大深さ
                random_state=0,   # 再現性確保のため
            )
            return model

        case _:
            print("It is out of pattern")
            return "error"
