import warnings
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import (  # 評価指標算出用
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from modules import machine_learning_models
warnings.filterwarnings("always", category=UserWarning)

model_name = "DecisionTree"  # Logistic, RandomForest, DecisionTreeの３種類から選ぶ
result_df = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])

with open("dataset/white_list.txt") as f:
    project_list = f.read().splitlines()

for project_name in project_list:
    print(project_name)
    try:
        df_value = pd.read_csv(f"./dataset/outputs/{project_name}_value.csv")
        df_label = pd.read_csv(f"./dataset/outputs/{project_name}_label.csv", header=None)
    except pd.errors.EmptyDataError as e:
        print(project_name, e)
    
    df_marge = pd.concat(
        [
            pd.get_dummies(df_value["Warning ID"]),
            df_value.drop(columns="Warning ID"),
        ],
        axis=1,
    )
        
    X_train, X_test, Y_train, Y_test = train_test_split(
        df_marge, df_label, test_size=0.2, shuffle=False
    )
    
    for i in range(1, 11):
        model = machine_learning_models.select_model(model_name)
        if i < 10:
            train_part, _ , ytrain_part, _= train_test_split(
                X_train, Y_train, test_size=((10-i)/10), shuffle=False
            )
        else:
            train_part, ytrain_part = X_train, Y_train
            
        ytrain_part = ytrain_part.values.ravel()
        # Y_test = Y_test.values.ravel()
        try:
            model.fit(train_part.drop(["Project_name"], axis=1), ytrain_part)
        except ValueError as e:
            print(e)
            
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

        # print(result)
        result_df = pd.concat([
            result_df,
            pd.DataFrame([result], index=[f"{project_name}@{i*10}%"]),
        ], axis=0)
    print(f"{round(sum(Y_test[0])/len(Y_test), 3)} {sum(Y_test[0])} {len(Y_test)}")
        
    result_df.to_csv(f"results/soloStep{model_name}.csv")

