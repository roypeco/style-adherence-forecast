import pandas as pd
from modules import machine_learning_models

# 宣言
model_name = "RandomForest"  # Logistic, RandomForest, SVMの３種類から選ぶ
result_df = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])

# for文を回すファイル名を取得
with open("dataset/white_list.txt") as f:
    project_list = f.read().splitlines()

for project_name in project_list:
    try:
        df_value = pd.read_csv(f"./dataset/outputs/{project_name}_value.csv")
        df_label = pd.read_csv(f"./dataset/outputs/{project_name}_label.csv", header=None)
    except pd.errors.EmptyDataError as e:
        print(project_name, e)
    
    machine_learning_models.step_predict(df_value, df_label, project_name, model_name)
    # result_df = pd.concat([result_df, tmp1], axis=0)
        
    # result_df.to_csv(f"results/stepBystepResult.csv")
    # print(result_df)