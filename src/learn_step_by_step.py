import pandas as pd
from modules import machine_learning_models

# 宣言
model_name = "Logistic"  # Logistic, RandomForest, SVMの３種類から選ぶ
result_df = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])
counter = 1

# for文を回すファイル名を取得
with open("dataset/white_list.txt") as f:
    project_list = f.read().splitlines()
project_list = ["pywal", "jenkinsapi", "analytics-python", "edx-search", "python-resize-image",
                "pyhomematic", "bidict", "azure-activedirectory-library-for-python", "django-sortedm2m", "edx-drf-extensions"]
counter_end = len(project_list)

for project_name in project_list:
    try:
        df_value = pd.read_csv(f"./dataset/outputs/{project_name}_value.csv")
        df_label = pd.read_csv(f"./dataset/outputs/{project_name}_label.csv", header=None)
    except pd.errors.EmptyDataError as e:
        print(project_name, e)
    
    tmp = machine_learning_models.step_predict(df_value, df_label, project_name, model_name)
    result_df = pd.concat([result_df, tmp], axis=0)
        
    result_df.to_csv(f"results/stepBystepResult.csv")
    # print(result_df)
    # with open('./progress.txt', 'w') as f:
    #     f.write(f"{counter}/{counter_end}")
    counter += 1