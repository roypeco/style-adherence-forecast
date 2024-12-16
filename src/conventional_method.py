import os

import pandas as pd
from modules import machine_learning_models

# 宣言
model_name = "RandomForest"  # Logistic, RandomForest, DecisionTreeの３種類から選ぶ
counter = 1

# for文を回すファイル名を取得
with open("dataset/white_list.txt") as f:
    project_list = f.read().splitlines()

# project_list = ["pywal"], "jenkinsapi", "analytics-python", "edx-search", "python-resize-image",
                #"pyhomematic", "bidict", "azure-activedirectory-library-for-python", "django-sortedm2m", "edx-drf-extensions"]
project_list = ["tldextract", "edx-search", "easyquotation", "django-rest-swagger", "flaky",
    "implicit", "python-sshpubkeys", "munch", "python-resize-image"]


# 結果格納用のDFの宣言
result_df = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])
bunseki_df = pd.DataFrame()

# 従来手法の実行:machine_learning_models.predict
for file_name in project_list:
    try:
        df_value = pd.read_csv(f"./dataset/outputs/{file_name}_value.csv")
        df_label = pd.read_csv(f"./dataset/outputs/{file_name}_label.csv", header=None)
    except pd.errors.EmptyDataError as e:
        print(file_name, e)
    tmp1, _ = machine_learning_models.predict(df_value, df_label, file_name, model_name)
    result_df = pd.concat([result_df, tmp1], axis=0)

result_df.to_csv(f"results/anyPercent.csv")
print(result_df)
