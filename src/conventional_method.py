import os
import pandas as pd
from modules import machine_learning_models

# 宣言
model_name = "SVM" # Logistic, RandomForest, SVMの３種類から選ぶ

# for文を回すファイル名を取得
# dir_path = "dataset/row_data"

# # dataset内のプロジェクト名一覧取得
# project_list = [
#     f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
# ]

project_list = ["GPflow", "hickle", "python-sdk"]


#結果格納用のDFの宣言
result_df = pd.DataFrame(columns=['precision', 'recall', 'f1_score', 'accuracy'])
bunseki_df = pd.DataFrame()

# 従来手法の実行:machine_learning_models.predict
for file_name in project_list:
  try:
    df_value = pd.read_csv(f'./dataset/some_of/{file_name}_value.csv')
    df_label = pd.read_csv(f'./dataset/some_of/{file_name}_label.csv', header=None)
  except pd.errors.EmptyDataError as e:
    print(file_name)
  tmp1, _ = machine_learning_models.predict(df_value, df_label, file_name, model_name)
  result_df = pd.concat([result_df, tmp1], axis=0)

result_df.to_csv(f"results/nogu/solo/{model_name}.csv")
print(result_df)