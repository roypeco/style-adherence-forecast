import os
import pandas as pd
from modules import machine_learning_models, test_SelectModel

# 宣言
model_name = "Logistic" # Logistic, RandomForest, SVMの３種類から選ぶ

# for文を回すファイル名を取得
# files = os.listdir('./sample_dataset')
# files_dir = [f for f in files if os.path.isdir(os.path.join('./sample_dataset', f))]
files_dir = ['python-bugzilla', 'howdoi', 'python-cloudant', 'hickle', 'pyscard',
            'transitions', 'pynput', 'OWSLib', 'schema_salad', 'schematics']

#結果格納用のDFの宣言
result_df = pd.DataFrame(columns=['precision', 'recall', 'f1_score', 'accuracy'])
bunseki_df = pd.DataFrame()

# 従来手法の実行:machine_learning_models.predict
for file_name in files_dir:
  df_value = pd.read_csv(f'./dataset/createData_10/{file_name}_train.csv')
  df_label = pd.read_csv(f'./dataset/createData_10/{file_name}_label.csv', header=None)
  tmp1, tmp2 = machine_learning_models.predict(df_value, df_label, file_name, model_name)
  result_df = pd.concat([result_df, tmp1], axis=0)
  
print(result_df)