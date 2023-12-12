import pandas as pd
import os
from modules import machine_learning_models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 宣言
id_dict = {}
model_name = "Logistic" # Logistic, RandomForest, SVMの３種類から選ぶ
bunseki_df = pd.DataFrame()
model_all, dummys = machine_learning_models.create_all_model(10, model_name)
result_df = pd.DataFrame(columns=['precision', 'recall', 'f1_score', 'accuracy'])
for i in list(dummys):
  id_dict[i] = []

# for文を回すファイル名を取得
dir_path = "dataset/row_data"

# dataset内のプロジェクト名一覧取得
project_list = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]

path = "dataset/outputs/"
    
for project_name in project_list:
  df_value = pd.read_csv(f'{path}{project_name}_value.csv')
  df_label = pd.read_csv(f'{path}{project_name}_label.csv', header=None)
  _, X_test, _, Y_test = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
  Y_test = Y_test.values.ravel()
  X_test["AnsTF"] = Y_test
  X_test = X_test.reset_index(drop=True)
  
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
  # print(test_df)
  
  # predict_result = model_all.predict(test_df.drop(['Warning ID', 'Project_name', 'Cluster_num', "AnsTF"], axis=1))
  predict_result = model_all.predict(test_df.drop(['Warning ID', 'Project_name', "AnsTF"], axis=1))
  
  # tmp = pd.DataFrame({'Cluster_num':list(test_df['Cluster_num']), 'real_TF':Y_test, 'predict_TF':predict_result})
  
  # print(predict_result)
    
  result = {'precision': format(precision_score(Y_test, predict_result), '.2f'), 'recall': format(recall_score(Y_test, predict_result), '.2f'),
            'f1_score': format(f1_score(Y_test, predict_result), '.2f'), 'accuracy': format(accuracy_score(Y_test, predict_result), '.2f')
          }
  # print(result)
  result_df = pd.concat([result_df, pd.DataFrame([result], index=[project_name])], axis=0)

result_df.to_csv("results/merge/out.csv")
print(result_df)