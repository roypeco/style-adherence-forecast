from modules import model_loader
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split

model_name = "Logistic"  # Logistic, RandomForest, SVM の３種類から選ぶ
dummys = model_loader.get_dummy()
model_all = model_loader.load_model("merge", model_name)
model_dict = model_loader.load_model("cross", model_name)
id_dict = {}

with open("dataset/white_list.txt") as f:
  project_list = f.read().splitlines()

for project_name in project_list:
  # プロジェクトごとのデータの読み込み
  df_value = pd.read_csv(f'./dataset/outputs/{project_name}_value.csv')
  df_label = pd.read_csv(f'./dataset/outputs/{project_name}_label.csv', header=None)
  df_cluster = pd.read_csv(f'./dataset/outputs/{project_name}_cluster.csv', header=None)
  df_merge = pd.concat([pd.get_dummies(df_value['Warning ID']), df_value.drop(columns='Warning ID')], axis=1)
  _,X_test,_,Y_test,_,Z_test = train_test_split(df_merge, df_label, df_cluster, test_size=0.2, shuffle=False)
  _,not_dummy = train_test_split(df_value, test_size=0.2, shuffle=False)
  Y_test  = Y_test.values.ravel()
  Z_test = Z_test.values.ravel()
  not_dummy["AnsTF"] = copy.deepcopy(Y_test)
  not_dummy["Cluster_num"] = copy.deepcopy(Z_test)
  predict_result = []
  ans_list = []
  cluster_list = []
  
  # 従来手法
  model = model_loader.load_model("conventional", model_name, project_name=project_name)
  print(model.predict(X_test.drop(['Project_name'], axis=1)))
  
# 提案手法1
  id_dict.clear()
  for i in list(dummys):
    id_dict[i] = []
  for wid in not_dummy["Warning ID"]:
    if wid in id_dict:
      id_dict[wid].append(1)
      for i in id_dict:
        id_dict[i].append(0)
      id_dict[wid].pop(-1)
    else:
      for i in id_dict:
        id_dict[i].append(0)
  
  id_df = pd.DataFrame(id_dict)
  not_dummy = not_dummy.reset_index(drop=True)
  test_df = pd.concat([id_df, not_dummy], axis=1)
  print(model_all.predict(test_df.drop(['Warning ID', 'Project_name', 'Cluster_num', 'AnsTF'], axis=1)))
  
  # 提案手法2
  for i in range(10):
    try:
      if len(list(test_df[test_df['Cluster_num'] == i]["AnsTF"])) != 0:
        match model_name:
          case "Logistic":
            if model_dict[f"cluster_{i}"].__getattribute__('coef_') is not None:
              predict_result.extend(model_dict[f"cluster_{i}"].predict(test_df[test_df['Cluster_num'] == i].drop(['Warning ID', 'Project_name', 'Cluster_num', "AnsTF"], axis=1)))
              ans_list.extend(list(test_df[test_df['Cluster_num'] == i]["AnsTF"]))
              for j in range(len(list(test_df[test_df['Cluster_num'] == i]["AnsTF"]))):
                cluster_list.append(i)

          case "RandomForest" | "SVM":
            predict_result.extend(model_dict[f"cluster_{i}"].predict(test_df[test_df['Cluster_num'] == i].drop(['Warning ID', 'Project_name', 'Cluster_num', "AnsTF"], axis=1)))
            ans_list.extend(list(test_df[test_df['Cluster_num'] == i]["AnsTF"]))
            for j in range(len(list(test_df[test_df['Cluster_num'] == i]["AnsTF"]))):
              cluster_list.append(i)

    except (AttributeError, KeyError) as e:
      print(f"{project_name}: skip cluster_{i} {e}")
  print(np.array(predict_result))
  
  print(Y_test)