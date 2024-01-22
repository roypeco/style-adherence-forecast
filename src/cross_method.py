import pandas as pd
import numpy as np
import copy
from modules import machine_learning_models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 宣言
cnum = 10
counter = 1
path = "dataset/outputs"
model_name = "SVM" # Logistic, RandomForest, SVMの３種類から選ぶ
id_dict = {}
bunseki_df = pd.DataFrame()
model_dict, dummys = machine_learning_models.create_model(cnum, model_name)
result_df = pd.DataFrame(columns=['precision', 'recall', 'f1_score', 'accuracy'])
for i in list(dummys):
  id_dict[i] = []

with open("dataset/white_list.txt") as f:
  project_list = f.read().splitlines()
    
for project_name in project_list:
  df_value = pd.read_csv(f'{path}/{project_name}_value.csv')
  df_label = pd.read_csv(f'{path}/{project_name}_label.csv', header=None)
  df_cluster = pd.read_csv(f'{path}/{project_name}_cluster.csv', header=None)
  X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(df_value, df_label, df_cluster, test_size=0.2, shuffle=False)

  # Y_train = Y_train.values.ravel()
  # X_train["AnsTF"] = Y_train
  # X_train = X_train.reset_index(drop=True)
  
  Y_test = Y_test.values.ravel()
  Z_test = Z_test.values.ravel()
  X_test["Cluster_num"] = copy.deepcopy(Z_test)
  X_test["AnsTF"] = copy.deepcopy(Y_test)
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
  
  predict_result = []
  ans_list = []
  cluster_list = []
  for i in range(cnum):
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
  
  # tmp = pd.DataFrame({'Cluster_num': cluster_list, 'real_TF':ans_list, 'predict_TF':predict_result})
  # bunseki_df = pd.concat([bunseki_df, count_cluster(cnum, project_name, tmp)], axis=0)
  
  # print(predict_result)
  
  result = {'precision': format(precision_score(ans_list, predict_result, zero_division=np.nan), '.2f'),
            'recall': format(recall_score(ans_list, predict_result, zero_division=np.nan), '.2f'),
            'f1_score': format(f1_score(ans_list, predict_result, zero_division=np.nan), '.2f'),
            'accuracy': format(accuracy_score(ans_list, predict_result), '.2f')
          }
  # print(result)
  result_df = pd.concat([result_df, pd.DataFrame([result], index=[project_name])], axis=0)
  
  print(project_name, f"{counter} / {len(project_list)}")
  counter += 1

result_df.to_csv(f"results/cross/{model_name}.csv")
print(result_df)