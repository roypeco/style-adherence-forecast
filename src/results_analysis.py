from modules import model_loader
import pandas as pd
from sklearn.model_selection import train_test_split

with open("dataset/white_list.txt") as f:
  project_list = f.read().splitlines()

for project_name in project_list:
  # 従来手法
  df_value = pd.read_csv(f'./dataset/outputs/{project_name}_value.csv')
  df_label = pd.read_csv(f'./dataset/outputs/{project_name}_label.csv', header=None)
  df_merge = pd.concat([pd.get_dummies(df_value['Warning ID']), df_value.drop(columns='Warning ID')], axis=1)
  _,X_test,_,Y_test = train_test_split(df_merge, df_label, test_size=0.2, shuffle=False)
  Y_test  = Y_test.values.ravel()
  model = model_loader.load_mdoel("conventional", "Logistic", project_name=project_name) # Logistic, RandomForest, SVM の３種類から選ぶ
  print(model.predict(X_test.drop(['Project_name'], axis=1)))