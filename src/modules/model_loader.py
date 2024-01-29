import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
path = "src/models"

def load_mdoel(method_name: str, model_name: str, **kwargs):
  if method_name == "conventional":
    for _, value in kwargs.items():
      model = joblib.load(f"{path}/{method_name}/{model_name}/{value}_model.sav")
    return model
  
  elif method_name == "merge":
    model = joblib.load(f"{path}/{method_name}/{model_name}.sav")
    return model
  
  else:
    model_dict = {}
    for i in range(10):
      try:
        model_dict[f"cluster_{i}"] = joblib.load(f"{path}/{method_name}/10clusters/{model_name}/cluster{i}_model.sav")
      except FileNotFoundError as e:
        print(e)
    return model_dict
  
def get_dummy():
  with open("dataset/white_list.txt") as f:
   project_list = f.read().splitlines()
  
  for project_name in project_list:
    df_value = pd.read_csv(f'{path}{project_name}_value.csv')

    X_train, _ = train_test_split(df_value, test_size=0.2, shuffle=False)
    train_df = pd.concat([train_df, X_train], axis=0)
    
    # コーディング規約IDをダミー変数化
  dummys = list(pd.get_dummies(train_df['Warning ID']))
  
  return dummys