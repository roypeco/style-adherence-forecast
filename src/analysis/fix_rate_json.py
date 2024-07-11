import pandas as pd
import json

result_dict = {}

with open("dataset/white_list.txt") as f:
  project_list = f.read().splitlines()

for file_name in project_list:
  fix_dict = {}
  rate_dict = {}
  df_label = pd.read_csv(f'./dataset/outputs/{file_name}_label.csv', header=None)
  df_value = pd.read_csv(f'./dataset/outputs/{file_name}_value.csv')
  df_wid = df_value["Warning ID"]
  for wid, label in zip(df_wid.to_list(), df_label[0]):
    if wid in fix_dict:
      fix_dict[wid][0] += label
      fix_dict[wid][1] += 1
    else:
      fix_dict[wid] = [label, 1]
    
  for key in fix_dict:
      rate_dict[key] = '{:.3f}'.format(
          fix_dict[key][0] / fix_dict[key][1]
      )
  
  result_dict[file_name] = rate_dict
  
print(json.dumps(result_dict, indent=4))
with open('dataset/fix_rate.json', 'w') as f:
    json.dump(result_dict, f, indent=2)
