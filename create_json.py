import pandas as pd
import json

result_dict = {}

with open("dataset/white_list.txt") as f:
  project_list = f.read().splitlines()

for file_name in project_list[0:61]:
  fix_dict = {"w":[0, 0],
              "c":[0, 0],
              "e":[0, 0],
              "r":[0, 0],
              "i":[0, 0],
              "f":[0, 0]}
  
  df_label = pd.read_csv(f'./dataset/outputs/{file_name}_label.csv', header=None)
  df_value = pd.read_csv(f'./dataset/outputs/{file_name}_value.csv')
  df_wid = df_value["Warning ID"]
  for wid, label in zip(df_wid.to_list(), df_label[0]):
    if "W" in wid:
      fix_dict["w"][0] += label
      fix_dict["w"][1] += 1
    elif "C" in wid:
      fix_dict["c"][0] += label
      fix_dict["c"][1] += 1
    elif "E" in wid:
      fix_dict["e"][0] += label
      fix_dict["e"][1] += 1
    elif "R" in wid:
      fix_dict["r"][0] += label
      fix_dict["r"][1] += 1
    elif "I" in wid:
      fix_dict["i"][0] += label
      fix_dict["i"][1] += 1
    elif "F" in wid:
      fix_dict["f"][0] += label
      fix_dict["f"][1] += 1
    
  result_dict[f"{file_name}"] = fix_dict

# print(json.dumps(result_dict, indent=4))
with open('dataset/test_output.json', 'w') as f:
    json.dump(result_dict, f, indent=2)