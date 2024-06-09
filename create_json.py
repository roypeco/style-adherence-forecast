import pandas as pd
import json

result_dict = {}
# all_id = {}
black_list = [
  'C0209',
  'W1514',
  'E0401',
  'W0611',
  'R1705',
  'R1732',
  'W0707',
]

with open("dataset/white_list.txt") as f:
  project_list = f.read().splitlines()

for file_name in project_list[0:61]:
  fix_dict = {"w":[0, 0],
              "c":[0, 0],
              "e":[0, 0],
              "r":[0, 0],
              }
  
  df_label = pd.read_csv(f'./dataset/outputs/{file_name}_label.csv', header=None)
  df_value = pd.read_csv(f'./dataset/outputs/{file_name}_value.csv')
  df_wid = df_value["Warning ID"]
  for wid, label in zip(df_wid.to_list(), df_label[0]):
    if "W" in wid and not wid in black_list:
      fix_dict["w"][0] += label
      fix_dict["w"][1] += 1
    elif "C" in wid and not wid in black_list:
      fix_dict["c"][0] += label
      fix_dict["c"][1] += 1
    elif "E" in wid and not wid in black_list:
      fix_dict["e"][0] += label
      fix_dict["e"][1] += 1
    elif "R" in wid and not wid in black_list:
      fix_dict["r"][0] += label
      fix_dict["r"][1] += 1
    # 類似度の差分を出すために消す
    # elif "I" in wid:
    #   fix_dict["i"][0] += label
    #   fix_dict["i"][1] += 1
    # elif "F" in wid:
    #   fix_dict["f"][0] += label
    #   fix_dict["f"][1] += 1
    
    # if wid in all_id:
    #   if all_id[wid][2]:
    #     all_id[wid][0] += 1
    #     all_id[wid][2] = False
    # else:
    #   all_id[wid] = [1, 62, False]
    
  result_dict[f"{file_name}"] = fix_dict
  
  # for flg in all_id:
  #   all_id[flg][2] = True

# print(json.dumps(result_dict, indent=4))
# with open('dataset/test_output.json', 'w') as f:
#     json.dump(result_dict, f, indent=2)

# sorted_dict = sorted(all_id.items(), key=lambda x:x[1][0], reverse=True)
# print(sorted_dict)