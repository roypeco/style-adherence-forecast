import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score # 評価指標算出用

# 従来手法の予測結果の算出
# 入力（説明変数:df，目的変数:df，プロジェクト名:str, モデルの名前:str）
# 出力（プロジェクトごとの予測結果（適合率，再現率，F1値，正解率）:df）
def predict(explanatory_variable, label, project_name: str, model_name: str):
  # モデルの初期化
  model = select_model(model_name)
  
  # コーディング規約IDをダミー変数化
  df_marge = pd.concat([pd.get_dummies(explanatory_variable['Warning ID']), explanatory_variable.drop(columns='Warning ID')], axis=1)
  
  # 説明変数，目的変数を学習用，テスト用に分割
  X_train,X_test,Y_train,Y_test = train_test_split(df_marge, label, test_size=0.2, shuffle=False)
  Y_train = Y_train.values.ravel()
  Y_test  = Y_test.values.ravel()
  
  # モデルの学習
  model.fit(X_train.drop(['Project_name', 'Cluster_num'], axis=1), Y_train)

  # モデルを使った学習
  predict_result = model.predict(X_test.drop(['Project_name', 'Cluster_num'], axis=1))
  
  # 分析用DF
  return_df = X_test
  return_df["real_TF"] = Y_test
  return_df["predict_TF"] = predict_result
  
  # 結果の格納
  result = {'precision': format(precision_score(Y_test, predict_result), '.2f'), 'recall': format(recall_score(Y_test, predict_result), '.2f'),
            'f1_score': format(f1_score(Y_test, predict_result), '.2f'), 'accuracy': format(accuracy_score(Y_test, predict_result), '.2f')
           }
  
  return pd.DataFrame([result], index=[project_name]), return_df.reset_index(drop=True)


# すべてのデータを結合したモデルの作成
# 入力（クラスタ数:int, モデルの名前:str）
# 出力（モデル:model，規約違反ダミー:list）
def create_all_model(cnum, model_name: str):
  # files = os.listdir('./sample_dataset')
  # files_dir = [f for f in files if os.path.isdir(os.path.join('./sample_dataset', f))]
  files_dir = ['python-bugzilla', 'howdoi', 'python-cloudant', 'hickle', 'pyscard',
            'transitions', 'pynput', 'OWSLib', 'schema_salad', 'schematics']
  
  train_df = pd.DataFrame()
  
  model_all = select_model(model_name)
  
  if cnum == 5:
    path = "./dataset/createData_05/"
  else:
    path = f"./dataset/createData_{cnum}/"
    
  for file_name in files_dir:
    df_value = pd.read_csv(f'{path}{file_name}_train.csv')
    df_label = pd.read_csv(f'{path}{file_name}_label.csv', header=None)

    # 説明変数，目的変数を学習用，テスト用に分割
    # X_train, _, Y_train, _ = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
    X_train, _, Y_train, _ = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
    Y_train = Y_train.values.ravel()
    X_train["AnsTF"] = Y_train.copy()
    train_df = pd.concat([train_df, X_train], axis=0)
    
    # コーディング規約IDをダミー変数化
  df_marge = pd.concat([pd.get_dummies(train_df['Warning ID']), train_df.drop(columns='Warning ID')], axis=1)
  dummys = list(pd.get_dummies(train_df['Warning ID']))
  
  # print(train_df.head())
  try:
    model_all.fit(df_marge.drop(['Project_name', 'Cluster_num', 'AnsTF'], axis=1), df_marge["AnsTF"])
  except ValueError as e:
    print(e)
    
  return model_all, dummys


# すべてのデータを結合しクラスタリング後に，クラスタごとの予測モデルの作成
# 入力（クラスタ数:int）
# 出力（モデル:dict(key:project name, value:model))，規約違反ダミー:list）  
def create_model(cnum: int, model_name: str):
  # files = os.listdir('./sample_dataset')
  # files_dir = [f for f in files if os.path.isdir(os.path.join('./sample_dataset', f))]
  files_dir = ['python-bugzilla', 'howdoi', 'python-cloudant', 'hickle', 'pyscard',
            'transitions', 'pynput', 'OWSLib', 'schema_salad', 'schematics']
  
  train_df = pd.DataFrame()
  
  model_dict = {}
  for i in range(cnum):
    model_dict["cluster_"+str(i)] = select_model(model_name)
  
  if cnum == 5:
    path = "./dataset/createData_05/"
  else:
    path = f"./dataset/createData_{cnum}/"
    
  for file_name in files_dir:
    df_value = pd.read_csv(f'{path}{file_name}_train.csv')
    df_label = pd.read_csv(f'{path}{file_name}_label.csv', header=None)

    # 説明変数，目的変数を学習用，テスト用に分割
    X_train, _, Y_train, _ = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
    Y_train = Y_train.values.ravel()
    X_train["AnsTF"] = Y_train
    train_df = pd.concat([train_df, X_train], axis=0)
    
    # コーディング規約IDをダミー変数化
  df_marge = pd.concat([pd.get_dummies(train_df['Warning ID']), train_df.drop(columns='Warning ID')], axis=1)
  dummys = list(pd.get_dummies(train_df['Warning ID']))
  
  # print(train_df.head())
  for i in range(cnum):
    try:
      model_dict["cluster_"+str(i)].fit(df_marge[df_marge['Cluster_num'] == i].drop(['Project_name', 'Cluster_num', 'AnsTF'], axis=1),
                                        df_marge[df_marge['Cluster_num'] == i]["AnsTF"])
    except ValueError:
      # model_dict["cluster_"+str(i)] = df_marge[df_marge['Cluster_num'] == i]["AnsTF"][0]
      pass
    
  return model_dict, dummys
  
  
def select_model(model_name: str):
  match model_name:
    case "Logistic":
      model = LogisticRegression(penalty='l2',          # 正則化項(L1正則化 or L2正則化が選択可能)
                            class_weight='balanced',  # クラスに付与された重み
                            random_state=0,     # 乱数シード
                            solver='lbfgs',        # ハイパーパラメータ探索アルゴリズム
                            max_iter=10000,          # 最大イテレーション数
                            multi_class='auto',    # クラスラベルの分類問題（2値問題の場合'auto'を指定）
                            warm_start=False,      # Trueの場合、モデル学習の初期化に前の呼出情報を利用
                            n_jobs=None,           # 学習時に並列して動かすスレッドの数
                          )
      return model
    
    case "RandomForest":
      model = RandomForestClassifier(class_weight='balanced',
                                     random_state=0,
                                     )
      return model
    
    case "SVM":
      model = SVC(kernel='linear',
                  C=1.0, 
                  random_state=0
                  )
      return model
    
    case _:
      print("It is out of pattern")
      return "error"
