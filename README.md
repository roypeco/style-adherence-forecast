# 研究コードリポジトリ

## ディレクトリ概要

### dataset_creation(データセット作成用コード)

- createdataset.py(生のデータから帰化学習モデルに入力できるように整形, 実行に5分程度必要)
- clusyering.py(提案手法2のために整形したデータからクラスタリング, データサイズが大きい場合はサーバでないと実行不可)
- modules/get_cluster(clustering.pyで利用しているモジュール)
- modules/lable_sep(createdatabase.pyで利用しているモジュール)

### src(修正予測用コード)

- conventional_method.py(従来手法再現モデル)
- cross_method.py(提案手法のクラスタリング利用モデル)
- merge_method.py(提案手法の全て結合モデル)
- results_analysis.py(予測結果分析用コード)

### etc(その他)

- Dockerfile, docker-comose.yml(サーバで動かすようのDocker関連ファイル)
- Pipfile, Pipfile.lock(ライブラリ管理pipenv関連ファイル)
- ruff.toml(SAT設定ファイル)
