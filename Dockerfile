# ベースイメージを指定
FROM python:3.10.13

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN pip install --upgrade pip

# 依存パッケージをインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY . /app