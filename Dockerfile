# ベースイメージ（Python 3.10）
FROM python:3.10-slim

# 作業ディレクトリ
WORKDIR /app

# 必要なパッケージをインストール
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN pip install --upgrade autogen

# アプリのコードをコピー
COPY . .

# ポート公開（GradioなどのUIが動く場合）
EXPOSE 7860

ENTRYPOINT ["gradio", "ui.py"]