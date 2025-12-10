FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# uv をインストール（pip最小限）
RUN pip install --upgrade pip && pip install uv

WORKDIR /workspace
ENV PYTHONPATH=/workspace/src

# pyproject から依存インストール
COPY pyproject.toml ./
RUN uv sync

# ソースコード
COPY . .

CMD ["uv", "run", "streamlit", "run", "src/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
