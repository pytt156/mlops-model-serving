FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY streamlit_app.py ./

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
