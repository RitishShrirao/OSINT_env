FROM python:3.12-slim

RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user pyproject.toml README.md openenv.yaml inference.py $HOME/app/
COPY --chown=user src $HOME/app/src
COPY --chown=user config $HOME/app/config
COPY --chown=user datasets $HOME/app/datasets
COPY --chown=user docs $HOME/app/docs
COPY --chown=user scripts $HOME/app/scripts
COPY --chown=user server.py $HOME/app/server.py

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
