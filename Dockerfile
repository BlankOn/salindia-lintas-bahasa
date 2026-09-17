# Salindia in a container.
#
# The image carries no models: MLX is Apple-Silicon only, so a deployed
# instance runs the OpenAI engine (FORCE_APPROACH=openai and a key). The MLX
# imports are lazy and the markers in requirements.txt skip those wheels off
# macOS, so nothing here pulls a GPU stack it cannot use.
FROM python:3.12-slim

# Bytecode on stdout as it happens, and no .pyc litter in the layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: they change far less often than the app does, so this
# layer survives most rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY web/ ./web/
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Talk records live on a volume, never in the image: a container restart or a
# new tag must not lose them. The directory (not just the file) is mounted,
# because SQLite writes salindia.sqlite3-wal and -shm next to it.
ENV DB_PATH=/data/salindia.sqlite3 \
    HOST=0.0.0.0 \
    PORT=8000
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin salindia \
    && mkdir -p /data && chown salindia:salindia /data
VOLUME /data

EXPOSE 8000

# /healthz answers before the models are ready, and says which engine is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/healthz\", timeout=4).status == 200 else 1)"

# The entrypoint starts as root only to put the mounted /data in reach of the
# app's uid, then drops to it with setpriv -- uvicorn itself never runs as root.
# Pass --user to docker run (or `user:` in compose) to skip that entirely.
ENTRYPOINT ["/docker-entrypoint.sh"]

# run.sh is for a laptop (it builds a venv); in here uvicorn is the command.
# One worker, deliberately: sessions, the engine and the talk rows are process
# state, so a second worker would serve pages that disagree about all three.
CMD ["sh", "-c", "exec python -m uvicorn server.main:app --host \"$HOST\" --port \"$PORT\" --proxy-headers --forwarded-allow-ips '*'"]
