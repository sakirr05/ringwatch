# RingWatch demo service.
#
# WHAT THIS CONTAINER CAN AND CANNOT DO
# --------------------------------------
# It renders committed results, receives Razorpay webhooks, and runs one clearly-labelled
# demonstration scorer. It CANNOT retrain, rescore, or recompute any published metric --
# .dockerignore keeps `data/` out, so the 683 MB dataset and the model cache are not in the
# image. The architectural claim is enforced by absence, not by policy.
#
# Python 3.12 to match render.yaml's PYTHON_VERSION. The code is 3.10+ compatible and the
# test suite runs on 3.14 locally; pinning the container to the deployment target means the
# image is a rehearsal of production rather than a third environment.
#
# Built and run as part of this phase, not shipped untested -- see tests/test_docker.py,
# which is skipped when no Docker daemon is available so CI without Docker stays green.

FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE: the filesystem is ephemeral, .pyc files are pure waste.
# PYTHONUNBUFFERED: without it, logs vanish when a container is killed mid-buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    RINGWATCH_DB=/tmp/ringwatch_events.db

# libgomp1 is LightGBM's OpenMP runtime. Without it `import lightgbm` fails at load with an
# error that names a .so rather than the package, which is a genuinely confusing 20 minutes
# if you meet it for the first time in a deploy log.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies before source, so an edit to a template does not reinstall LightGBM.
COPY requirements-web.txt ./
RUN pip install -r requirements-web.txt

# Only what the service actually runs. `core/` is here for the lazy imports the webhook and
# scoring routes make; `ai/` because core.clusters imports ai.contract.
COPY app/ ./app/
COPY core/ ./core/
COPY ai/ ./ai/
COPY docs/ ./docs/
COPY artifacts/ ./artifacts/

# Non-root. The service writes only to /tmp (the ephemeral SQLite log), so it needs nothing
# else writable, and /srv can stay owned by root.
RUN useradd --system --create-home --uid 10001 ringwatch
USER ringwatch

EXPOSE 8000

# Liveness, matching render.yaml's healthCheckPath. /health touches no disk and answers 200
# whenever the process is alive; readiness -- which does depend on the results artifact --
# is /ready, deliberately outside the restart path.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# A shell is needed to expand ${PORT}, so platforms that inject a port (Render, Cloud Run)
# work unchanged and `docker run -p 8000:8000` still works with no arguments. But `exec` is
# not optional: without it the shell stays PID 1, SIGTERM never reaches uvicorn, and every
# stop waits out the 10-second grace period before a SIGKILL cuts requests off mid-flight.
# `exec` replaces the shell with uvicorn, so it receives signals directly and shuts down
# cleanly. Docker's JSONArgsRecommended warning is pointing at exactly this.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
