#!/bin/sh
# Intentionally NOT running with `set -e`: a training failure must not
# bring down the API server. The Space's HTTP endpoints (dashboards,
# /healthz, /api/environment) need to stay reachable even if the
# self-play job crashes.
set -u

_is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ENV_CONFIG_PATH="${TRAIN_ENV_CONFIG_PATH:-config/shared_config.json}"
TRAIN_CONFIG_PATH="${TRAIN_SELF_PLAY_CONFIG_PATH:-config/self_play_training_hf_l40s_full.json}"
TRAIN_OUTPUT_DIR="${TRAIN_SELF_PLAY_OUTPUT_DIR:-}"
RUN_FLAG="${RUN_SELF_PLAY_TRAINING:-1}"
DRY_RUN_FLAG="${RUN_SELF_PLAY_DRY_RUN:-0}"
SERVE_API_FLAG="${RUN_SPACE_API_SERVER:-1}"
PORT_VALUE="${PORT:-7860}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-/tmp/self_play_training.log}"

UVICORN_PID=""
TRAIN_PID=""

_start_api_server_foreground_or_die() {
  if ! _is_true "$SERVE_API_FLAG"; then
    echo "[space_start] RUN_SPACE_API_SERVER disabled. Nothing to serve."
    exit 0
  fi
  echo "[space_start] Starting API server (foreground) on port ${PORT_VALUE}."
  uvicorn server:app --host 0.0.0.0 --port "${PORT_VALUE}" &
  UVICORN_PID=$!
  echo "[space_start] uvicorn pid=${UVICORN_PID}"
}

_stop_children() {
  if [ -n "${TRAIN_PID}" ] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[space_start] Forwarding shutdown to training pid=${TRAIN_PID}."
    kill "${TRAIN_PID}" 2>/dev/null || true
  fi
  if [ -n "${UVICORN_PID}" ] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "[space_start] Stopping uvicorn pid=${UVICORN_PID}."
    kill "${UVICORN_PID}" 2>/dev/null || true
  fi
}

# Only forward shutdown signals; do NOT kill children on every EXIT
# (otherwise a crashed training run would tear down uvicorn too).
trap '_stop_children; exit 0' INT TERM

_run_training_supervised() {
  if [ -n "${TRAIN_OUTPUT_DIR}" ]; then
    OUTPUT_ARG="--train-output-dir ${TRAIN_OUTPUT_DIR}"
  else
    OUTPUT_ARG=""
  fi

  if _is_true "$DRY_RUN_FLAG"; then
    echo "[space_start] Running self-play in dry-run mode (logs: ${TRAIN_LOG_PATH})."
    # shellcheck disable=SC2086
    osint-env train-self-play --config "${ENV_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}" ${OUTPUT_ARG} --dry-run \
      > "${TRAIN_LOG_PATH}" 2>&1 &
  else
    echo "[space_start] Running self-play training in background (logs: ${TRAIN_LOG_PATH})."
    # shellcheck disable=SC2086
    osint-env train-self-play --config "${ENV_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}" ${OUTPUT_ARG} \
      > "${TRAIN_LOG_PATH}" 2>&1 &
  fi
  TRAIN_PID=$!
  echo "[space_start] training pid=${TRAIN_PID}"

  # Watcher subshell: if training exits with non-zero status, log the
  # failure but do NOT propagate it to the parent script. Uvicorn must
  # keep serving so the dashboards stay reachable.
  (
    wait "${TRAIN_PID}" 2>/dev/null
    rc=$?
    if [ "${rc}" -eq 0 ]; then
      echo "[space_start] Self-play training finished cleanly (rc=0)."
    else
      echo "[space_start] Self-play training exited rc=${rc}. API server will stay up; see ${TRAIN_LOG_PATH}."
    fi
  ) &
}

_start_api_server_foreground_or_die

if _is_true "$RUN_FLAG"; then
  echo "[space_start] RUN_SELF_PLAY_TRAINING enabled."
  echo "[space_start] Training start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[space_start] Env config: ${ENV_CONFIG_PATH}"
  echo "[space_start] Train config: ${TRAIN_CONFIG_PATH}"
  if [ -n "${TRAIN_OUTPUT_DIR}" ]; then
    echo "[space_start] Train output dir: ${TRAIN_OUTPUT_DIR}"
  fi
  if [ -n "${OSINT_HF_CHECKPOINT_REPO_ID:-}" ]; then
    echo "[space_start] HF checkpoint repo: ${OSINT_HF_CHECKPOINT_REPO_ID}"
  fi
  _run_training_supervised
else
  echo "[space_start] RUN_SELF_PLAY_TRAINING disabled. Skipping self-play run."
fi

# Block on uvicorn so the container stays alive as long as the API
# server is healthy. If uvicorn exits (e.g. real platform shutdown),
# we exit the script normally.
if [ -n "${UVICORN_PID}" ]; then
  wait "${UVICORN_PID}"
fi
