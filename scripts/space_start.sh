#!/bin/sh
set -eu

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
UVICORN_LOG_PATH="${UVICORN_LOG_PATH:-/tmp/uvicorn.log}"

UVICORN_PID=""

_start_api_server_background() {
  if ! _is_true "$SERVE_API_FLAG"; then
    echo "[space_start] RUN_SPACE_API_SERVER disabled. Skipping API server."
    return
  fi
  echo "[space_start] Starting API server in background on port ${PORT_VALUE} (logs: ${UVICORN_LOG_PATH})."
  # API server runs in background ONLY for HF healthchecks. Training is the
  # primary process. If HF infrastructure SIGTERMs the container we still
  # want training to receive the signal and flush a final checkpoint, not
  # to silently die because PID 1 (uvicorn previously) exited first.
  uvicorn server:app --host 0.0.0.0 --port "${PORT_VALUE}" \
    >"${UVICORN_LOG_PATH}" 2>&1 &
  UVICORN_PID=$!
  echo "[space_start] uvicorn pid=${UVICORN_PID}"
}

_stop_api_server() {
  if [ -n "${UVICORN_PID}" ] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "[space_start] Stopping uvicorn pid=${UVICORN_PID}."
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}

trap '_stop_api_server' EXIT INT TERM

_train_self_play() {
  if [ -n "${TRAIN_OUTPUT_DIR}" ]; then
    OUTPUT_ARG="--train-output-dir ${TRAIN_OUTPUT_DIR}"
  else
    OUTPUT_ARG=""
  fi

  if _is_true "$DRY_RUN_FLAG"; then
    echo "[space_start] Running self-play in dry-run mode."
    # shellcheck disable=SC2086
    osint-env train-self-play --config "${ENV_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}" ${OUTPUT_ARG} --dry-run
  else
    echo "[space_start] Running self-play training (foreground)."
    # shellcheck disable=SC2086
    osint-env train-self-play --config "${ENV_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}" ${OUTPUT_ARG}
  fi

  echo "[space_start] Self-play command completed."
  echo "[space_start] Training end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}

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
  _start_api_server_background
  # Run training in the FOREGROUND so the script (and therefore PID 1)
  # blocks until training is finished. A graceful SIGTERM from HF will
  # propagate to the training process via the shell's signal handling
  # and the trap above will cleanly stop uvicorn afterwards.
  _train_self_play
  echo "[space_start] Training finished. Keeping API server alive for log inspection."
  if [ -n "${UVICORN_PID}" ] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    wait "${UVICORN_PID}"
  fi
else
  echo "[space_start] RUN_SELF_PLAY_TRAINING disabled. Skipping self-play run."
  _start_api_server_background
  if [ -n "${UVICORN_PID}" ] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    wait "${UVICORN_PID}"
  fi
fi
