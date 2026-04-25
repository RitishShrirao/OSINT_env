#!/bin/sh
set -eu

_is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

ENV_CONFIG_PATH="${TRAIN_ENV_CONFIG_PATH:-config/shared_config.json}"
TRAIN_CONFIG_PATH="${TRAIN_SELF_PLAY_CONFIG_PATH:-config/self_play_training_hf_a10g_smoke.json}"
TRAIN_OUTPUT_DIR="${TRAIN_SELF_PLAY_OUTPUT_DIR:-}"
RUN_FLAG="${RUN_SELF_PLAY_TRAINING:-0}"
DRY_RUN_FLAG="${RUN_SELF_PLAY_DRY_RUN:-0}"
BACKGROUND_FLAG="${RUN_SELF_PLAY_BACKGROUND:-1}"

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
    echo "[space_start] Running self-play training."
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
  if _is_true "$BACKGROUND_FLAG"; then
    echo "[space_start] Launching self-play in background so the Space API can stay online."
    _train_self_play &
  else
    _train_self_play
  fi
else
  echo "[space_start] RUN_SELF_PLAY_TRAINING disabled. Skipping self-play run."
fi

echo "[space_start] Starting API server."
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-7860}"
