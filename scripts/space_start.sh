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
RUN_FLAG="${RUN_SELF_PLAY_TRAINING:-0}"
DRY_RUN_FLAG="${RUN_SELF_PLAY_DRY_RUN:-0}"

if _is_true "$RUN_FLAG"; then
  echo "[space_start] RUN_SELF_PLAY_TRAINING enabled."
  echo "[space_start] Training start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[space_start] Env config: ${ENV_CONFIG_PATH}"
  echo "[space_start] Train config: ${TRAIN_CONFIG_PATH}"
  if _is_true "$DRY_RUN_FLAG"; then
    echo "[space_start] Running self-play in dry-run mode."
    osint-env train-self-play --config "${ENV_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}" --dry-run
  else
    echo "[space_start] Running self-play training."
    osint-env train-self-play --config "${ENV_CONFIG_PATH}" --train-config "${TRAIN_CONFIG_PATH}"
  fi
  echo "[space_start] Self-play command completed."
  echo "[space_start] Training end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
else
  echo "[space_start] RUN_SELF_PLAY_TRAINING disabled. Skipping self-play run."
fi

echo "[space_start] Starting API server."
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-7860}"
