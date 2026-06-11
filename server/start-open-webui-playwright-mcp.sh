#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=/workspace/open-webui-data
STATIC_DIR=/workspace/open-webui-static
PLAYWRIGHT_MCP_OUTPUT_DIR=/workspace/playwright-mcp-output
PLAYWRIGHT_MCP_USER_DATA_DIR=/workspace/playwright-mcp-profile

mkdir -p "${DATA_DIR}" "${STATIC_DIR}" "${PLAYWRIGHT_MCP_OUTPUT_DIR}" "${PLAYWRIGHT_MCP_USER_DATA_DIR}"

echo "Starting Playwright MCP OpenAPI tool server: http://127.0.0.1:55145"
mcpo \
    --host 127.0.0.1 \
    --port 55145 \
    -- \
    playwright-mcp \
    --headless \
    --browser chromium \
    --caps vision \
    --viewport-size 1920x1080 \
    --ignore-https-errors \
    --no-sandbox \
    --output-dir "${PLAYWRIGHT_MCP_OUTPUT_DIR}" \
    --user-data-dir "${PLAYWRIGHT_MCP_USER_DATA_DIR}" &
mcpo_pid=$!

echo "Starting Open WebUI: http://127.0.0.1:55146"
    HOST=127.0.0.1 \
    PORT=55146 \
    DATA_DIR="${DATA_DIR}" \
    STATIC_DIR="${STATIC_DIR}" \
    OPENAI_API_BASE_URLS=http://127.0.0.1:55144/v1\;http://127.0.0.1:55114/v1 \
    OPENAI_API_KEYS=not-needed\;not-needed \
    open-webui serve &
webui_pid=$!

terminate() {
    kill "${mcpo_pid}" "${webui_pid}" 2>/dev/null || true
    wait "${mcpo_pid}" "${webui_pid}" 2>/dev/null || true
}

trap terminate INT TERM EXIT

set +e
wait -n "${mcpo_pid}" "${webui_pid}"
status=$?
set -e

terminate
exit "${status}"
