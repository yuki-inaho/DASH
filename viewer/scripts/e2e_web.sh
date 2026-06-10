#!/usr/bin/env bash
# L4 web E2E with playwright-cli: serve the web player and drive a headless
# chromium to verify the DASH 4DGS playback renders and animates in a browser.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

# playwright-cli ships with node (nvm). Make it (and node) reachable.
export PATH="$HOME/.nvm/versions/node/v22.22.3/bin:$PATH"
mkdir -p e2e-out

if [ ! -f web/baked/manifest.json ]; then
  echo "E2E_WEB_FAIL: no baked frames; run 'pixi run build-web' first"
  exit 2
fi

SES="dashweb"
HTTP_LOG="/tmp/dash_e2e_http.log"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
python3 -u -m http.server "$PORT" --bind 127.0.0.1 --directory web >"$HTTP_LOG" 2>&1 &
HTTP_PID=$!

cleanup() {
  playwright-cli -s="$SES" close >/dev/null 2>&1 || true
  kill "$HTTP_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Wait until the server actually accepts connections.
ready=0
for _ in $(seq 1 50); do
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then ready=1; break; fi
  sleep 0.2
done
if [ "$ready" -ne 1 ]; then echo "E2E_WEB_FAIL: http server didn't start"; cat "$HTTP_LOG"; exit 2; fi
URL="http://127.0.0.1:${PORT}/"
echo "serving web/ at $URL"

if ! playwright-cli -s="$SES" open --browser chromium >/dev/null 2>&1; then
  echo "E2E_WEB_FAIL: could not launch chromium"; exit 2
fi
playwright-cli -s="$SES" goto "$URL" >/dev/null 2>&1
sleep 5  # load + decode PNGs and let the flipbook advance several frames

REPORT=$(playwright-cli --raw -s="$SES" eval "() => window.__report()" 2>&1)
echo "REPORT=$REPORT"
playwright-cli -s="$SES" screenshot --filename "$HERE/e2e-out/web_playwright.png" >/dev/null 2>&1 || true
CONSOLE=$(playwright-cli --raw -s="$SES" console error 2>&1 | head -3)
echo "CONSOLE_ERROR_LEVEL=${CONSOLE:-<none>}"

fail=0
echo "$REPORT" | grep -q "ready"     || { echo "ASSERT FAIL: player not ready"; fail=1; }
echo "$REPORT" | grep -q "errors=0"  || { echo "ASSERT FAIL: page/console errors present"; fail=1; }
DRAWN=$(echo "$REPORT"   | grep -oE "drawn=[0-9]+"      | grep -oE "[0-9]+" | head -1)
DISTINCT=$(echo "$REPORT"| grep -oE "distinct=[0-9]+"   | grep -oE "[0-9]+" | head -1)
NB=$(echo "$REPORT"      | grep -oE "nonblank=[0-9.]+"  | grep -oE "[0-9.]+" | head -1)
FRAMES=$(echo "$REPORT"  | grep -oE "frames=[0-9]+"     | grep -oE "[0-9]+" | head -1)
[ "${FRAMES:-0}"   -ge 2 ] || { echo "ASSERT FAIL: frameCount=$FRAMES"; fail=1; }
[ "${DRAWN:-0}"    -ge 2 ] || { echo "ASSERT FAIL: playback not advancing (drawn=$DRAWN)"; fail=1; }
[ "${DISTINCT:-0}" -ge 2 ] || { echo "ASSERT FAIL: rendered frames identical (distinct=$DISTINCT)"; fail=1; }
awk "BEGIN{exit !(${NB:-0} >= 0.05)}" || { echo "ASSERT FAIL: canvas mostly blank (nonblank=$NB)"; fail=1; }

if [ "$fail" -eq 0 ]; then
  echo "E2E_WEB_PASS  (playwright-cli verified: rendered=$NB nonblank, $DISTINCT distinct frames, $DRAWN draws, 0 errors)"
else
  echo "E2E_WEB_FAIL"
fi
exit "$fail"
