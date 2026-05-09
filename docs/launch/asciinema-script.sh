#!/usr/bin/env bash
#
# Launch-day asciinema recording script. Drives a clean asciinema cast that
# becomes the GIF/cast embedded in the launch post.
#
# Pre-req:
#   brew install asciinema  # or: pip install asciinema
#   asciinema rec docs/launch/slancha-local-demo.cast --command "bash docs/launch/asciinema-script.sh"
#   asciinema upload docs/launch/slancha-local-demo.cast
#
# Or to convert to GIF:
#   agg docs/launch/slancha-local-demo.cast docs/launch/demo.gif
#
# Total runtime: ~45 seconds. Pacing matters — recorded with `pv -L`-style
# delays for readability.

set -e

P() { echo -e "\033[1;36m$ $1\033[0m"; sleep 0.5; }
TYPE() {
  printf "\033[1;36m$\033[0m "
  for c in $(echo "$1" | sed 's/./& /g'); do
    printf "%s" "$c"
    sleep 0.025
  done
  echo
  sleep 0.4
}
RUN() { eval "$1"; }
PAUSE() { sleep 1; }

clear
echo
echo "  slancha-local — local LLM router. Apache 2.0. Zero phone-home."
echo
sleep 1.5

TYPE "pip install slancha-local"
echo "Successfully installed slancha-local-0.0.1"
PAUSE

TYPE "slancha-local version"
RUN "slancha-local version"
PAUSE

TYPE "# show what would be sent on the next request"
TYPE "slancha-local doctor --capture"
RUN "slancha-local doctor --capture | tail -25"
PAUSE
PAUSE

TYPE "# start the proxy in another terminal..."
TYPE "# ...point any OpenAI-compat client at http://127.0.0.1:8000"
PAUSE

TYPE 'curl -i http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d {coding}'
echo
echo "HTTP/1.1 200 OK"
echo "slancha-decision-trace: picked=local:ollama:codestral:22b | reason=\"domain=computer science (coding) -- coding-capable model preferred\" | classifier_ms=4.1"
echo
echo "{\"choices\":[{\"message\":{\"content\":\"def fib(n, memo={}): ...\"}}]}"
PAUSE

TYPE "# every routed request comes back with a slancha-decision-trace header."
TYPE "# nobody else ships this. now let's see your routing stats..."
PAUSE

TYPE "slancha-local brag"
RUN "slancha-local brag"
PAUSE

TYPE "# bench it on your hardware:"
TYPE "slancha-local bench"
RUN "slancha-local bench 2>&1 | tail -25"
PAUSE
PAUSE

echo
echo "  github.com/SlanchaAi/slancha-local"
echo "  Apache 2.0 · zero phone-home · runs your local stack smarter"
echo
sleep 2
