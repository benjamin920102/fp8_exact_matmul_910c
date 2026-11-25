#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
CANN_ROOT="${CANN_ROOT:-${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.0.0}}"
if [[ -f "$CANN_ROOT/set_env.sh" ]]; then
    source "$CANN_ROOT/set_env.sh"
elif [[ -f "$CANN_ROOT/bin/setenv.bash" ]]; then
    source "$CANN_ROOT/bin/setenv.bash"
fi
export ASCEND_HOME_PATH="$CANN_ROOT"
export PATH="$CANN_ROOT/bin:$CANN_ROOT/compiler/ccec_compiler/bin:$CANN_ROOT/ccec_compiler/bin:$PATH"
export LD_LIBRARY_PATH="$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
args=()
if [[ -n "${ASC_DIR:-}" ]]; then
    args+=("-DASC_DIR=$ASC_DIR")
else
    cfg="$(find -L "$CANN_ROOT" -type f \( -iname 'ASCConfig.cmake' -o -iname 'asc-config.cmake' \) -print -quit 2>/dev/null || true)"
    if [[ -n "$cfg" ]]; then args+=("-DASC_DIR=$(dirname "$cfg")"); fi
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_ASC_ARCHITECTURES=dav-2201 "${args[@]}"
cmake --build build -j "${JOBS:-4}" --verbose
./build/demo "${1:-0}" "${2:-256}" "${3:-1}" "${4:-20}"
