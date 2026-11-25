#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build_cpu
${CXX:-g++} -O2 -std=c++17 -Wall -Wextra -DCPU_ONLY main.cpp -o build_cpu/demo_cpu
./build_cpu/demo_cpu 0 "${1:-256}" "${2:-1}" "${3:-20}"
