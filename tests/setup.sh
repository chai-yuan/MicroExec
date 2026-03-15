#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== MicroExec End-to-End Test Setup ==="

# ---- Python venv ----
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating Python virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment already exists, skipping creation."
fi

echo "[2/4] Installing Python dependencies ..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q

# ---- Build compiler ----
echo "[3/4] Building compiler ..."
if [ ! -f "$PROJECT_ROOT/compiler/frontend/onnx.pb.h" ]; then
    make -C "$PROJECT_ROOT/compiler" init
fi
make -C "$PROJECT_ROOT/compiler"

# ---- Build runtime library ----
echo "[4/4] Building runtime library ..."
make -C "$PROJECT_ROOT/runtime"

echo ""
echo "=== Setup complete ==="
echo "Run tests with:"
echo "  $VENV_DIR/bin/python $SCRIPT_DIR/run_test.py <model.onnx>"
