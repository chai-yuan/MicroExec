#!/usr/bin/env python3
"""
MicroExec End-to-End Test Runner

Compiles an ONNX model via compiler.elf, runs inference through both the
MicroExec C runtime and Python onnxruntime, then compares the outputs.

Usage:
    python run_test.py <model.onnx> [--rtol 1e-5] [--atol 1e-5] [--seed 42]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMPILER_ELF = os.path.join(PROJECT_ROOT, "compiler", "compiler.elf")
HARNESS_DIR = os.path.join(SCRIPT_DIR, "test_harness")
HARNESS_ELF = os.path.join(HARNESS_DIR, "test_harness.elf")

# ---------------------------------------------------------------------------
# ONNX elem_type  ->  (numpy dtype, MeScalarType int)
# ---------------------------------------------------------------------------

_ONNX_DTYPE_MAP = {
    onnx.TensorProto.FLOAT: ("float32", 1),
    onnx.TensorProto.UINT8: ("uint8", 6),
    onnx.TensorProto.INT8: ("int8", 5),
    onnx.TensorProto.INT32: ("int32", 4),
    onnx.TensorProto.INT64: ("int64", 3),
    onnx.TensorProto.FLOAT16: ("float16", 2),
    onnx.TensorProto.BOOL: ("bool", 7),
}

_ME_DTYPE_TO_NP = {
    1: np.float32,
    2: np.float16,
    3: np.int64,
    4: np.int32,
    5: np.int8,
    6: np.uint8,
    7: np.bool_,
}


def onnx_elem_to_np_dtype(elem_type: int) -> str:
    entry = _ONNX_DTYPE_MAP.get(elem_type)
    if entry is None:
        raise ValueError(f"Unsupported ONNX elem_type: {elem_type}")
    return entry[0]


def onnx_elem_to_me_dtype(elem_type: int) -> int:
    entry = _ONNX_DTYPE_MAP.get(elem_type)
    if entry is None:
        raise ValueError(f"Unsupported ONNX elem_type: {elem_type}")
    return entry[1]


# ---------------------------------------------------------------------------
# ONNX model introspection
# ---------------------------------------------------------------------------


def get_input_specs(model: onnx.ModelProto, default_dynamic_dim: int = 1):
    """Return list of (name, numpy_dtype, concrete_shape, me_dtype_int)."""
    specs = []
    for inp in model.graph.input:
        name = inp.name
        tensor_type = inp.type.tensor_type
        elem_type = tensor_type.elem_type
        np_dtype = onnx_elem_to_np_dtype(elem_type)
        me_dtype = onnx_elem_to_me_dtype(elem_type)

        shape = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(default_dynamic_dim)

        specs.append((name, np_dtype, tuple(shape), me_dtype))
    return specs


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def ensure_compiler():
    if not os.path.isfile(COMPILER_ELF):
        print("[build] Compiling compiler.elf ...")
        subprocess.run(
            ["make", "-C", os.path.join(PROJECT_ROOT, "compiler")],
            check=True,
        )
    if not os.path.isfile(COMPILER_ELF):
        sys.exit("ERROR: compiler.elf not found after build")


def ensure_harness():
    print("[build] Building test harness ...")
    subprocess.run(["make", "-C", HARNESS_DIR], check=True)
    if not os.path.isfile(HARNESS_ELF):
        sys.exit("ERROR: test_harness.elf not found after build")


# ---------------------------------------------------------------------------
# Compile ONNX -> MVMP
# ---------------------------------------------------------------------------


def compile_onnx(onnx_path: str, mvmp_path: str):
    print(f"[compile] {onnx_path} -> {mvmp_path}")
    result = subprocess.run(
        [COMPILER_ELF, onnx_path, mvmp_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"ERROR: compilation failed (exit {result.returncode})")


# ---------------------------------------------------------------------------
# Run MicroExec inference via C harness
# ---------------------------------------------------------------------------


def run_microexec(mvmp_path, work_dir, input_specs):
    """Run the C harness and return list of (numpy_array,) outputs."""

    cmd = [HARNESS_ELF, mvmp_path, work_dir, str(len(input_specs))]
    for _, _, shape, me_dtype in input_specs:
        cmd.append(str(me_dtype))
        cmd.append(str(len(shape)))
        cmd.extend(str(d) for d in shape)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"ERROR: test harness failed (exit {result.returncode})")

    outputs = []
    for i, line in enumerate(result.stdout.strip().splitlines()):
        parts = line.split()
        dtype_int = int(parts[0])
        ndim = int(parts[1])
        shape = tuple(int(parts[2 + d]) for d in range(ndim))

        np_dtype = _ME_DTYPE_TO_NP.get(dtype_int)
        if np_dtype is None:
            sys.exit(f"ERROR: unknown MeScalarType {dtype_int} for output {i}")

        bin_path = os.path.join(work_dir, f"output_{i}.bin")
        data = np.fromfile(bin_path, dtype=np_dtype).reshape(shape)
        outputs.append(data)

    return outputs


# ---------------------------------------------------------------------------
# Run ONNX Runtime inference
# ---------------------------------------------------------------------------


def run_onnxruntime(onnx_path, input_feeds):
    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, input_feeds)


# ---------------------------------------------------------------------------
# Compare outputs
# ---------------------------------------------------------------------------


def compare_outputs(me_outputs, ort_outputs, rtol, atol):
    if len(me_outputs) != len(ort_outputs):
        print(
            f"FAIL: output count mismatch "
            f"(MicroExec={len(me_outputs)}, ORT={len(ort_outputs)})"
        )
        return False

    all_pass = True
    for i, (me_out, ort_out) in enumerate(zip(me_outputs, ort_outputs)):
        me_flat = me_out.astype(np.float64).flatten()
        ort_flat = ort_out.astype(np.float64).flatten()

        if me_flat.shape != ort_flat.shape:
            print(
                f"  Output {i}: FAIL  "
                f"shape mismatch MicroExec={me_out.shape} vs ORT={ort_out.shape}"
            )
            all_pass = False
            continue

        abs_diff = np.abs(me_flat - ort_flat)
        max_abs = np.max(abs_diff)
        denom = np.maximum(np.abs(ort_flat), 1e-12)
        max_rel = np.max(abs_diff / denom)

        passed = np.allclose(me_flat, ort_flat, rtol=rtol, atol=atol)
        status = "PASS" if passed else "FAIL"
        print(
            f"  Output {i}: {status}  "
            f"(max_abs_err={max_abs:.6e}, max_rel_err={max_rel:.6e})"
        )
        if not passed:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MicroExec end-to-end ONNX test runner"
    )
    parser.add_argument("onnx_model", help="Path to the ONNX model file")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible inputs (default: 42)",
    )
    parser.add_argument(
        "--dynamic-dim",
        type=int,
        default=1,
        help="Value for dynamic/symbolic dimensions (default: 1)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Do not remove the temporary work directory",
    )
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.onnx_model)
    if not os.path.isfile(onnx_path):
        sys.exit(f"ERROR: file not found: {onnx_path}")

    print(f"=== MicroExec E2E Test: {os.path.basename(onnx_path)} ===")
    print(f"    rtol={args.rtol}  atol={args.atol}  seed={args.seed}")
    print()

    # 1. Load ONNX model
    model = onnx.load(onnx_path)
    input_specs = get_input_specs(model, default_dynamic_dim=args.dynamic_dim)
    print(f"[info] Model inputs ({len(input_specs)}):")
    for name, dtype, shape, _ in input_specs:
        print(f"       {name}: {dtype}{list(shape)}")
    print()

    # 2. Build compiler & harness
    ensure_compiler()
    ensure_harness()
    print()

    # 3. Create work directory
    work_dir = tempfile.mkdtemp(prefix="me_test_")

    try:
        # 4. Compile ONNX -> MVMP
        mvmp_path = os.path.join(work_dir, "model.mvmp")
        compile_onnx(onnx_path, mvmp_path)
        print()

        # 5. Generate random inputs
        rng = np.random.RandomState(args.seed)
        input_feeds = {}
        for i, (name, np_dtype, shape, _) in enumerate(input_specs):
            if np_dtype in ("float32", "float16"):
                data = rng.randn(*shape).astype(np_dtype)
            elif np_dtype in ("int32", "int64"):
                data = rng.randint(-10, 10, size=shape).astype(np_dtype)
            elif np_dtype in ("uint8",):
                data = rng.randint(0, 255, size=shape).astype(np_dtype)
            elif np_dtype in ("int8",):
                data = rng.randint(-128, 127, size=shape).astype(np_dtype)
            elif np_dtype == "bool":
                data = rng.randint(0, 2, size=shape).astype(np.bool_)
            else:
                data = rng.randn(*shape).astype(np_dtype)

            bin_path = os.path.join(work_dir, f"input_{i}.bin")
            data.tofile(bin_path)
            input_feeds[name] = data

        # 6. Run MicroExec
        print("[run] MicroExec inference ...")
        me_outputs = run_microexec(mvmp_path, work_dir, input_specs)

        # 7. Run ONNX Runtime
        print("[run] ONNX Runtime inference ...")
        ort_outputs = run_onnxruntime(onnx_path, input_feeds)
        print()

        # 8. Compare
        print("[compare] Checking outputs ...")
        passed = compare_outputs(me_outputs, ort_outputs, args.rtol, args.atol)
        print()

        if passed:
            print("RESULT: ALL PASS")
        else:
            print("RESULT: FAIL")

    finally:
        if args.keep_tmp:
            print(f"[info] Temp directory kept at: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
