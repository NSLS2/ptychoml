"""CLI: build a TensorRT engine from an ONNX model.

Invoked via the `ptychoml-build-engine` console script or
`pixi run build-engine`.
"""
import argparse
import sys

from .trt import build_engine_from_onnx, save_engine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ptychoml-build-engine",
        description="Build a TensorRT engine from an ONNX model.",
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the built .engine file will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision (default: off).",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 precision (default: on).",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=1 << 30,
        help="TensorRT workspace size in bytes (default: 1 GiB).",
    )
    args = parser.parse_args(argv)

    engine = build_engine_from_onnx(
        args.onnx,
        fp16=args.fp16,
        tf32=not args.no_tf32,
        max_workspace_size_bytes=args.workspace_size,
    )
    save_engine(engine, args.output)
    print(f"Wrote TensorRT engine to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
