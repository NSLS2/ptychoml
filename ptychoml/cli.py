"""CLI entry points for ptychoml.

Console scripts:
    ptychoml-build-engine  — build a TensorRT engine from ONNX
    ptychoml-predict       — run inference on an HDF5 dataset
"""
import argparse
import sys


def main(argv=None) -> int:
    """Build a TensorRT engine from an ONNX model."""
    from .trt import build_engine_from_onnx, save_engine

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


def predict_main(argv=None) -> int:
    """Run PtychoViT inference on diffraction patterns stored in an HDF5 file."""
    import numpy as np

    parser = argparse.ArgumentParser(
        prog="ptychoml-predict",
        description="Run PtychoViT inference on an HDF5 dataset.",
    )
    parser.add_argument(
        "--engine",
        required=True,
        help="Path to a TensorRT .engine file.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the input HDF5 file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output HDF5 file with predictions.",
    )
    parser.add_argument(
        "--dataset",
        default="diffamp",
        help="HDF5 dataset key for diffraction amplitudes (default: 'diffamp').",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device ordinal (default: 0).",
    )
    parser.add_argument(
        "--shifted",
        action="store_true",
        help="Set if input data has been fftshift'd.",
    )
    args = parser.parse_args(argv)

    import h5py
    from .inference import PtychoViTInference

    with h5py.File(args.data, "r") as f_in:
        if args.dataset not in f_in:
            available = list(f_in.keys())
            print(
                f"Dataset '{args.dataset}' not found in {args.data}. "
                f"Available keys: {available}",
                file=sys.stderr,
            )
            return 1
        diff_amp = np.array(f_in[args.dataset], dtype=np.float32)

    print(f"Loaded {args.dataset}: shape={diff_amp.shape}, dtype={diff_amp.dtype}")

    with PtychoViTInference(
        engine_path=args.engine,
        gpu=args.gpu,
        data_is_shifted=args.shifted,
    ) as session:
        # Determine engine batch size from first predict call's lazy init
        session._init_engine()
        batch_size = session.expected_input_shape[0]
        n_frames = diff_amp.shape[0]

        all_preds = []
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch = diff_amp[start:end]
            pred, _ = session.predict(batch)
            all_preds.append(pred)
            print(
                f"  batch {start}:{end} -> pred shape {pred.shape}",
            )

    predictions = np.concatenate(all_preds, axis=0)
    print(f"Total predictions: shape={predictions.shape}")

    with h5py.File(args.output, "w") as f_out:
        f_out.create_dataset("predictions", data=predictions)
        # Copy scan points through if present in input
        with h5py.File(args.data, "r") as f_in:
            if "points" in f_in:
                f_out.create_dataset("points", data=np.array(f_in["points"]))

    print(f"Wrote predictions to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
