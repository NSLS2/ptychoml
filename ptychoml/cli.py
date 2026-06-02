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
        default=2 << 30,
        help="TensorRT workspace size in bytes (default: 2 GiB).",
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
    """Run PtychoViT inference on diffraction patterns stored in an HDF5 file.

    Mirrors holoptycho's live preprocessing pipeline so a recorded scan can
    be replayed offline through the exact same code path:

        intensity → preprocess_diffraction(
            normalization, scale,
            hot_pixel_count_threshold,
            dp_orient, fftshift={auto,on,off},
        ) → PtychoViTInference.predict

    Pass the same scaler/orient settings the live pipeline used; comparing
    the live mosaic against this CLI's output is the canonical way to
    isolate "is this a preprocessing bug or a model bug?".
    """
    import numpy as np

    parser = argparse.ArgumentParser(
        prog="ptychoml-predict",
        description="Run PtychoViT inference on an HDF5 dataset (matches holoptycho's live preprocessing).",
    )
    parser.add_argument(
        "--engine",
        required=True,
        help="Path to a TensorRT .engine file.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the input HDF5 file (expects intensity by default).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output file with predictions (.npy by default, .h5 if --h5 is set).",
    )
    parser.add_argument(
        "--h5",
        action="store_true",
        help="Write output as HDF5 (.h5) instead of the default NumPy (.npy) format.",
    )
    parser.add_argument(
        "--dataset",
        default="dp",
        help="HDF5 dataset key for input frames. 'dp' for hxn_to_vit.py output (intensity); "
             "'diffamp' for HXN-native (amplitude — combine with --input-kind=amplitude). "
             "Default: 'dp'.",
    )
    parser.add_argument(
        "--input-kind",
        choices=("intensity", "amplitude"),
        default="intensity",
        help="Whether the source dataset stores counts ('intensity') or sqrt(counts) "
             "('amplitude'). Amplitude is squared before preprocess_diffraction runs so "
             "the same caller-facing parameters apply in both cases. Default: intensity.",
    )
    parser.add_argument(
        "--normalization",
        type=float,
        default=None,
        help="Per-scan max intensity used to scale DPs onto the model's amplitude "
             "range. If omitted, computed from the input via "
             "compute_intensity_normalization (max with --hot-pixel-count-threshold "
             "applied as the exclusion cutoff).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10000.0,
        help="Global scale factor (default: 10000.0, matches ptycho-vit's config.yaml).",
    )
    parser.add_argument(
        "--hot-pixel-count-threshold",
        type=float,
        default=50000.0,
        help="Photon-count threshold for hot-pixel zeroing (default 50000.0, "
             "matches hxn_to_vit). Pass a non-finite value (e.g. inf) to disable.",
    )
    parser.add_argument(
        "--dp-orient",
        default="identity",
        help="D4 transform applied to detector frames before inference. Default 'identity' "
             "(assume input is already in model orientation, e.g. dp.hdf5 from hxn_to_vit.py). "
             "Pass the live-mode value to reproduce a holoptycho run.",
    )
    parser.add_argument(
        "--fftshift",
        choices=("auto", "on", "off"),
        default="auto",
        help="DC-convention control for the model input. 'auto' (default) "
             "detects whether the central beam is at the corners and applies "
             "np.fft.fftshift iff it is; 'on' forces a shift; 'off' skips it. "
             "The model is trained with the central beam at the center of the "
             "frame; 'auto' is correct for both hxn_to_vit dp.hdf5 (already "
             "centered) and raw /diffamp (centered at corner).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device ordinal (default: 0).",
    )
    args = parser.parse_args(argv)

    import h5py
    from .inference import PtychoViTInference
    from .preprocess import (
        compute_intensity_normalization,
        preprocess_diffraction,
    )

    with h5py.File(args.data, "r") as f_in:
        if args.dataset not in f_in:
            available = list(f_in.keys())
            print(
                f"Dataset '{args.dataset}' not found in {args.data}. "
                f"Available keys: {available}",
                file=sys.stderr,
            )
            return 1
        raw = np.array(f_in[args.dataset])

    print(
        f"Loaded {args.dataset}: shape={raw.shape}, dtype={raw.dtype}, "
        f"kind={args.input_kind}"
    )

    # Map amplitude → intensity so preprocess_diffraction can run a single
    # well-defined pipeline. Squaring is the inverse of the sqrt that
    # preprocess_diffraction applies internally, so a perfect-fidelity
    # amplitude input round-trips back to the same amplitude.
    if args.input_kind == "amplitude":
        intensity = (raw.astype(np.float64)) ** 2
    else:
        intensity = raw

    # In offline / CLI mode the full DP stack is in hand, so the per-scan
    # normalization can be derived directly from the data — match what
    # hxn_to_vit.py writes per scan. holoptycho's live path can't do this
    # (no full stack) and gets the value from the scan JSON instead.
    if args.normalization is None:
        normalization = compute_intensity_normalization(
            intensity, hot_pixel_count_threshold=args.hot_pixel_count_threshold,
        )
        print(
            f"Computed normalization from input (hot-pixel cutoff="
            f"{args.hot_pixel_count_threshold}): {normalization:g}"
        )
    else:
        normalization = args.normalization

    fftshift_choice = {'auto': None, 'on': True, 'off': False}[args.fftshift]
    diff_amp = preprocess_diffraction(
        intensity,
        normalization=normalization,
        scale=args.scale,
        hot_pixel_count_threshold=args.hot_pixel_count_threshold,
        dp_orient=args.dp_orient,
        fftshift=fftshift_choice,
    )
    print(
        f"After preprocess_diffraction: shape={diff_amp.shape}, dtype={diff_amp.dtype}, "
        f"min={float(diff_amp.min()):.3g}, max={float(diff_amp.max()):.3g}"
    )

    with PtychoViTInference(
        engine_path=args.engine,
        gpu=args.gpu,
        fftshift=False,  # preprocess_diffraction above already centered DC
    ) as session:
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

    if args.h5:
        with h5py.File(args.output, "w") as f_out:
            f_out.create_dataset("predictions", data=predictions)
            with h5py.File(args.data, "r") as f_in:
                if "points" in f_in:
                    f_out.create_dataset("points", data=np.array(f_in["points"]))
        print(f"Wrote predictions to {args.output} (HDF5)")
    else:
        np.save(args.output, predictions)
        print(f"Wrote predictions to {args.output} (npy)")
        with h5py.File(args.data, "r") as f_in:
            if "points" in f_in:
                from pathlib import Path
                points_path = Path(args.output).with_suffix("").as_posix() + "_points.npy"
                np.save(points_path, np.array(f_in["points"]))
                print(f"Wrote points to {points_path} (npy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
