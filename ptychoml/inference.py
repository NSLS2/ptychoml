"""Stateful PtychoViT inference session.

Open a session once (loads TRT engine, allocates GPU buffers), call predict()
many times. Same API serves both streaming (one batch at a time) and batch
(all frames at once) use cases.

No PyTorch, no Holoscan — just TensorRT + PyCUDA + numpy.
"""
import logging
from typing import Optional, Tuple

import numpy as np

from .trt import load_engine, allocate_io_buffers, infer, reshape_output_flat

logger = logging.getLogger(__name__)


class PtychoViTInference:
    """Run PtychoViT neural network inference via TensorRT.

    Lifecycle:
        session = PtychoViTInference(engine_path, gpu=0)
        # first predict() call lazily initializes the CUDA context + TRT engine
        pred, indices = session.predict(diff_amp, image_indices)
        # ...many more predict() calls...
        session.cleanup()

    Or as a context manager:
        with PtychoViTInference(engine_path) as session:
            pred, _ = session.predict(diff_amp)
    """

    def __init__(
        self,
        engine_path: str,
        gpu: int = 0,
        data_is_shifted: bool = False,
    ):
        """
        Args:
            engine_path:     Path to a TensorRT .engine file (built from ONNX).
            gpu:             CUDA device ordinal (default 0).
            data_is_shifted: If True, input diffraction patterns have been
                             fftshift'd by the caller and need to be unshifted
                             before inference (the model expects DC at the
                             corner). Set True in live mode where upstream
                             preprocessing applies fftshift; False when the
                             caller provides raw diffraction amplitudes.
        """
        self.engine_path = engine_path
        self.gpu = int(gpu)
        self._data_is_shifted = bool(data_is_shifted)

        # Lazy-initialized on first predict()
        self._initialized = False
        self.cuda_ctx = None
        self.trt_context = None
        self.trt_inputs = None
        self.trt_outputs = None
        self.trt_bindings = None
        self.trt_stream = None
        self.expected_input_shape: Optional[Tuple[int, ...]] = None
        self.expected_output_shape: Optional[Tuple[int, ...]] = None

        # Stats
        self.n_batches = 0

    def _init_engine(self) -> None:
        """Initialize CUDA context, load engine, and allocate buffers."""
        import pycuda.driver as drv

        drv.init()
        self.cuda_ctx = drv.Device(self.gpu).make_context()
        logger.info(
            "PyCUDA context created on GPU %d (%s)",
            self.gpu,
            drv.Device(self.gpu).name(),
        )

        engine = load_engine(self.engine_path)
        self.trt_context = engine.create_execution_context()
        (
            self.trt_inputs,
            self.trt_outputs,
            self.trt_bindings,
            self.trt_stream,
        ) = allocate_io_buffers(engine)

        self.expected_input_shape = tuple(self.trt_inputs[0]["shape"])
        self.expected_output_shape = tuple(self.trt_outputs[0]["shape"])
        logger.info(
            "TRT engine loaded: %s | input=%s | output=%s",
            self.engine_path,
            self.expected_input_shape,
            self.expected_output_shape,
        )
        self._initialized = True

    def predict(
        self,
        diff_amp: np.ndarray,
        image_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run inference on a batch of diffraction amplitudes.

        Args:
            diff_amp:      [B, H, W] float32 diffraction amplitudes.
            image_indices: Optional [B] int32 frame indices, passed through
                           in the return tuple for caller correlation.

        Returns:
            (pred, indices) where pred is [B, 2, H, W] or [B, H, W] float32
            (amplitude+phase or single output, depending on the trained model).
        """
        if not self._initialized:
            self._init_engine()

        # Model was trained on unshifted diffraction amplitudes (DC at corners).
        # For even-sized arrays, fftshift == ifftshift.
        if self._data_is_shifted:
            diff_amp = np.fft.fftshift(diff_amp, axes=(1, 2))

        B_actual = diff_amp.shape[0]
        H_data = diff_amp.shape[1]
        W_data = diff_amp.shape[2]
        B_engine = self.expected_input_shape[0]
        H_engine = self.expected_input_shape[2]
        W_engine = self.expected_input_shape[3]

        # [B, 1, H, W]
        model_input = diff_amp[:, np.newaxis, :, :]

        # Spatial padding: center-pad if data smaller than engine dims.
        spatial_pad = None
        if H_data != H_engine or W_data != W_engine:
            pad_h = H_engine - H_data
            pad_w = W_engine - W_data
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"Data spatial dims ({H_data},{W_data}) larger than engine "
                    f"({H_engine},{W_engine}). Cannot run inference."
                )
            top = pad_h // 2
            left = pad_w // 2
            spatial_pad = (top, top + H_data, left, left + W_data)
            padded = np.zeros(
                (B_actual, 1, H_engine, W_engine), dtype=np.float32
            )
            padded[:, :, top:top + H_data, left:left + W_data] = model_input
            model_input = padded

        # Batch padding: pad final batch if smaller than engine batch size.
        if B_actual < B_engine:
            pad = np.zeros(
                (B_engine - B_actual, 1, H_engine, W_engine), dtype=np.float32
            )
            model_input = np.concatenate([model_input, pad], axis=0)
        elif B_actual > B_engine:
            raise ValueError(
                f"Batch too large: input {B_actual} vs engine {B_engine}. "
                "Check that ONNX batch size matches caller's batch size."
            )

        model_input = np.ascontiguousarray(model_input, dtype=np.float32)

        # Run inference
        np.copyto(self.trt_inputs[0]["host"], model_input.ravel())
        output_flat = np.array(
            infer(
                self.trt_context,
                self.trt_inputs,
                self.trt_outputs,
                self.trt_bindings,
                self.trt_stream,
                cuda_context=self.cuda_ctx,
            )[0]
        )

        # Reshape output and strip padding
        pred = reshape_output_flat(
            output_flat,
            batch_size=B_engine,
            height=H_engine,
            width=W_engine,
        )
        pred = pred[:B_actual]  # strip batch padding

        if spatial_pad is not None:
            top, bot, left, right = spatial_pad
            if pred.ndim == 4:  # [B, 2, H, W]
                pred = pred[:, :, top:bot, left:right]
            else:  # [B, H, W]
                pred = pred[:, top:bot, left:right]

        self.n_batches += 1
        return pred, image_indices

    def cleanup(self) -> None:
        """Release CUDA context and TensorRT resources."""
        if self.cuda_ctx is not None:
            try:
                self.cuda_ctx.pop()
            except Exception:
                pass
            self.cuda_ctx = None
        self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
