"""Smoke tests for the ptychoml package.

Verify that every module imports cleanly. Modules that require TensorRT or
PyCUDA at runtime defer those imports to function bodies, so the modules
themselves can be imported in a plain CI environment (no TRT, no GPU).
Actual inference tests are gated with ``pytest.importorskip``.
"""
import importlib

import pytest


# All ptychoml modules should import cleanly without TensorRT or PyCUDA.
PURE_MODULES = [
    "ptychoml",
    "ptychoml.trt",
    "ptychoml.inference",
    "ptychoml.cli",
]


@pytest.mark.parametrize("module_name", PURE_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_public_api_exposed():
    """The top-level package re-exports the public API symbols."""
    import ptychoml

    assert hasattr(ptychoml, "PtychoViTInference")
    assert hasattr(ptychoml, "build_engine")
    assert hasattr(ptychoml, "load_engine")
    assert hasattr(ptychoml, "save_engine")


def test_cli_help_runs(capsys):
    """`ptychoml-build-engine --help` runs without touching TRT."""
    from ptychoml.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Build a TensorRT engine" in captured.out


def test_predict_cli_help_runs(capsys):
    """`ptychoml-predict --help` runs without touching TRT."""
    from ptychoml.cli import predict_main

    with pytest.raises(SystemExit) as exc_info:
        predict_main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Run PtychoViT inference" in captured.out


def test_predict_cli_missing_dataset(tmp_path):
    """predict_main reports an error when the dataset key is not in the file."""
    import h5py
    import numpy as np

    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("other_key", data=np.zeros((2, 8, 8), dtype=np.float32))

    from ptychoml.cli import predict_main

    ret = predict_main([
        "--engine", "dummy.engine",
        "--data", str(h5_path),
        "--output", str(tmp_path / "out.h5"),
        "--dataset", "diffamp",
    ])
    assert ret == 1


def test_reshape_output_flat():
    """Pure utility — reshape flat TRT output to B,H,W or B,2,H,W."""
    import numpy as np

    from ptychoml.trt import reshape_output_flat

    # Single-channel output
    flat = np.arange(4 * 8 * 8, dtype=np.float32)
    out = reshape_output_flat(flat, batch_size=4, height=8, width=8)
    assert out.shape == (4, 8, 8)

    # Dual-channel (amp + phase)
    flat = np.arange(4 * 2 * 8 * 8, dtype=np.float32)
    out = reshape_output_flat(flat, batch_size=4, height=8, width=8)
    assert out.shape == (4, 2, 8, 8)


def test_reshape_output_flat_invalid_size():
    import numpy as np

    from ptychoml.trt import reshape_output_flat

    flat = np.arange(123, dtype=np.float32)
    with pytest.raises(ValueError, match="Unexpected output size"):
        reshape_output_flat(flat, batch_size=4, height=8, width=8)


class TestInferenceInit:
    """Test the stateful session __init__ without touching TRT."""

    def test_basic_init(self):
        from ptychoml import PtychoViTInference

        session = PtychoViTInference(
            engine_path="/nonexistent/model.engine",
            gpu=0,
            data_is_shifted=False,
        )
        assert session.engine_path == "/nonexistent/model.engine"
        assert session.gpu == 0
        assert session._data_is_shifted is False
        assert session._initialized is False

    def test_defaults(self):
        from ptychoml import PtychoViTInference

        session = PtychoViTInference(engine_path="m.engine")
        assert session.gpu == 0
        assert session._data_is_shifted is False

    def test_context_manager_cleanup_safe(self):
        """__exit__ should not raise even if predict() was never called."""
        from ptychoml import PtychoViTInference

        with PtychoViTInference(engine_path="m.engine") as session:
            assert session._initialized is False
        # session.cleanup() was called via __exit__; should be idempotent
        session.cleanup()
