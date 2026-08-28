"""Optional ONNX export of ``gen_R`` from a slim checkpoint.

Not used by training or inference -- the native ``.pt`` is the primary format.
This exists so deployment is never blocked on a format decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from reality.training.checkpoint import load


class OnnxUnavailable(RuntimeError):
    """Raised when the optional onnx dependency is missing."""


def require_onnx():
    """Import the ONNX toolchain, or explain how to get it instead of crashing."""
    try:
        import onnx
    except ImportError as exc:
        raise OnnxUnavailable(
            "ONNX export needs the optional 'onnx' package: pip install onnx "
            "(and onnxruntime to verify the exported model)."
        ) from exc
    try:
        import onnxscript  # torch's current exporter builds the graph through this
    except ImportError as exc:
        raise OnnxUnavailable(
            "ONNX export also needs 'onnxscript' for this torch version: "
            "pip install onnx onnxscript"
        ) from exc
    return onnx


def export_gen_r(checkpoint_path: Union[str, Path], output_path: Union[str, Path],
                 image_shape: Tuple[int, int] = (64, 1024),
                 in_channels: Optional[int] = None,
                 opset: int = 17, dynamic_batch: bool = True) -> Path:
    """Export a checkpoint's ``gen_R`` to ONNX and return the written path."""
    require_onnx()
    from reality.models.picgan_adapter import PicganAdapter

    checkpoint = load(checkpoint_path)
    channels = in_channels or checkpoint.in_channels_s

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = PicganAdapter(workspace=output_path.parent, device="cpu")
    model.build_model(channels, int(checkpoint.metadata.get("in_channels_r", 1) or 1))
    model.gen_R.load_state_dict(checkpoint.state["gen_R"])
    model.use_statistics(checkpoint.stats)
    model.gen_R.eval()

    dummy = torch.zeros(1, channels, *image_shape)
    torch.onnx.export(
        model.gen_R, dummy, str(output_path),
        input_names=["source_stack"], output_names=["intensity"],
        opset_version=opset,
        dynamic_axes={"source_stack": {0: "batch"}, "intensity": {0: "batch"}}
        if dynamic_batch else None,
    )
    return output_path


def verify(onnx_path: Union[str, Path], reference_model, image_shape=(64, 1024),
           in_channels: int = 3, atol: float = 1e-4) -> float:
    """Compare an exported model against the torch one; returns the max difference."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise OnnxUnavailable(
            "verifying an ONNX export needs onnxruntime: pip install onnxruntime"
        ) from exc

    sample = torch.randn(1, in_channels, *image_shape)
    reference_model.eval()
    with torch.no_grad():
        expected = reference_model(sample).cpu().numpy()
    session = onnxruntime.InferenceSession(str(onnx_path),
                                           providers=["CPUExecutionProvider"])
    actual = session.run(None, {"source_stack": sample.numpy()})[0]
    return float(abs(expected - actual).max())
