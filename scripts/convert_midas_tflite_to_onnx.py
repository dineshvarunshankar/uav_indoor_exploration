"""Convert the deployed W8A8 MiDaS TFLite model to a batched ONNX graph.

Needs tf2onnx and tensorflow, which are not part of the training environment:

    python -m venv /tmp/midas_convert
    /tmp/midas_convert/bin/pip install tf2onnx "tensorflow-cpu<2.20" onnx
    /tmp/midas_convert/bin/python scripts/convert_midas_tflite_to_onnx.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper

TFLITE_PATH = Path("/home/ubuntu/starling/AI_VOXL2/tflite/MiDaS/midas.tflite")
ONNX_PATH = Path("/home/ubuntu/starling/AI_VOXL2/onnx/MiDaS/midas.onnx")


def make_batch_dynamic(model: onnx.ModelProto) -> onnx.ModelProto:
    """The TFLite graph bakes batch 1 into its input, output, and one Reshape constant."""

    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    for output in model.graph.output:
        output.type.tensor_type.shape.dim[0].dim_param = "N"

    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Reshape" or node.input[1] not in initializers:
            continue
        tensor = initializers[node.input[1]]
        shape = numpy_helper.to_array(tensor).copy()
        if shape.size and shape[0] == 1:
            shape[0] = -1
            tensor.CopyFrom(numpy_helper.from_array(shape, node.input[1]))

    del model.graph.value_info[:]
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, default=TFLITE_PATH)
    parser.add_argument("--output", type=Path, default=ONNX_PATH)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = args.output.with_suffix(".raw.onnx")
    subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert", "--tflite", str(args.tflite),
         "--output", str(staged), "--opset", str(args.opset)],
        check=True,
    )
    onnx.save(make_batch_dynamic(onnx.load(staged)), args.output)
    staged.unlink()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
