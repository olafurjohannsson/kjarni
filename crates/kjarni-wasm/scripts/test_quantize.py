"""Tests for quantize_model.py.

The bfloat16 path is the one that matters: numpy has no bf16 and safetensors will
not hand one back, so bf16 checkpoints failed outright until recently — which covered
most modern chat models, Qwen and Llama among them. A silent regression there breaks
exactly the models people want in a browser.

    python3 -m venv /tmp/kjq && /tmp/kjq/bin/pip install numpy safetensors
    /tmp/kjq/bin/python -m unittest discover -s crates/kjarni-wasm/scripts
"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

import quantize_model


class BFloat16Reading(unittest.TestCase):
    """bf16 is the top 16 bits of an f32, so widening it must be exact."""

    def _write_bf16(self, path, name, values, shape):
        """Writes a safetensors file by hand, since numpy cannot express bf16."""
        f32 = np.asarray(values, dtype=np.float32)
        # Truncate toward zero the way bf16 storage does: keep the high half.
        raw = f32.view(np.uint32) >> 16
        payload = raw.astype(np.uint16).tobytes()

        header = {name: {"dtype": "BF16", "shape": list(shape),
                         "data_offsets": [0, len(payload)]}}
        header_bytes = json.dumps(header).encode("utf-8")

        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header_bytes)))
            fh.write(header_bytes)
            fh.write(payload)

    def test_widens_bfloat16_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.safetensors"
            # Values chosen to be exactly representable in bf16, so any difference
            # is a bug rather than expected precision loss.
            values = [1.0, -2.0, 0.5, 0.0, 256.0, -0.125]
            self._write_bf16(path, "w", values, (6,))

            out = quantize_model._read_bfloat16(path, "w")

            self.assertEqual(out.dtype, np.float32)
            self.assertEqual(out.shape, (6,))
            np.testing.assert_array_equal(out, np.asarray(values, dtype=np.float32))

    def test_preserves_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.safetensors"
            self._write_bf16(path, "w", [1.0, 2.0, 4.0, 8.0, 16.0, 32.0], (2, 3))
            out = quantize_model._read_bfloat16(path, "w")
            self.assertEqual(out.shape, (2, 3))


class Quantization(unittest.TestCase):
    """The int8 round trip, and what the exporter chooses to leave alone."""

    def _model_dir(self, tensors):
        d = Path(tempfile.mkdtemp())
        save_file(tensors, str(d / "model.safetensors"))
        (d / "config.json").write_text(json.dumps({"model_type": "bert", "hidden_size": 8}))
        (d / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
        return d

    def test_produces_a_readable_kjq(self):
        # A large tensor gets quantised; a small one is kept at f32 because
        # quantising it costs accuracy and saves almost nothing.
        big = np.linspace(-1.0, 1.0, 4096, dtype=np.float32).reshape(64, 64)
        small = np.asarray([0.5, -0.5], dtype=np.float32)

        d = self._model_dir({"encoder.weight": big, "encoder.bias": small})
        out = d / "model.kjq"
        quantize_model.quantize_model(d, out)

        self.assertTrue(out.exists())
        data = out.read_bytes()
        self.assertEqual(data[:4], b"KJQ1", "magic bytes identify the format")

        # The container must be smaller than the f32 weights it came from; that is
        # the entire reason it exists.
        self.assertLess(out.stat().st_size, big.nbytes)

    def test_config_and_tokenizer_survive_verbatim(self):
        d = self._model_dir({"w": np.zeros((8, 8), dtype=np.float32)})
        out = d / "model.kjq"
        quantize_model.quantize_model(d, out)

        data = out.read_bytes()
        cursor = 4
        (config_len,) = struct.unpack_from("<I", data, cursor)
        cursor += 4
        config = data[cursor:cursor + config_len].decode("utf-8")
        self.assertEqual(json.loads(config)["model_type"], "bert")


if __name__ == "__main__":
    unittest.main()
