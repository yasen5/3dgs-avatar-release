#!/usr/bin/env python3
"""CUDA-only BF16 Sapiens depth worker for the preparation script."""
from __future__ import annotations

import argparse
import os
import pickle
import struct
import sys
from pathlib import Path

import torch

# Do not recursively launch another worker when importing SapiensDepth.
os.environ.pop("SAPIENS_DEPTH_PYTHON", None)

from prepare_sapiens_ga_avatar import SapiensDepth, _read_exact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the BF16 depth worker requires CUDA")
    model = SapiensDepth(args.checkpoint, "cuda")
    reader = sys.stdin.buffer
    writer = sys.stdout.buffer
    while True:
        header = reader.read(8)
        if not header:
            break
        if len(header) != 8:
            raise RuntimeError("truncated CUDA depth worker request")
        size = struct.unpack("!Q", header)[0]
        crop = pickle.loads(_read_exact(reader, size))
        try:
            result = model.predict(crop)
            response = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # return the traceback-bearing error to parent
            response = pickle.dumps({"error": repr(exc)}, protocol=pickle.HIGHEST_PROTOCOL)
        writer.write(struct.pack("!Q", len(response)))
        writer.write(response)
        writer.flush()
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
