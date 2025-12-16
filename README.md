## Triton Inference Server Rust Client for Python

This repository provides a high‑performance Python client for the NVIDIA Triton Inference Server,
implemented in Rust and exposed to Python via [PyO3](https://pyo3.rs/) and [maturin](https://github.com/PyO3/maturin).

It is designed to be:

- **Fast**: core networking and protobuf handling are written in Rust.
- **Python‑friendly**: exposes Rust gRPC/protobuf types directly to Python.
- **Shared‑memory aware**: supports high‑throughput data transfer via `multiprocessing.shared_memory` (see examples).
- **Modern**: targets Python ≥ 3.12 and is tested on Linux (x86_64, aarch64) and Windows (x86_64).

---

## Project layout

- `src/` – Rust crate `triton_client`
  - gRPC/protobuf bindings generated from Triton’s `.proto` files
  - `Client` type wrapping the gRPC client
  - All protobuf request/response types (e.g. `ModelInferRequest`, `ModelInferResponse`, …)
- `py_vec_types/` – Rust crate providing Python‑visible `List*` helper types for zero‑copy / low‑copy Vec bindings
- `triton_client/` – `.pyi` type stubs for the Rust extension module

The published wheel on PyPI is the Rust extension module `triton_client`.

---

## Features

- **Rust‑backed gRPC client**
  - Uses `tonic`, `prost`, and the official Triton protobuf definitions.
  - Exposes a `Client` type to Python as `triton_client.Client`.

- **Rich type surface**
  - All key Triton request/response protobuf messages are available directly in Python
    (for example `ModelInferRequest`, `ModelInferResponse`, `ModelMetadataRequest`, `ModelConfig`, …).

- **High‑throughput data handling**
  - Numpy arrays are converted to Triton tensor contents via Rust using zero‑copy / low‑copy `replace_*_contents` helpers.
  - Examples under `examples/` show how to combine this with `multiprocessing.shared_memory` for very fast data paths.

---

## Installation

### From PyPI

Once this project is published to PyPI you will be able to install it via:

```bash
pip install triton_client
```

This installs:

- the Rust extension module `triton_client`

### From source

You need:

- Rust toolchain (stable)
- Python 3.12+
- `maturin` (build backend)

Build a development wheel and install it into the current environment:

```bash
pip install maturin

# Build and install in editable mode (local virtualenv recommended)
maturin develop --release
```

Or build wheels only:

```bash
maturin build --release
```

The wheels will appear under `target/wheels/`.

---

## Quick start

The Rust crate exports its types to Python via the `triton_client` extension module.
A simple pattern is:

```python
import numpy as np
import triton_client

# Create Rust client (gRPC)
client = triton_client.Client(url="localhost:8001", access_token=None)

# Prepare tensor contents
data = np.random.randn(1, 3, 224, 224).astype(np.float32)
tensor_contents = triton_client.InferTensorContents()
data = triton_client.ListF32.from_array(data)
tensor_contents.Replace_fp32_contents(data))

# Build input tensor
input_tensor = triton_client.InferInputTensor(
    name="input",
    datatype="FP32",
    shape=list(data.shape),
    parameters={},
    contents=tensor_contents,
)

# Build requested output tensor
requested_output = triton_client.InferRequestedOutputTensor(
    name="output",
    parameters={},
)

# Build inference request
request = triton_client.ModelInferRequest(
    model_name="your_model",
    model_version="",
    id="",
    parameters={},
    inputs=[input_tensor],
    outputs=[requested_output],
)

# Run inference
response = client.model_infer(request)
# Access outputs (see examples/ for more helpers)
print("Model infer response id:", response.id)
```

More complete Python examples (including shared memory usage and a `tritonclient`‑style wrapper)
are available under the `examples/` directory.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

