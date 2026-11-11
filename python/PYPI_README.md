# batches

Python bindings for [batchplan](https://github.com/natsuku-lab/batchplan): batched, collision-aware
robot motion generation on GPUs and CPU threads. Inverse kinematics, motion planning, tool paths
and pick-and-place tasks take many problems and worlds in one call, with NumPy arrays in and out.

```bash
pip install --pre batches
```

The distribution is named `batches`; the import is `batchplan`:

```python
import batchplan as bp
```

Python 3.10 or newer is required. Source installations also need a Rust toolchain.
The [Python guide](https://github.com/natsuku-lab/batchplan/blob/main/docs/python.md) covers
installation, examples and which parts of the Rust library are bound.
