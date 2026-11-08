//! `batchplan::Error`'s kinds as exception classes under one base, `batchplan.Error`, so a host
//! catches all of them or one kind, as it matches on them in Rust.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    batchplan,
    Error,
    PyException,
    "Every error batchplan reports; its subclasses are `batchplan::Error`'s kinds."
);
create_exception!(
    batchplan,
    InputError,
    Error,
    "The call cannot work with what it was given: array shapes, world indices, worlds uploaded to \
     another device, options out of range, degenerate geometry."
);
create_exception!(
    batchplan,
    LoadError,
    Error,
    "A robot, scene or collision-model file could not be read or understood."
);
create_exception!(
    batchplan,
    GpuError,
    Error,
    "No GPU can run the kernels, or the GPU failed while working."
);
create_exception!(
    batchplan,
    ThreadsError,
    Error,
    "The CPU device could not start its threads."
);
create_exception!(
    batchplan,
    UnsafeError,
    Error,
    "A trajectory breaks a joint range or a motion limit."
);
create_exception!(
    batchplan,
    WriteError,
    Error,
    "Writing a file (a dataset or a collision model) failed."
);

/// A batchplan result as a Python one, its error raised as its kind's exception.
pub(crate) trait OrRaise<T> {
    fn or_raise(self) -> PyResult<T>;
}

impl<T> OrRaise<T> for batchplan::error::Result<T> {
    fn or_raise(self) -> PyResult<T> {
        self.map_err(|e| {
            let message = e.to_string();
            match e {
                batchplan::Error::Input(_) => InputError::new_err(message),
                batchplan::Error::Load { .. } => LoadError::new_err(message),
                batchplan::Error::Gpu(_) => GpuError::new_err(message),
                batchplan::Error::Threads(_) => ThreadsError::new_err(message),
                batchplan::Error::Unsafe(_) => UnsafeError::new_err(message),
                batchplan::Error::Write(_) => WriteError::new_err(message),
                _ => Error::new_err(message),
            }
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("Error", py.get_type::<Error>())?;
    m.add("InputError", py.get_type::<InputError>())?;
    m.add("LoadError", py.get_type::<LoadError>())?;
    m.add("GpuError", py.get_type::<GpuError>())?;
    m.add("ThreadsError", py.get_type::<ThreadsError>())?;
    m.add("UnsafeError", py.get_type::<UnsafeError>())?;
    m.add("WriteError", py.get_type::<WriteError>())?;
    Ok(())
}
