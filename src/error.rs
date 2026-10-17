//! What can go wrong, in kinds a host can act on. Messages say what was wrong and where.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The call cannot work with what it was given: array shapes, world indices, worlds uploaded
    /// to another device, options out of range, degenerate geometry.
    #[error("{0}")]
    Input(String),
    /// A robot, scene or collision-model file could not be read or understood.
    #[error("{path}: {message}")]
    Load { path: PathBuf, message: String },
    /// No GPU can run the kernels, or the GPU failed while working.
    #[error("{0}")]
    Gpu(String),
    /// The CPU device could not start its threads.
    #[error("{0}")]
    Threads(String),
    /// A trajectory breaks a joint range or a motion limit ([`crate::Trajectory::check`]).
    #[error("{0}")]
    Unsafe(String),
    /// Writing a file (a dataset or a collision model) failed.
    #[error("{0}")]
    Write(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Typed code reads no files; its I/O is writing exports.
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Write(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Write(e.to_string())
    }
}

/// Returns `Error::Input` with a formatted message unless the condition holds.
macro_rules! ensure_input {
    ($cond:expr, $($message:tt)+) => {
        let holds: bool = $cond;
        if !holds {
            return Err($crate::error::Error::Input(format!($($message)+)));
        }
    };
}

/// `Error::Input` with a formatted message.
macro_rules! input {
    ($($message:tt)+) => {
        $crate::error::Error::Input(format!($($message)+))
    };
}

pub(crate) use {ensure_input, input};
