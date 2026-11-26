//! What can go wrong: batchplan's errors, of which a look returns one kind, `Error::Input`, with
//! what was wrong and the value given.

pub(crate) use batchplan::error::Result;

/// Returns `Error::Input` with a formatted message unless the condition holds.
macro_rules! ensure_input {
    ($cond:expr, $($message:tt)+) => {
        let holds: bool = $cond;
        if !holds {
            return Err(::batchplan::Error::Input(format!($($message)+)));
        }
    };
}

/// `Error::Input` with a formatted message.
macro_rules! input {
    ($($message:tt)+) => {
        ::batchplan::Error::Input(format!($($message)+))
    };
}

pub(crate) use ensure_input;
pub(crate) use input;
