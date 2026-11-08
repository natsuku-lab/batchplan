//! How values cross between Rust and Python, and the frozen classes that hold batchplan's value
//! types.
//!
//! A value class holds the Rust value itself, so passing it to a call costs a clone, not a
//! conversion. Its fields are read through getters that return immutable views (tuples, read-only
//! arrays), and changed by `replace(**changes)`, as Rust's struct update syntax does: a getter's
//! result can't be mutated in the belief that the value changes.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Duration;

use batchplan::glam::{Quat, Vec3};
use numpy::{AllowTypeChange, Element, PyArray1, PyArrayLike1, PyArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyString, PyTuple};

use crate::error::InputError;

/// A Rust value as Python sees it, both ways.
pub(crate) trait Field: Sized {
    /// The value as a Python object.
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
    /// The value from a Python object.
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self>;
}

macro_rules! plain_field {
    ($($ty:ty),*) => {$(
        impl Field for $ty {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.clone().into_bound_py_any(py)
            }
            fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                value.extract()
            }
        }
    )*};
}

plain_field!(bool, u32, u64, usize, String, PathBuf);

/// Python floats are f64: an f32 goes out as the shortest decimal that reads back as it, so
/// `IkOptions().damping == 0.02` holds. Every f32 the package returns goes through here.
pub(crate) fn shortest(x: f32) -> f64 {
    x.to_string()
        .parse()
        .expect("an f32's decimal form, `inf` and `NaN` included, parses as an f64")
}

impl Field for f32 {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        shortest(*self).into_bound_py_any(py)
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let x: f64 = value.extract()?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "batchplan computes in f32; the nearest f32 is the value meant"
        )]
        Ok(x as f32)
    }
}

/// Seconds.
impl Field for Duration {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.as_secs_f64().into_bound_py_any(py)
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let seconds: f64 = value.extract()?;
        Duration::try_from_secs_f64(seconds).map_err(|_| {
            InputError::new_err(format!(
                "need finite seconds ≥ 0 that fit a duration, got {seconds}"
            ))
        })
    }
}

impl<T: Field> Field for Option<T> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Some(v) => v.to_py(py),
            None => Ok(py.None().into_bound(py)),
        }
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.is_none() {
            Ok(None)
        } else {
            T::from_py(value).map(Some)
        }
    }
}

/// Maps go out as read-only views (`types.MappingProxyType`), so they can't be changed in place,
/// and come in from any mapping.
macro_rules! map_field {
    ($($map:ident),*) => {$(
        impl<T: Field> Field for $map<String, T> {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let dict = PyDict::new(py);
                for (key, value) in self {
                    dict.set_item(key, value.to_py(py)?)?;
                }
                py.import("types")?.getattr("MappingProxyType")?.call1((dict,))
            }
            fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                let items = value.cast::<PyMapping>()?.items()?;
                let entries = items.iter().map(|item| {
                    let (k, v): (String, Bound<'_, PyAny>) = item.extract()?;
                    Ok((k, T::from_py(&v)?))
                });
                entries.collect()
            }
        }
    )*};
}

map_field!(HashMap, BTreeMap);

/// Two values from any two-item iterable but a string, out as a tuple.
impl<A: Field, B: Field> Field for (A, B) {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyTuple::new(py, [self.0.to_py(py)?, self.1.to_py(py)?])?.into_any())
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let items = items(value)?;
        let [a, b] = <[_; 2]>::try_from(items).map_err(|items: Vec<_>| {
            InputError::new_err(format!("need a pair, got {} items", items.len()))
        })?;
        Ok((A::from_py(&a)?, B::from_py(&b)?))
    }
}

/// The items of any iterable but a string, which would be taken letter by letter.
pub(crate) fn items<'py>(value: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if value.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("need a sequence, got a string"));
    }
    value.try_iter()?.collect()
}

/// Sequences go out as tuples, so they can't be appended to in place, and come in from any
/// iterable but a string.
macro_rules! tuple_field {
    ($($ty:ty),*) => {$(
        impl $crate::field::Field for Vec<$ty> {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let items: Vec<Bound<'py, PyAny>> =
                    self.iter().map(|v| v.to_py(py)).collect::<PyResult<_>>()?;
                Ok(pyo3::types::PyTuple::new(py, items)?.into_any())
            }
            fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                let items = $crate::field::items(value)?;
                items.iter().map(<$ty as $crate::field::Field>::from_py).collect()
            }
        }
    )*};
}

pub(crate) use tuple_field;

tuple_field!(usize, String, PathBuf, (String, String));

/// A read-only array of `values` (`float32`, `bool`), shaped `shape`.
pub(crate) fn array<'py, T: Element + Copy>(
    py: Python<'py>,
    values: &[T],
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let flat = PyArray1::from_slice(py, values);
    read_only(flat.reshape(shape)?.into_any())
}

/// `array`, made read-only: copies of a value's data that can't be changed in the belief that
/// the value changes.
pub(crate) fn read_only(array: Bound<'_, PyAny>) -> PyResult<Bound<'_, PyAny>> {
    let flags = PyDict::new(array.py());
    flags.set_item("write", false)?;
    array.call_method("setflags", (), Some(&flags))?;
    Ok(array)
}

/// Floats from any array-like (a NumPy array of any numeric type, a list), flattened in C order.
pub(crate) fn floats(value: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let np = value.py().import("numpy")?;
    let flat = np.call_method1("ravel", (value,))?;
    let array: PyArrayLike1<'_, f32, AllowTypeChange> = flat.extract()?;
    Ok(array.as_array().iter().copied().collect())
}

/// Floats from any array-like, with exactly `n` of them: an `InputError` naming `what` if not.
pub(crate) fn floats_of(value: &Bound<'_, PyAny>, n: usize, what: &str) -> PyResult<Vec<f32>> {
    let v = floats(value)?;
    if v.len() == n {
        Ok(v)
    } else {
        Err(InputError::new_err(format!(
            "{what} needs {n} numbers, got {}",
            v.len()
        )))
    }
}

/// A configuration or other vector: a 1-D array.
impl Field for Vec<f32> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        array(py, self, &[self.len()])
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        floats(value)
    }
}

/// `(x, y, z)`.
impl Field for Vec3 {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        array(py, &self.to_array(), &[3])
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Vec3::from_slice(&floats_of(value, 3, "a point")?))
    }
}

/// A unit quaternion, scalar last: `(x, y, z, w)`, as ROS, SciPy and glam order it.
impl Field for Quat {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        array(py, &self.to_array(), &[4])
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let q = floats_of(value, 4, "a quaternion (x, y, z, w)")?;
        Ok(Quat::from_slice(&q))
    }
}

/// Spheres, `(n, 4)`: `x, y, z, radius`, from any array-like of that shape.
impl Field for Vec<[f32; 4]> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        array(py, self.as_flattened(), &[self.len(), 4])
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape: Vec<usize> = value
            .py()
            .import("numpy")?
            .call_method1("shape", (value,))?
            .extract()?;
        // `[]` reads as shape (0,): no spheres.
        if !matches!(shape.as_slice(), [_, 4] | [0]) {
            return Err(InputError::new_err(format!(
                "spheres need the shape (n, 4), as x, y, z, radius each; got {shape:?}"
            )));
        }
        Ok(floats(value)?.as_chunks::<4>().0.to_vec())
    }
}

/// `(lower, upper)`.
impl Field for [f32; 2] {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        (self[0], self[1]).to_py(py)
    }
    fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (lower, upper) = <(f32, f32)>::from_py(value)?;
        Ok([lower, upper])
    }
}

/// Declares a frozen Python class around a batchplan value type: a getter per field (its doc the
/// property's), `replace(**changes)`, a `repr` in constructor form, and the [`Field`] conversion
/// that lets other classes hold it. Every field of the Rust struct is named, as a getter or under
/// `hidden`, so a field added to it fails to build here until the class covers it.
macro_rules! value_class {
    (
        $(#[$meta:meta])*
        $name:ident($($inner:ident)::+) {
            $( $(#[$fmeta:meta])* $field:ident: $ty:ty ),* $(,)?
        }
        $( hidden { $( $hidden:ident ),* $(,)? } )?
        $( methods { $($methods:tt)* } )?
    ) => {
        $(#[$meta])*
        #[pyclass(module = "batchplan", frozen)]
        #[derive(Clone)]
        pub(crate) struct $name(pub(crate) $($inner)::+);

        impl $name {
            /// `inner` with `changes` (field name to value) applied.
            fn changed(
                mut inner: $($inner)::+,
                changes: Option<&Bound<'_, pyo3::types::PyDict>>,
            ) -> PyResult<$($inner)::+> {
                let Some(changes) = changes else { return Ok(inner) };
                for (key, value) in changes.iter() {
                    let key: String = key.extract()?;
                    match key.as_str() {
                        $( stringify!($field) => {
                            inner.$field = <$ty as $crate::field::Field>::from_py(&value)?;
                        } )*
                        _ => {
                            let _ = &value;
                            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                "{} has no field '{key}'", stringify!($name)
                            )));
                        }
                    }
                }
                Ok(inner)
            }

            /// Names every field, so a new one fails to build until it is covered.
            #[expect(dead_code, reason = "only compiled, to check the fields are all named")]
            fn every_field(v: &$($inner)::+) {
                let $($inner)::+ { $( $field: _, )* $( $( $hidden: _, )* )? } = v;
            }
        }

        #[pymethods]
        impl $name {
            $(
                $(#[$fmeta])*
                #[getter]
                fn $field<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                    $crate::field::Field::to_py(&self.0.$field, py)
                }
            )*

            /// A copy with the fields given changed.
            #[pyo3(signature = (**changes))]
            fn replace(&self, changes: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
                Ok(Self(Self::changed(self.0.clone(), changes)?))
            }

            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let fields: Vec<String> = vec![$(
                    format!(
                        "{}={}",
                        stringify!($field),
                        $crate::field::Field::to_py(&self.0.$field, py)?.repr()?
                    ),
                )*];
                Ok(format!("{}({})", stringify!($name), fields.join(", ")))
            }

            $( $($methods)* )?
        }

        impl $crate::field::Field for $($inner)::+ {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                Ok(Bound::new(py, $name(self.clone()))?.into_any())
            }
            fn from_py(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(value.cast::<$name>()?.get().0.clone())
            }
        }
    };
}

pub(crate) use value_class;
