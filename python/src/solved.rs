//! What results share: `Solved`, and arrays built once per result.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

/// A solved problem of a batch: its index in the batch, the problem, and its answer.
#[pyclass(module = "batchplan", frozen, get_all)]
pub(crate) struct Solved {
    /// The problem's index in the batch.
    pub(crate) index: usize,
    /// The problem.
    pub(crate) problem: Py<PyAny>,
    /// Its answer: a configuration, or a path's control points `(points, dof)`.
    pub(crate) solution: Py<PyAny>,
}

#[pymethods]
impl Solved {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Solved(index={}, problem={}, solution={})",
            self.index,
            self.problem.bind(py).repr()?,
            self.solution.bind(py).repr()?
        ))
    }
}

/// An array of a result's, built on first use and the same object after: a loop that reads
/// `result.q[i]` copies the result's data once, not once per read.
pub(crate) struct Cached(PyOnceLock<Py<PyAny>>);

impl Default for Cached {
    fn default() -> Self {
        Self(PyOnceLock::new())
    }
}

impl Cached {
    pub(crate) fn get<'py>(
        &self,
        py: Python<'py>,
        build: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let built = self.0.get_or_try_init(py, || build().map(Bound::unbind))?;
        Ok(built.bind(py).clone())
    }
}
