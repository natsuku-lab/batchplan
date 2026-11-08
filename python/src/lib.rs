//! Python bindings for batchplan: the `batchplan._native` module the `batchplan` package
//! re-exports. Each module here binds the library module of its name.

mod device;
mod error;
mod field;
mod held;
mod ik;
mod plan;
mod robot;
mod solved;
mod task;
mod timing;
mod tool_path;
mod world;

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    error::register(m)?;
    m.add_class::<world::Pose>()?;
    m.add_class::<world::Distance>()?;
    m.add_class::<world::Obstacle>()?;
    m.add_class::<world::Touch>()?;
    m.add_class::<world::World>()?;
    m.add_class::<robot::SphereGeometry>()?;
    m.add_class::<robot::JointLimits>()?;
    m.add_class::<robot::SphereOptions>()?;
    m.add_class::<robot::CollisionModel>()?;
    m.add_class::<robot::RobotOptions>()?;
    m.add_class::<robot::Robot>()?;
    m.add_class::<device::Contact>()?;
    m.add_class::<device::Device>()?;
    m.add_class::<device::Worlds>()?;
    m.add_class::<solved::Solved>()?;
    ik::register(m)?;
    plan::register(m)?;
    timing::register(m)?;
    held::register(m)?;
    tool_path::register(m)?;
    task::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
