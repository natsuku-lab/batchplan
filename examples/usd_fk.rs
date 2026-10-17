//! Forward kinematics of an OpenUSD robot, for scripts/validate_usd.py.
//!
//! Reads `{"file", "links", "samples", "seed"}` as JSON on stdin and prints
//! `{"joints", "q": [[..]], "poses": [[[x, y, z, qx, qy, qz, qw], ..], ..]}`: random joint values
//! within the limits and the pose of each named link at each of them.

use std::collections::HashMap;

use anyhow::{Context, Result};
use batchplan::rng::Rng;
use batchplan::*;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let input: Value = serde_json::from_reader(std::io::stdin())?;
    let file = input["file"].as_str().context("file")?;
    let links: Vec<String> = serde_json::from_value(input["links"].clone())?;
    let samples = input["samples"].as_u64().unwrap_or(20) as usize;
    let variants: HashMap<String, String> = serde_json::from_value(input["variants"].clone()).unwrap_or_default();
    // Kinematics only: no collision model to fit.
    let options = RobotOptions {
        collision_model: Some(CollisionModel::default()),
        variants: variants.into_iter().collect(),
        ..Default::default()
    };
    let robot = Robot::load(file, &options)?;
    let mut rng = Rng::new(input["seed"].as_u64().unwrap_or(1));
    let mut qs = vec![];
    let mut poses = vec![];
    for _ in 0..samples {
        let q: Vec<f32> = (0..robot.dof()).map(|j| rng.range(robot.lower()[j], robot.upper()[j])).collect();
        let at: Vec<Value> = links
            .iter()
            .map(|l| {
                robot.link_pose(&q, l).map_or(Value::Null, |p| {
                    json!([
                        p.position.x,
                        p.position.y,
                        p.position.z,
                        p.rotation.x,
                        p.rotation.y,
                        p.rotation.z,
                        p.rotation.w
                    ])
                })
            })
            .collect();
        poses.push(at);
        qs.push(q);
    }
    println!("{}", json!({"joints": robot.joint_names(), "q": qs, "poses": poses}));
    Ok(())
}
