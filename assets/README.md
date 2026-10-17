# Test robots

Each folder keeps its upstream licence. Only collision geometry is included; visual meshes the
descriptions reference are not.

| Folder | Source | Licence | Changes |
|---|---|---|---|
| `franka/` | `franka_panda.urdf` from franka_ros 0.7.0; collision meshes (`meshes/collision/*.obj`) from NVlabs/curobo at `78fd485`, converted from franka_ros | Apache-2.0 (`franka/LICENSE`) | none |
| `franka/panda_collision.json` | Collision spheres, self-collision buffers and ignore pairs from NVlabs/curobo `franka.yml` | Apache-2.0, © NVIDIA | converted to batchplan's collision-model format |
| `ur5e/ur_description/` | UniversalRobots/Universal_Robots_ROS2_Description at `6662e15` (ur5e collision meshes, `package.xml`) | BSD-3-Clause (`ur5e/ur_description/LICENSE`) | `urdf/ur5e.urdf` expanded from `urdf/ur.urdf.xacro ur_type:=ur5e name:=ur5e` |
| `robotiq_2f85/robotiq_description/` | PickNikRobotics/ros2_robotiq_gripper at `a74d007` (2F-85 collision meshes, `package.xml`) | BSD-3-Clause (`robotiq_2f85/robotiq_description/LICENSE`) | `urdf/robotiq_2f_85.urdf` expanded from `urdf/robotiq_2f_85_gripper.urdf.xacro` |
| `so101/` | TheRobotStudio/SO-ARM100 at `a758567` (`Simulation/SO101/so101_new_calib.urdf`) | Apache-2.0 (`so101/LICENSE`) | Each mesh replaced by its convex hull (`assets/*_hull.stl`, about 0.15 MB instead of 15 MB): vertices snapped to a 2 mm grid, then every face pushed out until the hull contains all original vertices. The URDF points at the hulls. |

Two downloads stay out of the repository, in `data/` (git-ignored): one more robot, from `scripts/fetch_rebot.sh`, and the Panda's visual meshes, from `scripts/fetch_panda_meshes.sh`:

| Folder | Source | Licence | Changes |
|---|---|---|---|
| `data/rebot/DM/` | Seeed-Projects/reBot-DevArm at `9ac6169` (`Rebot_Arm_description/DM`: URDF, visual and collision meshes) | CERN-OHL-W-2.0 (`data/rebot/LICENSE`) | `urdf/ReBot_Arm_DM_tcp.urdf` adds a `tcp` link, the grasp point between the fingers 2 cm behind their tips |
| `data/franka_visual/` | frankaemika/franka_ros 0.7.0 (`franka_description/meshes/visual`), by `scripts/fetch_panda_meshes.sh`, for the Rerun demos | Apache-2.0 (`data/franka_visual/LICENSE`) | none |

The UR5e and 2F-85 sit inside their ROS packages, so their `package://` mesh paths resolve by
searching upward for `package.xml`.

## MJCF and USD

| Folder | Source | Licence | Changes |
|---|---|---|---|
| `menagerie/franka_emika_panda/` | google-deepmind/mujoco_menagerie at `0059d43` (`panda.xml`, `scene.xml`) | Apache-2.0 (its `LICENSE`) | none; meshes not included |
| `menagerie/universal_robots_ur5e/` | mujoco_menagerie at `0059d43` (`ur5e.xml`) | BSD-3-Clause (its `LICENSE`) | none; meshes not included |
| `menagerie/robotiq_2f85/` | mujoco_menagerie at `0059d43` (`2f85.xml`) | BSD-2-Clause (its `LICENSE`) | none; meshes not included |
| `newton/universal_robots_ur5e/` | newton-physics/newton-assets at `a054754` | BSD-3-Clause (its `LICENSE`) | `GeometryLibrary.usdc` and `MaterialsLibrary.usdc` (visual meshes) not included; collision geometry is primitives |
| `newton/robotiq_2f85_v4/` | newton-assets at `a054754` | BSD-2-Clause (its `LICENSE`) | `README.md` not included |
| `franka/usd/` | Converted from `franka/franka_panda.urdf` with NVIDIA's urdf-usd-converter 0.3.3 | Apache-2.0 (`franka/LICENSE`) | visual meshes absent (as in the URDF assets) |
| `usd/` | Written for batchplan's tests | MIT OR Apache-2.0 | |

