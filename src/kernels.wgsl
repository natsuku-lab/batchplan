// Batched kinematics, collision cost/gradient, IK and trajectory optimization.
// Mirrors src/cpu.rs function for function. gpu.rs prepends the constants (obstacle kinds,
// MAX_HISTORY, LINE_STEPS and line_search) and the shared structs (Params, Link, Sphere, Obstacle),
// generated from their Rust definitions so the two sides cannot drift apart, and the code written
// for the robot (`robot_wgsl`): MAX_DOF, JAC_LEN, HANDS, fk(), ee_pos(h), ee_rot(h),
// ee_jacobian(h), mark_moving(h) and collision().

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> links: array<Link>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> limits: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> obstacles: array<Obstacle>;
@group(0) @binding(6) var<storage, read> world_ranges: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> item_world: array<u32>;
// IK targets: 4 vec4 per item and hand (position with w = 1 when the hand has a target, rotation
// columns).
@group(0) @binding(8) var<storage, read> targets: array<vec4<f32>>;
// Distance grid values, two half floats per word (see Obstacle.grid).
@group(0) @binding(9) var<storage, read> grid_data: array<u32>;
@group(0) @binding(10) var<storage, read_write> qbuf: array<f32>;
// trajopt: the L-BFGS state of each path (see lbfgs_stride).
@group(0) @binding(11) var<storage, read_write> aux: array<f32>;
@group(0) @binding(12) var<storage, read_write> outbuf: array<f32>;

const FAR: f32 = 1e30;

var<private> q: array<f32, MAX_DOF>;
var<private> grad: array<f32, MAX_DOF>;
var<private> jac: array<f32, JAC_LEN>;  // 6 x MAX_DOF, row-major
var<private> chol: array<f32, 36>;
// IK: each hand's target (position with w = 1 when it has one, rotation), loaded once per item,
// and the joints the targeted hands move.
var<private> goal_pos: array<vec4<f32>, HANDS>;
var<private> goal_rot: array<mat3x3<f32>, HANDS>;
var<private> moving: array<bool, MAX_DOF>;

fn item_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

fn rodrigues(a: vec3<f32>, angle: f32) -> mat3x3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    let t = 1.0 - c;
    return mat3x3<f32>(
        vec3<f32>(t * a.x * a.x + c, t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y),
        vec3<f32>(t * a.x * a.y - s * a.z, t * a.y * a.y + c, t * a.y * a.z + s * a.x),
        vec3<f32>(t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c));
}

// Returns (gradient direction, signed distance). Mirrors the distance functions in world.rs.
fn obstacle_distance(o: Obstacle, p: vec3<f32>) -> vec4<f32> {
    let v = p - o.center.xyz;
    let kind = u32(o.center.w);
    if (kind == SPHERE) {
        let len = length(v);
        var g = vec3<f32>(0.0, 0.0, 1.0);
        if (len > 1e-9) {
            g = v / len;
        }
        return vec4<f32>(g, len - o.half.x);
    }
    let lp = vec3<f32>(dot(o.r0.xyz, v), dot(o.r1.xyz, v), dot(o.r2.xyz, v));
    var gl: vec3<f32>;
    var d: f32;
    if (kind == CUBOID) {
        let sgn = select(vec3<f32>(-1.0), vec3<f32>(1.0), lp >= vec3<f32>(0.0));
        let qd = abs(lp) - o.half.xyz;
        let outside = max(qd, vec3<f32>(0.0));
        let olen = length(outside);
        if (olen > 0.0) {
            d = olen;
            gl = sgn * outside / olen;
        } else if (qd.x >= qd.y && qd.x >= qd.z) {
            d = qd.x;
            gl = vec3<f32>(sgn.x, 0.0, 0.0);
        } else if (qd.y >= qd.z) {
            d = qd.y;
            gl = vec3<f32>(0.0, sgn.y, 0.0);
        } else {
            d = qd.z;
            gl = vec3<f32>(0.0, 0.0, sgn.z);
        }
    } else if (kind == CYLINDER) {
        let rho = sqrt(lp.x * lp.x + lp.y * lp.y);
        var radial = vec3<f32>(1.0, 0.0, 0.0);
        if (rho > 1e-9) {
            radial = vec3<f32>(lp.x / rho, lp.y / rho, 0.0);
        }
        let axial = vec3<f32>(0.0, 0.0, select(-1.0, 1.0, lp.z >= 0.0));
        let dr = rho - o.half.x;
        let dz = abs(lp.z) - o.half.y;
        if (dr > 0.0 && dz > 0.0) {
            d = sqrt(dr * dr + dz * dz);
            gl = (radial * dr + axial * dz) / d;
        } else if (dr >= dz) {
            d = dr;
            gl = radial;
        } else {
            d = dz;
            gl = axial;
        }
    } else if (kind == SDF) {
        let gd = grid_distance(o, lp);
        d = gd.w;
        gl = gd.xyz;
    } else {
        let w = lp - vec3<f32>(0.0, 0.0, clamp(lp.z, -o.half.y, o.half.y));
        let len = length(w);
        d = len - o.half.x;
        gl = vec3<f32>(1.0, 0.0, 0.0);
        if (len > 1e-9) {
            gl = w / len;
        }
    }
    return vec4<f32>(o.r0.xyz * gl.x + o.r1.xyz * gl.y + o.r2.xyz * gl.z, d);
}

// Rotation vector and angle of r (column-major: r[col][row]).
fn rot_log(r: mat3x3<f32>) -> vec4<f32> {
    let cos_t = clamp((r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5, -1.0, 1.0);
    let v = 0.5 * vec3<f32>(r[1][2] - r[2][1], r[2][0] - r[0][2], r[0][1] - r[1][0]);
    let sv = length(v);
    let theta = atan2(sv, cos_t);
    if (sv > 1e-6) {
        return vec4<f32>(v * (theta / sv), theta);
    }
    if (cos_t > 0.0) {
        return vec4<f32>(v, theta);
    }
    var axis: vec3<f32>;
    if (r[1][1] > r[0][0]) {
        if (r[2][2] > r[1][1]) { axis = r[2] + vec3<f32>(0.0, 0.0, 1.0); } else { axis = r[1] + vec3<f32>(0.0, 1.0, 0.0); }
    } else if (r[2][2] > r[0][0]) {
        axis = r[2] + vec3<f32>(0.0, 0.0, 1.0);
    } else {
        axis = r[0] + vec3<f32>(1.0, 0.0, 0.0);
    }
    return vec4<f32>(normalize(axis) * theta, theta);
}

fn chol6() {
    for (var i = 0u; i < 6u; i++) {
        for (var j = 0u; j <= i; j++) {
            var s = chol[i * 6u + j];
            for (var k = 0u; k < j; k++) {
                s -= chol[i * 6u + k] * chol[j * 6u + k];
            }
            if (i == j) {
                chol[i * 6u + j] = sqrt(max(s, 1e-12));
            } else {
                chol[i * 6u + j] = s / chol[j * 6u + j];
            }
        }
    }
}

fn chol6_solve(b: array<f32, 6>) -> array<f32, 6> {
    var y: array<f32, 6>;
    for (var i = 0u; i < 6u; i++) {
        var s = b[i];
        for (var k = 0u; k < i; k++) {
            s -= chol[i * 6u + k] * y[k];
        }
        y[i] = s / chol[i * 6u + i];
    }
    var x: array<f32, 6>;
    for (var ii = 0; ii < 6; ii++) {
        let i = u32(5 - ii);
        var s = y[i];
        for (var k = i + 1u; k < 6u; k++) {
            s -= chol[k * 6u + i] * x[k];
        }
        x[i] = s / chol[i * 6u + i];
    }
    return x;
}

// Item `item`'s targets into goal_pos and goal_rot, and the joints they move into `moving`.
fn load_goals(item: u32) {
    for (var j = 0u; j < MAX_DOF; j++) {
        moving[j] = false;
    }
    for (var h = 0u; h < HANDS; h++) {
        let t = (item * HANDS + h) * 4u;
        goal_pos[h] = targets[t];
        goal_rot[h] = mat3x3<f32>(targets[t + 1u].xyz, targets[t + 2u].xyz, targets[t + 3u].xyz);
        if (goal_pos[h].w > 0.5) {
            mark_moving(h);
        }
    }
}

// Hand h's Jacobian into `jac`, and its damped normal matrix, factored, into `chol`.
fn hand_system(h: u32) {
    let n = P.n_dof;
    ee_jacobian(h);
    for (var r = 0u; r < 6u; r++) {
        for (var c = 0u; c <= r; c++) {
            var s = 0.0;
            for (var j = 0u; j < n; j++) {
                s += jac[r * MAX_DOF + j] * jac[c * MAX_DOF + j];
            }
            if (r == c) {
                s += P.damping * P.damping;
            }
            chol[r * 6u + c] = s;
            chol[c * 6u + r] = s;
        }
    }
    chol6();
}

// Hand h's step toward its target, J^T (J J^T + damping^2 I)^-1 e, into `dq`.
fn pose_step(h: u32, dq: ptr<function, array<f32, MAX_DOF>>) {
    let ep = goal_pos[h].xyz - ee_pos(h);
    let eo = rot_log(goal_rot[h] * transpose(ee_rot(h))).xyz * P.rot_weight;
    hand_system(h);
    let y = chol6_solve(array<f32, 6>(ep.x, ep.y, ep.z, eo.x, eo.y, eo.z));
    for (var j = 0u; j < P.n_dof; j++) {
        var s = 0.0;
        for (var r = 0u; r < 6u; r++) {
            s += jac[r * MAX_DOF + j] * y[r];
        }
        (*dq)[j] += s;
    }
}

// J^T (J J^T + damping^2 I)^-1 J grad for the system in jac and chol, into `push`.
fn null_part(push: ptr<function, array<f32, MAX_DOF>>) {
    let n = P.n_dof;
    var jg: array<f32, 6>;
    for (var r = 0u; r < 6u; r++) {
        var s = 0.0;
        for (var j = 0u; j < n; j++) {
            s += jac[r * MAX_DOF + j] * grad[j];
        }
        jg[r] = s;
    }
    let z = chol6_solve(jg);
    for (var j = 0u; j < n; j++) {
        var s = 0.0;
        for (var r = 0u; r < 6u; r++) {
            s += jac[r * MAX_DOF + j] * z[r];
        }
        (*push)[j] += s;
    }
}

// One damped-least-squares step toward each targeted hand's pose, then a collision push in the
// null space of every targeted hand's Jacobian: `moving` joints take
// -collision_step (grad - sum over hands of J^T (J J^T + damping^2 I)^-1 J grad). Hands on separate
// arms make that the step of the stacked system. Mirrors `ik_step` in cpu.rs.
fn ik_step(world: u32, lid: u32, lane: u32) {
    let n = P.n_dof;
    fk();
    var dq: array<f32, MAX_DOF>;
    var push: array<f32, MAX_DOF>;
    // One hand, the usual case, keeps its system in jac and chol for the push: a loop over hands
    // around this code costs 4-10% of IK throughput on an M4 Pro's GPU. HANDS is a constant, so
    // only one branch is compiled.
    var last = 0u;
    if (HANDS == 1u) {
        pose_step(0u, &dq);
    } else {
        for (var h = 0u; h < HANDS; h++) {
            if (goal_pos[h].w > 0.5) {
                pose_step(h, &dq);
                last = h;
            }
        }
    }
    if (P.collision_step > 0.0) {
        for (var j = 0u; j < n; j++) {
            grad[j] = 0.0;
        }
        collision(world, P.w_world, P.w_self, P.margin, P.self_margin, true, lane);
        group_grad(lid);
        if (HANDS == 1u) {
            null_part(&push);
        } else {
            for (var h = 0u; h < HANDS; h++) {
                if (goal_pos[h].w > 0.5) {
                    if (h != last) {
                        hand_system(h);
                    }
                    null_part(&push);
                }
            }
        }
        for (var j = 0u; j < n; j++) {
            if (moving[j]) {
                dq[j] -= P.collision_step * (grad[j] - push[j]);
            }
        }
    }
    var largest = 0.0;
    for (var j = 0u; j < n; j++) {
        largest = max(largest, abs(dq[j]));
    }
    var scale = 1.0;
    if (largest > P.max_step) {
        scale = P.max_step / largest;
    }
    for (var j = 0u; j < n; j++) {
        let lim = limits[j];
        q[j] = clamp(q[j] + dq[j] * scale, lim.x, lim.y);
    }
}

// Collision kernels give each configuration LANES consecutive invocations, which split its
// collision work (see collision()). gpu.rs builds them twice: with one lane, where the lane code
// folds away, for large batches, and with several for batches too small to fill the GPU. They have
// no early returns, because the lanes meet at barriers: invocations past the end work on the last
// configuration and write nothing.
override LANES: u32 = 1u;
var<workgroup> grads: array<f32, WORKGROUP * MAX_DOF>;

// The configuration and lane of invocation `thread` among `configurations`, and whether it has
// work of its own.
fn lane_of(thread: u32, configurations: u32) -> vec3<u32> {
    let config = thread >> countTrailingZeros(LANES);
    return vec3<u32>(min(config, configurations - 1u), thread & (LANES - 1u), u32(config < configurations));
}

// Combines the lanes' collision() results: costs add, clearances take the minimum. Every lane gets
// the total.
fn group_collision(lid: u32, c: vec3<f32>) -> vec3<f32> {
    let lanes = LANES;
    if (lanes == 1u) {
        return c;
    }
    workgroupBarrier();
    partial[lid] = vec4<f32>(c, 0.0);
    workgroupBarrier();
    for (var stride = lanes / 2u; stride > 0u; stride >>= 1u) {
        if ((lid & (lanes - 1u)) < stride) {
            let o = partial[lid + stride];
            let m = partial[lid];
            partial[lid] = vec4<f32>(m.x + o.x, min(m.y, o.y), min(m.z, o.z), 0.0);
        }
        workgroupBarrier();
    }
    return partial[lid & ~(lanes - 1u)].xyz;
}

// Sums the lanes' partial gradients in `grad`; every lane gets the total.
fn group_grad(lid: u32) {
    let lanes = LANES;
    let n = P.n_dof;
    if (lanes == 1u) {
        return;
    }
    workgroupBarrier();
    for (var j = 0u; j < n; j++) {
        grads[lid * MAX_DOF + j] = grad[j];
    }
    workgroupBarrier();
    for (var stride = lanes / 2u; stride > 0u; stride >>= 1u) {
        if ((lid & (lanes - 1u)) < stride) {
            for (var j = 0u; j < n; j++) {
                grads[lid * MAX_DOF + j] += grads[(lid + stride) * MAX_DOF + j];
            }
        }
        workgroupBarrier();
    }
    let first = (lid & ~(lanes - 1u)) * MAX_DOF;
    for (var j = 0u; j < n; j++) {
        grad[j] = grads[first + j];
    }
}

// One configuration per lane group: out = [world clearance, self clearance, cost, grad...].
@compute @workgroup_size(WORKGROUP)
fn evaluate_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let at = lane_of(item_index(gid, nwg), P.n_items);
    let item = at.x;
    let n = P.n_dof;
    for (var j = 0u; j < n; j++) {
        q[j] = qbuf[item * n + j];
        grad[j] = 0.0;
    }
    fk();
    let part = collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin, true, at.y);
    let c = group_collision(lid, part);
    group_grad(lid);
    if (at.z == 1u && at.y == 0u) {
        let base = item * (3u + n);
        outbuf[base] = c.y;
        outbuf[base + 1u] = c.z;
        outbuf[base + 2u] = c.x;
        for (var j = 0u; j < n; j++) {
            outbuf[base + 3u + j] = grad[j];
        }
    }
}

// One configuration per lane group: out = [world clearance, self clearance], without cost or
// gradient.
@compute @workgroup_size(WORKGROUP)
fn clearance_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let at = lane_of(item_index(gid, nwg), P.n_items);
    let item = at.x;
    let n = P.n_dof;
    for (var j = 0u; j < n; j++) {
        q[j] = qbuf[item * n + j];
    }
    fk();
    let c = group_collision(lid, collision(item_world[item], 0.0, 0.0, 0.0, 0.0, false, at.y));
    if (at.z == 1u && at.y == 0u) {
        outbuf[item * 2u] = c.y;
        outbuf[item * 2u + 1u] = c.z;
    }
}

// One IK seed per lane group; runs P.iterations steps, writes q back and [pos err, rot err]. The
// lanes split each step's collision gradient and otherwise do the same work.
@compute @workgroup_size(WORKGROUP)
fn ik_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let at = lane_of(item_index(gid, nwg), P.n_items);
    let item = at.x;
    let n = P.n_dof;
    for (var j = 0u; j < n; j++) {
        q[j] = qbuf[item * n + j];
    }
    let world = item_world[item];
    load_goals(item);
    for (var it = 0u; it < P.iterations; it++) {
        ik_step(world, lid, at.y);
    }
    if (at.z == 1u && at.y == 0u) {
        for (var j = 0u; j < n; j++) {
            qbuf[item * n + j] = q[j];
        }
        fk();
        var errors = vec2<f32>(0.0);
        for (var h = 0u; h < HANDS; h++) {
            if (goal_pos[h].w > 0.5) {
                let pos_err = length(goal_pos[h].xyz - ee_pos(h));
                let rot_err = rot_log(goal_rot[h] * transpose(ee_rot(h))).w;
                errors = max(errors, vec2<f32>(pos_err, rot_err));
            }
        }
        outbuf[item * 2u] = errors.x;
        outbuf[item * 2u + 1u] = errors.y;
    }
}

// Uniform cubic B-spline weights of a span's four control points at u in [0, 1]. Mirrors spline.rs.
fn basis(u: f32) -> vec4<f32> {
    let v = 1.0 - u;
    let u2 = u * u;
    let u3 = u2 * u;
    return vec4<f32>(v * v * v, 3.0 * u3 - 6.0 * u2 + 4.0, -3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0, u3) / 6.0;
}

// Collision sample s of a span sits at u = (s + 0.5) / P.samples.
fn sample_u(s: u32) -> f32 {
    return (f32(s) + 0.5) / f32(P.samples);
}

// Collision samples along one path.
fn path_samples() -> u32 {
    return (P.points - 3u) * P.samples;
}

// Each path's L-BFGS state in aux, in this order: the collision gradient per sample, the
// collision cost per line-search step and sample, then per control point the gradient, the
// previous gradient, the direction, the pending step, the step history and the gradient-change
// history, then five scalars: cost, history count, newest slot, started, pending. Mirrors
// lbfgs_stride in gpu.rs and the Lbfgs struct in cpu.rs.
fn lbfgs_stride() -> u32 {
    let tn = P.points * P.n_dof;
    return path_samples() * P.n_dof + LINE_STEPS * path_samples() + (4u + 2u * P.history) * tn + 5u;
}

fn costs_at(item: u32) -> u32 {
    return item * lbfgs_stride() + path_samples() * P.n_dof;
}

fn grad_at(item: u32) -> u32 {
    return costs_at(item) + LINE_STEPS * path_samples();
}

fn prev_grad_at(item: u32) -> u32 {
    return grad_at(item) + P.points * P.n_dof;
}

fn dir_at(item: u32) -> u32 {
    return prev_grad_at(item) + P.points * P.n_dof;
}

fn pending_step_at(item: u32) -> u32 {
    return dir_at(item) + P.points * P.n_dof;
}

fn steps_at(item: u32, slot: u32) -> u32 {
    return pending_step_at(item) + (1u + slot) * P.points * P.n_dof;
}

fn changes_at(item: u32, slot: u32) -> u32 {
    return steps_at(item, P.history + slot);
}

fn scalars_at(item: u32) -> u32 {
    return steps_at(item, 2u * P.history);
}

// Control point t, joint j of the candidate path x + alpha d: free points move and are clamped
// to the joint range, the three pinned at each end stay.
fn candidate(item: u32, t: u32, j: u32, alpha: f32) -> f32 {
    let i = t * P.n_dof + j;
    let x = qbuf[item * P.points * P.n_dof + i];
    if (t < 3u || t >= P.points - 3u) {
        return x;
    }
    let lim = limits[j];
    return clamp(x + alpha * aux[dir_at(item) + i], lim.x, lim.y);
}

// One lane group per line-search step and collision sample: the collision cost at that point of
// the candidate path.
@compute @workgroup_size(WORKGROUP)
fn traj_costs(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let per_item = path_samples();
    let at = lane_of(item_index(gid, nwg), P.n_items * LINE_STEPS * per_item);
    let idx = at.x;
    let item = idx / (LINE_STEPS * per_item);
    let c = idx / per_item % LINE_STEPS;
    let sample = idx % per_item;
    let span = sample / P.samples;
    let w = basis(sample_u(sample % P.samples));
    let alpha = line_search(c);
    for (var j = 0u; j < P.n_dof; j++) {
        q[j] = w.x * candidate(item, span, j, alpha) + w.y * candidate(item, span + 1u, j, alpha)
            + w.z * candidate(item, span + 2u, j, alpha) + w.w * candidate(item, span + 3u, j, alpha);
    }
    fk();
    let part = collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin, false, at.y);
    let c3 = group_collision(lid, part);
    if (at.z == 1u && at.y == 0u) {
        aux[costs_at(item) + c * per_item + sample] = c3.x;
    }
}

// Per-path kernels run one workgroup per path, each invocation owning every WORKGROUP-th element,
// so their loads coalesce; sums over a path go through workgroup memory.
var<workgroup> partial: array<vec4<f32>, WORKGROUP>;
var<workgroup> reduced: vec4<f32>;

// The component-wise sum of `v` over the workgroup, uniform in every invocation. Every invocation
// must call it.
fn workgroup_sum(lid: u32, v: vec4<f32>) -> vec4<f32> {
    partial[lid] = v;
    workgroupBarrier();
    for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        reduced = partial[0];
    }
    return workgroupUniformLoad(&reduced);
}

// The component-wise maximum of `v` over the workgroup, like workgroup_sum.
fn workgroup_max(lid: u32, v: vec4<f32>) -> vec4<f32> {
    partial[lid] = v;
    workgroupBarrier();
    for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            partial[lid] = max(partial[lid], partial[lid + stride]);
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        reduced = partial[0];
    }
    return workgroupUniformLoad(&reduced);
}

// `aux[at]`, uniform in every invocation.
fn uniform_aux(lid: u32, at: u32) -> f32 {
    // No invocation may still be reading the previous value.
    workgroupBarrier();
    if (lid == 0u) {
        reduced = vec4<f32>(aux[at]);
    }
    return workgroupUniformLoad(&reduced).x;
}

// One workgroup per path: the trajectory cost of each line-search step (up to four, one per
// component); the cheapest moves the path if it lowers the cost, otherwise the history resets.
@compute @workgroup_size(WORKGROUP)
fn traj_search(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nwg.x;
    if (item >= P.n_items) {
        return;
    }
    let n = P.n_dof;
    let tn = P.points;
    let per_item = path_samples();
    let sc = scalars_at(item);
    var part = vec4<f32>(0.0);
    for (var s = lid; s < per_item; s += WORKGROUP) {
        for (var c = 0u; c < LINE_STEPS; c++) {
            part[c] += aux[costs_at(item) + c * per_item + s];
        }
    }
    // Velocity and acceleration ending at each control point.
    for (var k = n + lid; k < tn * n; k += WORKGROUP) {
        let t = k / n;
        let j = k % n;
        for (var c = 0u; c < LINE_STEPS; c++) {
            let alpha = line_search(c);
            let current = candidate(item, t, j, alpha);
            let previous = candidate(item, t - 1u, j, alpha);
            let v = current - previous;
            part[c] += P.w_vel * v * v;
            if (t >= 2u) {
                let a = current - 2.0 * previous + candidate(item, t - 2u, j, alpha);
                part[c] += P.w_acc * a * a;
            }
        }
    }
    let totals = workgroup_sum(lid, part);
    let started = uniform_aux(lid, sc + 3u) > 0.5;
    var cost = uniform_aux(lid, sc);
    var best = -1;
    for (var c = 0u; c < LINE_STEPS; c++) {
        if ((best < 0 && !started) || totals[c] < cost) {
            cost = totals[c];
            best = i32(c);
        }
    }
    if (best < 0) {
        if (lid == 0u) {
            aux[sc + 4u] = 0.0;
            aux[sc + 1u] = 0.0;
        }
        return;
    }
    let alpha = line_search(u32(best));
    for (var i = 3u * n + lid; i < (tn - 3u) * n; i += WORKGROUP) {
        let moved = candidate(item, i / n, i % n, alpha);
        let x = item * tn * n + i;
        aux[pending_step_at(item) + i] = moved - qbuf[x];
        qbuf[x] = moved;
    }
    if (lid == 0u) {
        aux[sc] = cost;
        aux[sc + 3u] = 1.0;
        aux[sc + 4u] = 1.0;
    }
}

// The dot products a.b and c.e over a path's free control points, uniform in every invocation.
fn lbfgs_dots(lid: u32, a: u32, b: u32, c: u32, e: u32) -> vec2<f32> {
    var part = vec4<f32>(0.0);
    for (var i = 3u * P.n_dof + lid; i < (P.points - 3u) * P.n_dof; i += WORKGROUP) {
        part.x += aux[a + i] * aux[b + i];
        part.y += aux[c + i] * aux[e + i];
    }
    return workgroup_sum(lid, part).xy;
}

// One workgroup per path: records the last step and the gradient change it caused, then the
// L-BFGS two-loop recursion for the next direction; steepest descent scaled to initial_step
// without history or when the recursion fails to descend.
@compute @workgroup_size(WORKGROUP)
fn lbfgs_direction(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let item = wid.x + wid.y * nwg.x;
    if (item >= P.n_items) {
        return;
    }
    let first = 3u * P.n_dof + lid;
    let end = (P.points - 3u) * P.n_dof;
    let m = P.history;
    let sc = scalars_at(item);
    let g = grad_at(item);
    let gp = prev_grad_at(item);
    let d = dir_at(item);
    var count = u32(uniform_aux(lid, sc + 1u));
    var newest = u32(uniform_aux(lid, sc + 2u));
    if (uniform_aux(lid, sc + 4u) > 0.5) {
        let s = pending_step_at(item);
        var part = 0.0;
        for (var i = first; i < end; i += WORKGROUP) {
            part += aux[s + i] * (aux[g + i] - aux[gp + i]);
        }
        let sy = workgroup_sum(lid, vec4<f32>(part, 0.0, 0.0, 0.0)).x;
        if (sy > 1e-10) {
            var slot = 0u;
            if (count > 0u) {
                slot = (newest + 1u) % m;
            }
            for (var i = first; i < end; i += WORKGROUP) {
                aux[steps_at(item, slot) + i] = aux[s + i];
                aux[changes_at(item, slot) + i] = aux[g + i] - aux[gp + i];
            }
            newest = slot;
            count = min(count + 1u, m);
        }
        if (lid == 0u) {
            aux[sc + 4u] = 0.0;
        }
    }
    var big = 0.0;
    for (var i = first; i < end; i += WORKGROUP) {
        aux[gp + i] = aux[g + i];
        aux[d + i] = -aux[g + i];
        big = max(big, abs(aux[g + i]));
    }
    let largest = workgroup_max(lid, vec4<f32>(big)).x;
    var alpha: array<f32, MAX_HISTORY>;
    for (var a = 0u; a < count; a++) {
        let slot = (newest + m - a) % m;
        let s = steps_at(item, slot);
        let y = changes_at(item, slot);
        let dots = lbfgs_dots(lid, s, d, y, s);
        alpha[a] = dots.x / dots.y;
        for (var i = first; i < end; i += WORKGROUP) {
            aux[d + i] -= alpha[a] * aux[y + i];
        }
    }
    var gamma = 0.0;
    if (count > 0u) {
        let s = steps_at(item, newest);
        let y = changes_at(item, newest);
        let dots = lbfgs_dots(lid, s, y, y, y);
        gamma = dots.x / dots.y;
    } else if (largest > 0.0) {
        gamma = P.initial_step / largest;
    }
    for (var i = first; i < end; i += WORKGROUP) {
        aux[d + i] *= gamma;
    }
    for (var k = 0u; k < count; k++) {
        let a = count - 1u - k;
        let slot = (newest + m - a) % m;
        let s = steps_at(item, slot);
        let y = changes_at(item, slot);
        let dots = lbfgs_dots(lid, y, d, y, s);
        let beta = dots.x / dots.y;
        for (var i = first; i < end; i += WORKGROUP) {
            aux[d + i] += (alpha[a] - beta) * aux[s + i];
        }
    }
    if (largest > 0.0) {
        if (lbfgs_dots(lid, d, g, d, g).x >= 0.0) {
            count = 0u;
            for (var i = first; i < end; i += WORKGROUP) {
                aux[d + i] = -aux[g + i] * (P.initial_step / largest);
            }
        }
    }
    if (lid == 0u) {
        aux[sc + 1u] = f32(count);
        aux[sc + 2u] = f32(newest);
    }
}

// One lane group per collision sample: writes d(collision cost)/dq at that point of the curve.
@compute @workgroup_size(WORKGROUP)
fn traj_samples(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let per_item = (P.points - 3u) * P.samples;
    let at = lane_of(item_index(gid, nwg), P.n_items * per_item);
    let idx = at.x;
    let item = idx / per_item;
    let span = (idx % per_item) / P.samples;
    let w = basis(sample_u(idx % P.samples));
    let n = P.n_dof;
    let base = (item * P.points + span) * n;
    for (var j = 0u; j < n; j++) {
        q[j] = w.x * qbuf[base + j] + w.y * qbuf[base + n + j] + w.z * qbuf[base + 2u * n + j] + w.w * qbuf[base + 3u * n + j];
        grad[j] = 0.0;
    }
    fk();
    collision(item_world[item], P.w_world, P.w_self, P.margin, P.self_margin, true, at.y);
    group_grad(lid);
    if (at.z == 1u && at.y == 0u) {
        let sample = idx % per_item;
        for (var j = 0u; j < n; j++) {
            aux[item * lbfgs_stride() + sample * n + j] = grad[j];
        }
    }
}

// One invocation per free control point: the collision gradients of the samples it shapes, each
// weighted by its basis value there, plus smoothness on the control points.
@compute @workgroup_size(64)
fn traj_grad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = item_index(gid, nwg);
    let tn = P.points;
    let free = tn - 6u;
    if (idx >= P.n_items * free) {
        return;
    }
    let item = idx / free;
    let t = idx % free + 3u;
    let n = P.n_dof;
    let spans = tn - 3u;
    let base = (item * tn + t) * n;
    for (var j = 0u; j < n; j++) {
        grad[j] = 0.0;
    }
    // Control point t is point i of span t - i.
    for (var i = 0u; i < 4u; i++) {
        if (t < i || t - i >= spans) {
            continue;
        }
        for (var s = 0u; s < P.samples; s++) {
            let w = basis(sample_u(s))[i];
            let g = item * lbfgs_stride() + ((t - i) * P.samples + s) * n;
            for (var j = 0u; j < n; j++) {
                grad[j] += w * aux[g + j];
            }
        }
    }
    for (var j = 0u; j < n; j++) {
        let qm = qbuf[base - n + j];
        let q0 = qbuf[base + j];
        let qp = qbuf[base + n + j];
        var g = 2.0 * P.w_vel * (2.0 * q0 - qm - qp);
        g += 2.0 * P.w_acc * -2.0 * (qp - 2.0 * q0 + qm);
        if (t >= 2u) {
            g += 2.0 * P.w_acc * (q0 - 2.0 * qm + qbuf[base - 2u * n + j]);
        }
        if (t + 2u < tn) {
            g += 2.0 * P.w_acc * (qbuf[base + 2u * n + j] - 2.0 * qp + q0);
        }
        aux[grad_at(item) + t * n + j] = grad[j] + g;
    }
}
