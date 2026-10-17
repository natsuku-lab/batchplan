// Distance grids on the GPU: what a depth image says about each point of a grid, fusing images
// into an occupancy map's log-odds, and the exact squared distance transform. Mirrors sdf.rs:
// `DepthImage::classify`, `depth_occupancy`, `integrate`, `squared_edt`, `Line`, `meet` and
// `finish`. The transform is integer arithmetic and matches the CPU exactly, and the CPU computes
// the table that finishes each value (`finishing_table`). Projections are floating
// point, so a point that projects within rounding of a pixel's edge or of a reading's depth may be
// classified differently.
//
// Generated before this file: `GridParams`, `WORKGROUP`, and sdf.rs's `HIT`, `MISS`, `CLAMP_LO`,
// `CLAMP_HI`, `UNOBSERVED` and `SHADOWED`.

@group(0) @binding(0) var<uniform> G: GridParams;
// Per pixel, row by row: the depth reading (0 without one). The robot mask: the depth to the robot
// per pixel, then `G.spheres` grown spheres (centre, radius).
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var<storage, read> robot: array<f32>;
// One bit per grid point, x fastest: the occupancy the transform reads, or the points a map's
// readings hit.
@group(0) @binding(3) var<storage, read_write> bits: array<atomic<u32>>;
// Squared distances, in voxels, to the nearest occupied and the nearest free point: the x pass
// writes `to_*`, the y pass `next_*`, and the z pass each point's distance to the other kind into
// `to_solid`.
@group(0) @binding(4) var<storage, read_write> to_solid: array<u32>;
@group(0) @binding(5) var<storage, read_write> to_free: array<u32>;
@group(0) @binding(6) var<storage, read_write> next_solid: array<u32>;
@group(0) @binding(7) var<storage, read_write> next_free: array<u32>;
// Each line's envelope (`Line` in sdf.rs), entry `k` of line `l` at `k * G.lines + l`: the root
// in the low half, where it starts in the high half.
@group(0) @binding(8) var<storage, read_write> envelope: array<u32>;
// A map's log-odds (`OccupancyMap`), four points per word, little end first.
@group(0) @binding(9) var<storage, read_write> log_odds: array<u32>;
// `finishing_table` in sdf.rs: per squared distance, a free point's value in the low half and an
// occupied point's in the high half.
@group(0) @binding(10) var<storage, read> table: array<u32>;

const FAR: u32 = 0xffffffffu;

const UNSEEN: u32 = 0u;
const FREE: u32 = 1u;
const SURFACE: u32 = 2u;
const HIDDEN: u32 = 3u;
const SHADOW: u32 = 4u;

// `G.behind`, as `Occlusion` in sdf.rs.
const BEHIND_FREE: u32 = 0u;
const BEHIND_OCCUPIED: u32 = 1u;
const BEHIND_OCCUPIED_EXCEPT_SHADOW: u32 = 2u;

fn invocation(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * WORKGROUP;
}

fn grid_point(n: u32) -> vec3<f32> {
    let i = vec3<u32>(n % G.nx, n / G.nx % G.ny, n / (G.nx * G.ny));
    return G.origin.xyz + vec3<f32>(i) * G.origin.w;
}

fn nearest(p: vec3<f32>) -> u32 {
    let top = vec3<f32>(f32(G.nx - 1u), f32(G.ny - 1u), f32(G.nz - 1u));
    let i = vec3<u32>(clamp(floor((p - G.origin.xyz) / G.origin.w + 0.5), vec3<f32>(0.0), top));
    return i.x + G.nx * (i.y + G.ny * i.z);
}

fn is_set(n: u32) -> bool {
    return ((atomicLoad(&bits[n >> 5u]) >> (n & 31u)) & 1u) == 1u;
}

// Whether `p` is inside the robot (`RobotMask::contains`).
fn inside_robot(p: vec3<f32>) -> bool {
    let first = G.width * G.height;
    for (var s = 0u; s < G.spheres; s++) {
        let k = first + 4u * s;
        let d = p - vec3<f32>(robot[k], robot[k + 1u], robot[k + 2u]);
        if dot(d, d) < robot[k + 3u] * robot[k + 3u] {
            return true;
        }
    }
    return false;
}

// What the image says about the grid point `p`, with the space the robot fills free
// (`DepthImage::classify`).
fn classify(p: vec3<f32>) -> u32 {
    let seen = sees(p);
    if (seen == HIDDEN || seen == SHADOW || seen == UNSEEN) && inside_robot(p) {
        return FREE;
    }
    return seen;
}

// What the image shows of the grid point `p` (`DepthImage::sees`).
fn sees(p: vec3<f32>) -> u32 {
    let voxel = G.origin.w;
    let half = 0.5 * voxel;
    let d = p - G.camera.xyz;
    let c = vec3<f32>(dot(G.to_camera0.xyz, d), dot(G.to_camera1.xyz, d), dot(G.to_camera2.xyz, d));
    if c.z <= 0.0 {
        return UNSEEN;
    }
    let u = floor(G.intrinsics.x * c.x / c.z + G.intrinsics.z + 0.5);
    let v = floor(G.intrinsics.y * c.y / c.z + G.intrinsics.w + 0.5);
    if u < 0.0 || v < 0.0 || u >= f32(G.width) || v >= f32(G.height) {
        return UNSEEN;
    }
    let i = u32(v) * G.width + u32(u);
    let z = depth[i];
    if z <= 0.0 {
        return UNSEEN;
    }
    if G.has_robot == 1u {
        let r = robot[i];
        if z >= r - voxel {
            return select(SHADOW, FREE, c.z < r - half);
        }
    }
    if c.z < z - half {
        return FREE;
    }
    if c.z <= z + half {
        return SURFACE;
    }
    return HIDDEN;
}

// One word of occupancy bits for `SdfGrid::from_depth`; `depth_hits` adds the readings' points.
@compute @workgroup_size(WORKGROUP)
fn depth_occupancy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let w = invocation(gid, groups);
    if w >= (G.points + 31u) / 32u {
        return;
    }
    var word = 0u;
    for (var b = 0u; b < 32u; b++) {
        let n = w * 32u + b;
        if n >= G.points {
            break;
        }
        let seen = classify(grid_point(n));
        let hidden_solid = seen == HIDDEN && G.behind != BEHIND_FREE;
        let shadow_solid = seen == SHADOW && G.behind == BEHIND_OCCUPIED;
        if seen == SURFACE || hidden_solid || shadow_solid {
            word |= 1u << b;
        }
    }
    atomicStore(&bits[w], word);
}

// Sets the bit of the grid point holding each reading off the robot. Maps take no readings
// beyond their edge.
@compute @workgroup_size(WORKGROUP)
fn depth_hits(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = invocation(gid, groups);
    if i >= G.width * G.height {
        return;
    }
    let z = depth[i];
    if z <= 0.0 || (G.has_robot == 1u && z >= robot[i] - G.origin.w) {
        return;
    }
    let uv = vec2<f32>(f32(i % G.width), f32(i / G.width));
    let x = (uv.x - G.intrinsics.z) / G.intrinsics.x * z;
    let y = (uv.y - G.intrinsics.w) / G.intrinsics.y * z;
    let p = G.from_camera0.xyz * x + G.from_camera1.xyz * y + G.from_camera2.xyz * z + G.camera.xyz;
    if G.map == 1u {
        let half = 0.5 * G.origin.w;
        let hi = G.origin.xyz + vec3<f32>(f32(G.nx - 1u), f32(G.ny - 1u), f32(G.nz - 1u)) * G.origin.w + half;
        if any(p < G.origin.xyz - half) || any(p >= hi) {
            return;
        }
    }
    let n = nearest(p);
    atomicOr(&bits[n >> 5u], 1u << (n & 31u));
}

// Updates four points of a map: hits (classified at a reading or holding a reading's point) and
// misses (classified free), clamped.
@compute @workgroup_size(WORKGROUP)
fn map_update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let w = invocation(gid, groups);
    if w >= (G.points + 3u) / 4u {
        return;
    }
    var word = log_odds[w];
    for (var b = 0u; b < 4u; b++) {
        let n = w * 4u + b;
        if n >= G.points {
            break;
        }
        let seen = classify(grid_point(n));
        let shift = 8u * b;
        var l = i32(word << (24u - shift)) >> 24u;
        var step: i32;
        if is_set(n) || seen == SURFACE {
            step = HIT;
        } else if seen == FREE {
            step = MISS;
        } else if seen == SHADOW && l == UNOBSERVED {
            // Only the robot hides it: noted, until an image observes it.
            word = (word & ~(0xffu << shift)) | ((u32(SHADOWED) & 0xffu) << shift);
            continue;
        } else {
            continue;
        }
        if l == UNOBSERVED || l == SHADOWED {
            l = 0;
        }
        l = clamp(l + step, CLAMP_LO, CLAMP_HI);
        word = (word & ~(0xffu << shift)) | ((u32(l) & 0xffu) << shift);
    }
    log_odds[w] = word;
}

// The first position at which the parabola rooted at `q` is at most the one rooted at `p < q`.
fn meet(p: u32, fp: u32, q: u32, fq: u32) -> i32 {
    let num = i32(fq + q * q) - i32(fp + p * p);
    let den = i32(2u * (q - p));
    return num / den + select(0, 1, num % den > 0);
}

// Point `i` of the pass's input for sites `site` (0 occupied, 1 free).
fn height(site: u32, i: u32) -> u32 {
    switch G.axis {
        case 0u: {
            return select(FAR, 0u, is_set(i) == (site == 0u));
        }
        case 1u: {
            return select(to_free[i], to_solid[i], site == 0u);
        }
        default: {
            return select(next_free[i], next_solid[i], site == 0u);
        }
    }
}

// One axis of the distance transform: each invocation runs Felzenszwalb and Huttenlocher's lower
// envelope of parabolas along one line of points, for both kinds of site.
@compute @workgroup_size(WORKGROUP)
fn edt_pass(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let line = invocation(gid, groups);
    if line >= G.lines {
        return;
    }
    var n: u32;
    var stride: u32;
    var base: u32;
    switch G.axis {
        case 0u: {
            n = G.nx;
            stride = 1u;
            base = line * G.nx;
        }
        case 1u: {
            n = G.ny;
            stride = G.nx;
            base = line % G.nx + line / G.nx * G.nx * G.ny;
        }
        default: {
            n = G.nz;
            stride = G.nx * G.ny;
            base = line;
        }
    }
    for (var site = 0u; site < 2u; site++) {
        var top = 0u;
        for (var q = 0u; q < n; q++) {
            let fq = height(site, base + q * stride);
            if fq == FAR {
                continue;
            }
            var s = 0u;
            while top > 0u {
                let entry = envelope[(top - 1u) * G.lines + line];
                let p = entry & 0xffffu;
                s = u32(clamp(meet(p, height(site, base + p * stride), q, fq), 0, i32(n)));
                if s > entry >> 16u {
                    break;
                }
                top--;
                s = 0u;
            }
            envelope[top * G.lines + line] = q | s << 16u;
            top++;
        }
        var k = 0u;
        var root = envelope[line] & 0xffffu;
        for (var x = 0u; x < n; x++) {
            let i = base + x * stride;
            var d = FAR;
            if top > 0u {
                while k + 1u < top {
                    let next = envelope[(k + 1u) * G.lines + line];
                    if next >> 16u > x {
                        break;
                    }
                    k++;
                    root = next & 0xffffu;
                }
                let dx = select(root - x, x - root, x > root);
                d = dx * dx + height(site, base + root * stride);
            }
            switch G.axis {
                case 0u: {
                    if site == 0u {
                        to_solid[i] = d;
                    } else {
                        to_free[i] = d;
                    }
                }
                case 1u: {
                    if site == 0u {
                        next_solid[i] = d;
                    } else {
                        next_free[i] = d;
                    }
                }
                default: {
                    if is_set(i) != (site == 0u) {
                        to_solid[i] = d;
                    }
                }
            }
        }
    }
}

// The stored values of two points, as half floats, into `to_free`.
@compute @workgroup_size(WORKGROUP)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let w = invocation(gid, groups);
    if w >= (G.points + 1u) / 2u {
        return;
    }
    var word = 0u;
    for (var b = 0u; b < 2u; b++) {
        let n = w * 2u + b;
        if n >= G.points {
            break;
        }
        let entry = table[to_solid[n]];
        word |= select(entry & 0xffffu, entry >> 16u, is_set(n)) << (16u * b);
    }
    to_free[w] = word;
}
