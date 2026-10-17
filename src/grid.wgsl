// Distance grids as the kernels read them, included in kernels.wgsl, which binds `grid_data`:
// every grid's half floats, two per word.

fn grid_value(o: Obstacle, n: u32) -> f32 {
    let pair = unpack2x16float(grid_data[o.grid + (n >> 1u)]);
    return select(pair.x, pair.y, (n & 1u) == 1u);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Returns (gradient, signed distance) at p in the grid's frame. Mirrors grid_distance in sdf.rs.
fn grid_distance(o: Obstacle, p: vec3<f32>) -> vec4<f32> {
    let voxel = o.half.w;
    let top = vec3<f32>(f32(o.nx), f32(o.ny), f32(o.nz)) - 1.0;
    let x = (p - o.half.xyz) / voxel;
    let c = clamp(x, vec3<f32>(0.0), top);
    let i = min(floor(c), top - 1.0);
    let f = c - i;
    let sy = o.nx;
    let sz = o.nx * o.ny;
    let n = u32(i.x) + sy * u32(i.y) + sz * u32(i.z);
    let v000 = grid_value(o, n);
    let v100 = grid_value(o, n + 1u);
    let v010 = grid_value(o, n + sy);
    let v110 = grid_value(o, n + sy + 1u);
    let v001 = grid_value(o, n + sz);
    let v101 = grid_value(o, n + sz + 1u);
    let v011 = grid_value(o, n + sz + sy);
    let v111 = grid_value(o, n + sz + sy + 1u);
    let x00 = lerp(v000, v100, f.x);
    let x10 = lerp(v010, v110, f.x);
    let x01 = lerp(v001, v101, f.x);
    let x11 = lerp(v011, v111, f.x);
    let y0 = lerp(x00, x10, f.y);
    let y1 = lerp(x01, x11, f.y);
    let dx = lerp(lerp(v100 - v000, v110 - v010, f.y), lerp(v101 - v001, v111 - v011, f.y), f.z);
    let inside = vec3<f32>(dx, lerp(x10 - x00, x11 - x01, f.z), y1 - y0) / voxel;
    let outside = (x - c) * voxel;
    let away = length(outside);
    let g = select(inside, outside / max(away, 1e-30), x != c);
    return vec4<f32>(g, lerp(y0, y1, f.z) + away);
}
