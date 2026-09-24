extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Isolation test for `project_rolling_snowball_demo_attempt_2026-08-16`:
/// a bare kinematic ball on a slope, NO snow, NO accretion -- just gravity
/// + the slope-normal contact correction. If this alone doesn't roll
///   downhill and approach Rubin 2019's real (1/6)*g*sin(theta) target, the
///   bug is in the slope-normal physics itself, not the accretion/snow
///   interaction (rules one of the two suspected confounds in/out).
use glam::Vec2;

const GRID: usize = 64;
const SLOPE_START_X: usize = 4;
const SLOPE_END_X: usize = 34;
const SLOPE_START_H: f32 = 26.0;
const SLOPE_END_H: f32 = 10.0;
const RADIUS: f32 = 2.0;
const DT: f32 = 0.01; // fine dt, no MPM substep coupling to worry about here

fn heightmap() -> Vec<f32> {
    (0..GRID)
        .map(|x| {
            if x < SLOPE_START_X {
                SLOPE_START_H
            } else if x < SLOPE_END_X {
                let t = (x - SLOPE_START_X) as f32 / (SLOPE_END_X - SLOPE_START_X) as f32;
                SLOPE_START_H + t * (SLOPE_END_H - SLOPE_START_H)
            } else {
                SLOPE_END_H
            }
        })
        .collect()
}

fn main() {
    let heights = heightmap();
    let dh = SLOPE_START_H - SLOPE_END_H;
    let dx = (SLOPE_END_X - SLOPE_START_X) as f32;
    let theta = (dh / dx).atan();
    let target_accel = (1.0 / 6.0) * 9.81 * theta.sin();
    println!(
        "slope angle={:.1}deg  Rubin target accel=(1/6)*g*sin(theta)={:.4}",
        theta.to_degrees(),
        target_accel
    );

    let mut pos = Vec2::new(SLOPE_START_X as f32 + 1.0, SLOPE_START_H + RADIUS + 2.0);
    let mut vel = Vec2::ZERO;
    let gravity = Vec2::new(0.0, -9.81);

    const STEPS: usize = 3000;
    let mut last_speed = 0.0f32;
    let mut speed_at = [0.0f32; 4];
    let checkpoints = [500, 1000, 2000, 2999];
    let mut ci = 0;

    for step in 0..STEPS {
        vel += gravity * DT;
        pos += vel * DT;

        let col = (pos.x.round() as isize).clamp(1, GRID as isize - 2) as usize;
        let terrain_h = heights[col];
        let dh_dx = (heights[col + 1] - heights[col - 1]) / 2.0;
        let normal = Vec2::new(-dh_dx, 1.0).normalize();
        let terrain_point = Vec2::new(col as f32, terrain_h);
        let perp_dist = (pos - terrain_point).dot(normal);
        if perp_dist < RADIUS {
            pos += normal * (RADIUS - perp_dist);
            let v_n = vel.dot(normal);
            if v_n < 0.0 {
                vel -= v_n * normal;
            }
        }
        if pos.x < 2.0 + RADIUS {
            pos.x = 2.0 + RADIUS;
            vel.x = vel.x.max(0.0);
        }

        last_speed = vel.length();
        if ci < checkpoints.len() && step == checkpoints[ci] {
            speed_at[ci] = last_speed;
            println!(
                "step={step:>4} pos=({:.3},{:.3}) vel=({:.4},{:.4}) speed={:.4}",
                pos.x, pos.y, vel.x, vel.y, last_speed
            );
            ci += 1;
        }
    }

    println!("\nfinal speed={last_speed:.4} after {STEPS} steps (dt={DT})");
    if speed_at[1] > speed_at[0] + 1.0e-3 {
        let accel_estimate =
            (speed_at[2] - speed_at[1]) / ((checkpoints[2] - checkpoints[1]) as f32 * DT);
        println!(
            "ROLLING CONFIRMED: speed genuinely increasing over time. Rough accel estimate (checkpoint 1->2) = {:.4} vs Rubin target {:.4}",
            accel_estimate, target_accel
        );
    } else {
        println!(
            "NOT ROLLING: speed did not meaningfully increase between early checkpoints -- slope-normal physics itself is the bug, not accretion."
        );
    }
}
