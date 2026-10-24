extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;
#[path = "vonmises_clay_scene.rs"]
mod vonmises_clay_scene;
use vonmises_clay_scene::*;

use emerge::diagnostics::{HEAT_BANDS, OCCUPANCY_BANDS, SimSnapshot, scene_map};
use emerge::particle::Particle;
/// `VonMisesMaterial` interactive showcase: three blobs of real saturated
/// clay at three consistencies, dropped under real gravity at 1 cm cells.
///
/// Saturated clay loaded faster than it drains is the standard case of
/// von Mises (Tresca) plasticity in soil mechanics: its undrained strength
/// is the threshold, its undrained modulus the stiffness. Constants, read
/// on the documents (`CLAYS` below for each value's source):
///
///   - LEFT   very soft clay: unconfined strength 20 kPa, E 3 MPa
///   - MIDDLE soft clay:      37.5 kPa, E 5.6 MPa
///   - RIGHT  medium clay:    75 kPa, E 11 MPa
///
/// Each holds its own weight (`make_sim` checks it before the scene starts,
/// 10 to 38 times over at this size). Measured headless on this scene
/// (`tests/probes/vonmises_clay.rs`), after a 10 cm drop at 1.4 m/s: the
/// three spread to 17.8, 16.1 and 14.9 cells wide from 13.5, with largest
/// accumulated plastic strains of 0.50, 0.31 and 0.19, keep their volume,
/// and come to rest without bouncing, none left at yield. No damping of any
/// kind is enabled; the energy goes into plastic flow.
///
/// Slow by physics, not by choice: undrained clay is nearly
/// incompressible, so its pressure waves run at 80 to 160 m/s and an
/// explicit step at 1 cm must stay near 1/40 000 s, about 650 substeps a
/// frame. Measured: simulated time runs at 0.027 of real time in the dev
/// profile. The panel shows the live ratio.
///
///   LMB push  RMB pull  V toggle own-yield view  R reset  Q quit
///   cargo run --example basic_vonmises --features render
///
/// Headless reading of the own-yield view, for a reviewer without the
/// screen: `VONMISES_START_VIEW=yield` starts with it on (so a capture
/// through `VONMISES_CAPTURE_DIR` shows it), and `VONMISES_LOG=<file>`
/// writes a `FrameLogger` line per frame with, per blob, the share of its
/// particles at yield (`yield_ratio` at least 0.99), its mean ratio, and
/// how many particles fell back from at least 0.99 to under 0.9 since the
/// frame before, and a text picture of what the screen shows
/// (`scene_map`): the own-yield view in the colour map's bands (`.` blue,
/// `:` teal, `-` green, `+` yellow to orange, `#` red), or where material
/// is.
/// `VONMISES_STRESS_TEST=1` scripts the pushes.
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DiagnosticsPlugin, DiagnosticsRegistry, FrameLogger, SimConfig, Simulation, VonMisesMaterial,
    per_material_stats,
};
use glam::Vec2;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn read_full_frame_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vonmises_capture_readback_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vonmises_capture_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let mut tight = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        tight.extend_from_slice(&mapped[start..start + unpadded_bytes_per_row as usize]);
    }
    drop(mapped);
    staging.unmap();
    tight
}

/// Same real capture aid as `basic_membrane.rs`'s own `CaptureState` -- see
/// that file's doc for the mechanism. Opt-in via `VONMISES_CAPTURE_DIR`.
struct CaptureState {
    file: std::fs::File,
    path: std::path::PathBuf,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    stride: u64,
    target_frames: u32,
    captured: u32,
}

/// The own-yield view in numbers, as a diagnostics plugin: per blob, the
/// share of its particles at yield (`yield_ratio` at least 0.99), its mean
/// ratio, and how many fell back from at least 0.99 to under 0.9 since the
/// frame before. Stateful for that last count; particle order is stable in
/// this scene, which removes none.
struct OwnYieldPlugin {
    materials: [VonMisesMaterial; 3],
    last: Vec<f32>,
}

impl DiagnosticsPlugin for OwnYieldPlugin {
    fn name(&self) -> &'static str {
        "own_yield"
    }

    fn collect(&mut self, particles: &[Particle], _snapshot: &SimSnapshot) -> Vec<(String, f32)> {
        let ratio: Vec<f32> = particles
            .iter()
            .map(|p| {
                self.materials[p.material_id as usize]
                    .yield_ratio_of(p.deformation_gradient, p.friction_hardening)
            })
            .collect();
        let mut out = Vec::with_capacity(9);
        for (slot, blob) in ["very_soft", "soft", "medium"].iter().enumerate() {
            let (mut n, mut at, mut sum, mut fell) = (0usize, 0usize, 0.0f32, 0usize);
            for (i, (p, &r)) in particles.iter().zip(&ratio).enumerate() {
                if p.material_id != slot as u32 {
                    continue;
                }
                n += 1;
                at += usize::from(r >= 0.99);
                sum += r;
                let was_at = self.last.get(i).is_some_and(|&last| last >= 0.99);
                fell += usize::from(was_at && r < 0.9);
            }
            let n = n.max(1) as f32;
            out.push((format!("{blob}_at_yield"), at as f32 / n));
            out.push((format!("{blob}_ratio_mean"), sum / n));
            out.push((format!("{blob}_fell_back"), fell as f32));
        }
        self.last = ratio;
        out
    }
}

fn own_yield_diagnostics(materials: [VonMisesMaterial; 3]) -> DiagnosticsRegistry {
    DiagnosticsRegistry::new().with(Box::new(OwnYieldPlugin {
        materials,
        last: Vec::new(),
    }))
}

/// Per-material worst-case readout -- `friction_hardening` IS kappa
/// (accumulated equivalent plastic strain) for this material, per
/// `von_mises.rs`'s doc ("kappa is accumulated into
/// `Particle::friction_hardening` each substep"): how far each clay has
/// flowed, not just how it looks.
fn print_diagnostic(sim: &Simulation, frame: u64) {
    let mut kappa_max = [0.0f32; 3];
    let mut j_dev_max = [0.0f32; 3];
    let mut speed_max = [0.0f32; 3];
    let mut v_sum = [Vec2::ZERO; 3];
    let mut count = [0u32; 3];
    for p in sim.particles().iter() {
        let slot = p.material_id as usize;
        if slot >= 3 {
            continue;
        }
        kappa_max[slot] = kappa_max[slot].max(p.friction_hardening);
        j_dev_max[slot] = j_dev_max[slot].max((p.deformation_gradient.determinant() - 1.0).abs());
        speed_max[slot] = speed_max[slot].max(p.v.length());
        v_sum[slot] += p.v;
        count[slot] += 1;
    }
    let v_com = |slot: usize| {
        if count[slot] == 0 {
            0.0
        } else {
            (v_sum[slot] / count[slot] as f32).length()
        }
    };
    println!(
        "LIVE frame={frame} very_soft(kappa={:.3} |J-1|={:.3} vmax={:.4} vcom={:.5}) soft(kappa={:.3} |J-1|={:.3} vmax={:.4} vcom={:.5}) medium(kappa={:.3} |J-1|={:.3} vmax={:.4} vcom={:.5})",
        kappa_max[0],
        j_dev_max[0],
        speed_max[0],
        v_com(0),
        kappa_max[1],
        j_dev_max[1],
        speed_max[1],
        v_com(1),
        kappa_max[2],
        j_dev_max[2],
        speed_max[2],
        v_com(2)
    );
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    cursor_force: cursor_force::CursorForce,
    gravity_fraction: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    capture: Option<CaptureState>,
    // Toggled with V: each particle against its OWN yield surface
    // (`VonMisesMaterial::yield_ratio`), since the whole point of this scene
    // IS watching where/when a material actually crosses it -- visible
    // directly instead of only inferred from shape change afterward. Red
    // says "at or beyond its yield", never how far beyond: a particle the
    // return mapping has put back on its surface reads 1 however much it
    // has flowed. How far is kappa (`friction_hardening`), printed live.
    show_stress: bool,
    /// The three materials, slot order, for that view.
    materials: [VonMisesMaterial; 3],
    /// `VONMISES_LOG`: the own-yield view in numbers, per blob, per frame,
    /// through `OwnYieldPlugin`.
    yield_log: Option<(FrameLogger, DiagnosticsRegistry)>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let gravity_fraction = 1.0;
        let (sim, materials) = make_sim(gravity_fraction);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.6, true);
        // Physically grounded shading instead of the ByVolume debug heat map (see
        // basic_membrane.rs). Von Mises models clay or ductile soil here, so this
        // reuses SOIL's cited absorption spectrum (Baumgardner et al. 1985,
        // humic-acid-dominated soil reflectance), scaled by the same factor as
        // basic_membrane.rs's TISSUE constant. One spectrum for all three blobs: they
        // are one material family (only yield and hardening differ).
        const SOIL_SIGMA_A: [f32; 3] = [0.200, 0.275, 0.550];
        let start_on_yield = std::env::var("VONMISES_START_VIEW").is_ok_and(|v| v == "yield");
        renderer.set_color_mode(if start_on_yield {
            ColorMode::ByStress
        } else {
            ColorMode::ByPhysics
        });
        let yield_log = std::env::var("VONMISES_LOG").ok().map(|path| {
            let log = FrameLogger::open(path).expect("failed to open VONMISES_LOG");
            (log, own_yield_diagnostics(materials))
        });
        for slot in [MAT_VERY_SOFT, MAT_SOFT, MAT_MEDIUM] {
            renderer.set_optical_params(&gfx.queue, slot as usize, SOIL_SIGMA_A);
            renderer.set_optical_scattering(&gfx.queue, slot as usize, 0.02);
            renderer.set_specular_r0(&gfx.queue, slot as usize, 0.02);
            renderer.set_refractive_index(slot as usize, 1.5); // real soil index, Baumgardner 1985
        }

        let capture = std::env::var("VONMISES_CAPTURE_DIR").ok().map(|dir_str| {
            let dir = std::path::PathBuf::from(dir_str);
            std::fs::create_dir_all(&dir).expect("failed to create VONMISES_CAPTURE_DIR");
            let target_frames: u32 = std::env::var("VONMISES_CAPTURE_FRAMES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(150);
            let stride: u64 = std::env::var("VONMISES_CAPTURE_STRIDE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let texture = gfx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vonmises_capture_target"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: gfx.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            std::fs::write(
                dir.join("format.txt"),
                format!("{:?} {} {}", gfx.format, size.width, size.height),
            )
            .expect("failed to write capture format.txt");
            let path = dir.join("frames.raw");
            let file = std::fs::File::create(&path).expect("failed to create frames.raw");
            println!(
                "[capture] writing up to {target_frames} frames (stride={stride}, format={:?}, {}x{}) to {path:?}",
                gfx.format, size.width, size.height
            );
            CaptureState {
                file,
                path,
                texture,
                width: size.width,
                height: size.height,
                stride,
                target_frames,
                captured: 0,
            }
        });

        println!(
            "basic_vonmises: {} particles (3 blobs of clay: very soft/soft/medium)  |  LMB push  RMB pull  V toggle own-yield view  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            gfx,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            cursor_force: cursor_force::CursorForce::new(5.0, 5.0, 5.0),
            gravity_fraction,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            capture,
            show_stress: start_on_yield,
            materials,
            yield_log,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            GRID,
        )
    }

    fn reset(&mut self) {
        (self.sim, self.materials) = make_sim(self.gravity_fraction);
        self.frame = 0;
        if let Some((_, diagnostics)) = &mut self.yield_log {
            *diagnostics = own_yield_diagnostics(self.materials);
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        let base_gravity = SimConfig::earth(GRID, 0.01, DT).gravity;
        self.sim.set_gravity(base_gravity * self.gravity_fraction);

        let g = self.sim.config().gravity.length();
        let stress_test = std::env::var("VONMISES_STRESS_TEST").is_ok();
        if stress_test {
            // Scripted stress test (same discipline as basic_membrane.rs's
            // MEMBRANE_STRESS_TEST): settle first, then repeatedly PUSH each
            // blob in turn with real rest gaps between hits so kappa's
            // per-hit increment is directly readable -- this is the concrete
            // test of this material's doc claim ("dents a lot on the
            // FIRST hit, then visibly resists more on each subsequent hit"),
            // not just a generic robustness check.
            const SETTLE: u64 = 90;
            const HIT: u64 = 40;
            const REST: u64 = 20;
            const HITS_PER_BLOB: u64 = 3;
            const CYCLE: u64 = HIT + REST;
            let blob_x = [14.0f32, 32.0, 50.0];
            let blob_names = ["very_soft", "soft", "medium"];
            if self.frame >= SETTLE {
                let t = self.frame - SETTLE;
                let phase = (t / (CYCLE * HITS_PER_BLOB)).min(2) as usize;
                let within = t % (CYCLE * HITS_PER_BLOB);
                let hit_index = within / CYCLE;
                let in_hit = within % CYCLE < HIT;
                if in_hit {
                    let cursor = Vec2::new(blob_x[phase], 12.0);
                    self.cursor_force
                        .apply(self.sim.particles_mut(), cursor, g, DT, false);
                    if within.is_multiple_of(CYCLE) {
                        let kappa = self
                            .sim
                            .particles()
                            .iter()
                            .filter(|p| p.material_id == phase as u32)
                            .map(|p| p.friction_hardening)
                            .fold(0.0f32, f32::max);
                        println!(
                            "[stress] {} hit #{} starting, kappa before={:.4}",
                            blob_names[phase],
                            hit_index + 1,
                            kappa
                        );
                    }
                }
            }
        } else {
            let cursor = self.cursor_grid();
            if self.lmb {
                self.cursor_force
                    .apply(self.sim.particles_mut(), cursor, g, DT, false);
            }
            if self.rmb {
                self.cursor_force
                    .apply(self.sim.particles_mut(), cursor, g, DT, true);
            }
        }

        self.sim.step();

        if let Some((log, diagnostics)) = &mut self.yield_log {
            let snapshot = self.sim.diagnostics_snapshot();
            let particles = self.sim.particles().to_vec();
            let frame = diagnostics.collect(&particles, &snapshot);
            let extra: Vec<(&str, f32)> = frame.iter().collect();
            log.log(
                self.frame,
                DT,
                &per_material_stats(self.sim.particles()),
                &snapshot,
                &[
                    (MAT_VERY_SOFT, "very_soft"),
                    (MAT_SOFT, "soft"),
                    (MAT_MEDIUM, "medium"),
                ],
                &extra,
            );
            // What the screen shows, as text: the camera frames the whole
            // grid, one character per cell across and two cells per
            // character up, since a character is about twice as tall as wide.
            let p = self.sim.particles();
            let region = (Vec2::ZERO, Vec2::splat(GRID as f32));
            if self.show_stress {
                let ratio: Vec<f32> = (0..p.len())
                    .map(|i| self.materials[p.material_id[i] as usize].yield_ratio(p, i))
                    .collect();
                let map = scene_map(p, region, GRID, GRID / 2, |i| ratio[i], &HEAT_BANDS);
                log.log_map(self.frame, "own_yield", &map);
            } else {
                let map = scene_map(p, region, GRID, GRID / 2, |_| 1.0, &OCCUPANCY_BANDS);
                log.log_map(self.frame, "occupancy", &map);
            }
        }

        if stress_test {
            for p in self.sim.particles().iter() {
                let bad = !p.x.is_finite()
                    || !p.v.is_finite()
                    || !p.volume.is_finite()
                    || p.volume <= 0.0
                    || !p.density.is_finite()
                    || p.density <= 0.0;
                if bad {
                    println!(
                        "STRESS FAIL frame={} x={:?} v={:?} volume={} density={}",
                        self.frame, p.x, p.v, p.volume, p.density
                    );
                    std::process::exit(1);
                }
            }
            const TOTAL: u64 = 90 + (40 + 20) * 3 * 3;
            if self.frame >= TOTAL {
                println!("[stress] done -- {TOTAL} frames, no admissibility failure");
                std::process::exit(0);
            }
        }

        if self.frame.is_multiple_of(120) {
            print_diagnostic(&self.sim, self.frame);
        }

        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        if self.show_stress {
            // Each particle against its own yield surface, by the code its
            // material's return mapping runs: below 1 elastic, 1 on the
            // surface, which is where a particle flowing plastically sits
            // whether it has just reached yield or flowed a long way.
            // One scale for all three blobs cannot say that: the medium clay's
            // yield is 3.75 times the very soft one's.
            let p = self.sim.particles();
            let ratio: Vec<f32> = (0..p.len())
                .map(|i| self.materials[p.material_id[i] as usize].yield_ratio(p, i))
                .collect();
            self.renderer.set_stress_field(ratio);
            self.renderer.set_stress_scale(1.0);
        }

        let output = match self.gfx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(
            &self.gfx.device,
            &self.gfx.queue,
            self.sim.particles(),
            &view,
            true,
        );

        if let Some(cap) = &mut self.capture
            && cap.captured < cap.target_frames
            && self.frame.is_multiple_of(cap.stride)
        {
            let capture_view = cap
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.renderer.render(
                &self.gfx.device,
                &self.gfx.queue,
                self.sim.particles(),
                &capture_view,
                true,
            );
            let raw = read_full_frame_rgba(
                &self.gfx.device,
                &self.gfx.queue,
                &cap.texture,
                cap.width,
                cap.height,
            );
            use std::io::Write;
            cap.file
                .write_all(&raw)
                .unwrap_or_else(|e| panic!("failed to append frame to {:?}: {e}", cap.path));
            cap.captured += 1;
            if cap.captured.is_multiple_of(30) || cap.captured == cap.target_frames {
                println!(
                    "[capture] {}/{} frames written",
                    cap.captured, cap.target_frames
                );
            }
            if cap.captured >= cap.target_frames {
                cap.file.flush().ok();
                println!(
                    "[capture] done -- {} frames in {:?}",
                    cap.captured, cap.path
                );
                std::process::exit(0);
            }
        }

        let fps = self.last_fps;
        let mut gravity_fraction = self.gravity_fraction;
        let mut push_strength = self.cursor_force.push_strength;
        let mut pull_strength = self.cursor_force.pull_strength;
        let n_particles = self.sim.particles().len();
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Von Mises (elastoplastic)")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.1}  particles={n_particles}"));
                    // Each frame advances DT of simulated time, whatever it costs.
                    ui.label(format!("simulated time runs at {:.3}x real time", fps * DT));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.separator();
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=15.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=15.0));
                    ui.separator();
                    ui.label("Real saturated clay, undrained (FHWA, NAVFAC):");
                    ui.label("Left = very soft, 20 kPa, E 3 MPa");
                    ui.label("Middle = soft, 37.5 kPa, E 5.6 MPa");
                    ui.label("Right = medium, 75 kPa, E 11 MPa");
                    ui.label("Color = soil optics; V = each particle against its own yield:");
                    ui.label("  red = at or beyond yield (not how far), blue = well inside");
                    ui.separator();
                    ui.label("LMB push  RMB pull  V toggle own-yield view  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.gravity_fraction = gravity_fraction;
        self.cursor_force.push_strength = push_strength;
        self.cursor_force.pull_strength = pull_strength;
        if reset {
            self.reset();
        }

        output.present();
    }
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Von Mises (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 640u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        if let Some(w) = &self.window {
            let resp = s.gfx.egui_state.on_window_event(w, &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => s.lmb = state == ElementState::Pressed,
                MouseButton::Right => s.rmb = state == ElementState::Pressed,
                _ => {}
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == ElementState::Pressed;
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyR if pressed => {
                        s.reset();
                        println!("reset");
                    }
                    KeyCode::KeyV if pressed => {
                        s.show_stress = !s.show_stress;
                        s.renderer.set_color_mode(if s.show_stress {
                            ColorMode::ByStress
                        } else {
                            ColorMode::ByPhysics
                        });
                        println!(
                            "own-yield view {}",
                            if s.show_stress { "ON" } else { "off" }
                        );
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                if let Some(w) = &self.window {
                    let w = w.clone();
                    s.update_and_render(&w);
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    el.run_app(&mut app).unwrap();
}
