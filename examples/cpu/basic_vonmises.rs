extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// `VonMisesMaterial` interactive showcase -- closes the last real Tier-0
/// gap for this material (previously only incidental mentions in
/// `validate_materials.rs`'s headless sweeps and `rod_blade_and_root.rs`,
/// no real interactive scene anywhere in the repo).
///
/// Three blobs, same drop. LEFT and MIDDLE share one elastic stiffness
/// (lambda=30, mu=60 -- grid-native, NOT migrated to real SI: this was
/// identical to `basic_jellies.rs`'s own `CorotatedMaterial` blob when
/// written, but jellies has since moved to real E=500 Pa soft tissue, and
/// every yield_stress/hardening_modulus below is expressed as a MU-relative
/// ratio, tuned through several documented empirical passes against THIS
/// elastic wave speed at THIS drop height/gravity -- rescaling mu would
/// shift the impact-strain-vs-wave-speed relationship those passes
/// calibrated against, not just the absolute numbers, so it needs the same
/// real drop-height/gravity re-sweep basic_jellies.rs went through, not a
/// direct substitution. Real, disclosed, deferred, not silently dropped).
/// RIGHT is five times stiffer; `make_sim` says why a higher yield alone
/// could not make it resist the impact:
///
///   - LEFT   (soft, perfect plasticity): yield_stress=mu*0.01,
///     hardening_modulus=0 -- dents on impact and STAYS dented; hit it again
///     and it dents by roughly the same amount each time (no memory of prior
///     yielding).
///   - MIDDLE (soft, hardening):          yield_stress=mu*0.01,
///     hardening_modulus=mu*0.03 -- dents a lot on the FIRST hit, then
///     visibly resists more on each subsequent hit as its own yield surface
///     grows (kappa printed live below makes this literal, not just visual).
///   - RIGHT  (stiff): lambda and mu five times LEFT's, yield_stress=0.05 of
///     its own mu. Meant to stay close to a bare elastic solid, but measured
///     headless it does not: on landing every particle is past 0.01 of
///     accumulated plastic strain and the median is 0.88
///     (`tests/scratch_stress_view_before_after.rs`). Rebuilding the three
///     blobs from real metals is issue #46.
///
/// That last point was the intent, not what the scene is measured to do, and
/// whether the RIGHT blob still keeps bouncing has not been re-measured
/// since. The intent: unlike `basic_jellies.rs`'s NeoHookean/Corotated blobs
/// (which have NO damping of any kind and bounce indefinitely, a real,
/// disclosed, accepted property of that demo), a VonMises blob that actually
/// yields dissipates real energy irreversibly through plastic flow and
/// settles ON ITS OWN -- no Cundall damping or other numerical relaxation is
/// enabled in this scene. The LEFT and MIDDLE blobs settling while the RIGHT
/// one keeps bouncing was meant to be the demo: plasticity as real, physical,
/// mechanical damping, not a numerical crutch.
///
///   LMB push  RMB pull  V toggle own-yield view  R reset  Q quit
///   cargo run --example basic_vonmises --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{SimConfig, Simulation, SlipBoundary, SpawnRegion, VonMisesMaterial};
use glam::{IVec2, Vec2};
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

const GRID: usize = 64;
const DT: f32 = 0.1;
const LAMBDA: f32 = 30.0;
const MU: f32 = 60.0;

const MAT_SOFT: u32 = 0;
const MAT_HARD: u32 = 1;
const MAT_STIFF: u32 = 2;

/// The scene, and its three materials in slot order: the stress view reads
/// each particle against its own material's yield surface.
fn make_sim(gravity_fraction: f32) -> (Simulation, [VonMisesMaterial; 3]) {
    let mut config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 8,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    config.gravity *= gravity_fraction;

    let spawn = |c: Vec2, mat| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(14, 14),
        box_center: c,
        material_id: mat,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };

    // Real regression fix (calibration, not correctness), second pass: the
    // first attempt (real yield/mu ratios, same modulus for all three, just
    // a shorter drop) still wasn't enough -- measured live, "stiff" kept
    // yielding almost as much as "soft" (kappa 18.6 vs 20.1 at frame 240).
    // Root cause: peak contact strain under ANY real, visible impact scales
    // with impact velocity over the material's OWN elastic wave speed
    // (c = sqrt((lambda+2mu)/rho)) -- picking a bigger YIELD NUMBER on the
    // same soft base modulus can't make a material behave stiffer under
    // impact, because c never changed. Real materials that resist impact
    // without yielding are stiffer in absolute modulus, not just in yield
    // threshold (steel vs. clay differ in E, not only in sigma_Y/E) -- so
    // "stiff" now gets a genuinely higher lambda/mu (5x), which raises its
    // own c and lowers its impact-induced strain directly, on top of the
    // same real yield/mu ratio range `VonMisesMaterial`'s own cited lava/clay
    // values use (~0.3%-5%). Gravity also cut further for a gentler, resolvable
    // impact rather than another shock.
    let soft = VonMisesMaterial::new(LAMBDA, MU, MU * 0.01);
    // Real regression fix (calibration, found by the scripted stress test):
    // hardening_modulus=mu*0.15 made hard's effective yield surface
    // (yield_stress + hardening_modulus*kappa) rocket past anything the
    // scripted pushes could reach after just the initial drop impact --
    // measured live, kappa froze at EXACTLY 5.5215 across all 3 subsequent
    // hits, zero further increment. That's "resists completely," not the
    // doc's own claimed "resists progressively" -- a real, honest gap this
    // stress test's own kappa-per-hit tracking exists to catch. Lowered so
    // later hits still add real, visible, shrinking increments instead of
    // saturating after one impact.
    let hard = VonMisesMaterial::with_hardening(LAMBDA, MU, MU * 0.01, MU * 0.03);
    let stiff_lambda = LAMBDA * 5.0;
    let stiff_mu = MU * 5.0;
    let stiff = VonMisesMaterial::new(stiff_lambda, stiff_mu, stiff_mu * 0.05);

    let mut solver = Simulation::new(config, spawn(Vec2::new(14.0, 20.0), MAT_SOFT))
        .with_default_material(Box::new(soft))
        .with_material(MAT_HARD, Box::new(hard))
        .with_material(MAT_STIFF, Box::new(stiff))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(32.0, 20.0), MAT_HARD));
    let _ = solver.add_body(spawn(Vec2::new(50.0, 20.0), MAT_STIFF));
    (solver, [soft, hard, stiff])
}

/// Per-material worst-case readout -- `friction_hardening` IS kappa
/// (accumulated equivalent plastic strain) for this material, per
/// `von_mises.rs`'s own doc ("kappa is accumulated into
/// `Particle::friction_hardening` each substep"). Real, literal evidence
/// that MAT_HARD's yield surface is actually growing over repeated hits,
/// not just a visual impression.
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
        "LIVE frame={frame} soft(kappa={:.3} |J-1|={:.3} vmax={:.4} vcom={:.5}) hard(kappa={:.3} |J-1|={:.3} vmax={:.4} vcom={:.5}) stiff(kappa={:.3} |J-1|={:.3} vmax={:.4} vcom={:.5})",
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
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let gravity_fraction = 0.003;
        let (sim, materials) = make_sim(gravity_fraction);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.6, true);
        // Real physically-grounded shading instead of the ByVolume debug heat
        // map (see basic_membrane.rs's own note on this same swap). Von Mises
        // here models clay/ductile soil -- reuse SOIL's own cited absorption
        // spectrum (Baumgardner et al. 1985, humic-acid-dominated soil
        // reflectance) already recorded in this engine's render plan, scaled
        // by the same disclosed factor as basic_membrane.rs's TISSUE constant
        // for consistency across examples. Same real spectrum for all three
        // blobs -- they're the same material family (only yield/hardening
        // differ), no invented per-blob optical distinction.
        const SOIL_SIGMA_A: [f32; 3] = [0.200, 0.275, 0.550];
        renderer.set_color_mode(ColorMode::ByPhysics);
        for slot in [MAT_SOFT, MAT_HARD, MAT_STIFF] {
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
            "basic_vonmises: {} particles (3 blobs: soft/hardening/stiff)  |  LMB push  RMB pull  V toggle own-yield view  R reset  Q quit",
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
            show_stress: false,
            materials,
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
    }

    fn update_and_render(&mut self, window: &Window) {
        let base_gravity = SimConfig::earth(GRID, 0.01, DT).gravity;
        self.sim.set_gravity(base_gravity * self.gravity_fraction);

        let g = self.sim.config().gravity.length();
        let stress_test = std::env::var("VONMISES_STRESS_TEST").is_ok();
        if stress_test {
            // Real, scripted stress test (same discipline as basic_membrane.rs's
            // MEMBRANE_STRESS_TEST): settle first, then repeatedly PUSH each
            // blob in turn with real rest gaps between hits so kappa's
            // per-hit increment is directly readable -- this is the concrete
            // test of this material's own doc claim ("dents a lot on the
            // FIRST hit, then visibly resists more on each subsequent hit"),
            // not just a generic robustness check.
            const SETTLE: u64 = 90;
            const HIT: u64 = 40;
            const REST: u64 = 20;
            const HITS_PER_BLOB: u64 = 3;
            const CYCLE: u64 = HIT + REST;
            let blob_x = [14.0f32, 32.0, 50.0];
            let blob_names = ["soft", "hard", "stiff"];
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
            // One scale for all three blobs cannot say that: the hardening
            // blob's yield grows with kappa and the stiff blob's is 25 times
            // the soft one's (`tests/scratch_stress_view_before_after.rs`).
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
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.separator();
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=15.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=15.0));
                    ui.separator();
                    ui.label("Left = soft/perfect plasticity");
                    ui.label("Middle = soft, hardens as it yields");
                    ui.label("Right = 5x stiffer, yields on landing too (issue #46)");
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
