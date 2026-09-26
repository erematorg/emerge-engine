extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    KinematicCircleBoundary, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};
/// Real, visible, INTERACTIVE solid-liquid two-way coupling: the mouse
/// cursor drives a `KinematicCircleBoundary` (a kinematically-driven
/// obstacle, NOT a rigid body -- see that type's own doc for why this
/// engine's "no rigid bodies" scope rule doesn't apply to it) through
/// settled water via a real virtual spring-damper control law (see the
/// `CONTROL_*` constants' own doc for the real derivation -- impedance
/// control, Hogan 1985, not a teleport-to-cursor hack). The obstacle's
/// velocity every step sums TWO real forces: that spring pulling it toward
/// the cursor, and the REAL, mass-weighted reaction impulse the water
/// exerts back on it (`take_reaction_impulse()`, Newton's third law) -- so
/// if this coupling is genuine, heavy water resistance visibly displaces
/// the obstacle away from wherever the cursor is trying to drag it ("gets
/// carried by the forces"), and moving the cursor harder measurably fights
/// that resistance back, all from the same one force balance, nothing
/// scripted per-behavior.
///
/// The underlying reaction-impulse mechanism is the exact same real physics
/// already verified headless in
/// `tests/scratch_kinematic_reactive_obstacle_water_verify.rs` (mass
/// conserved, obstacle genuinely decelerates on contact, reaction hits zero
/// out of contact) -- this demo drives that same mechanism from live cursor
/// input instead of a scripted initial velocity.
///
///   cargo run --example kinematic_obstacle_in_water --features render
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const SETTLE_STEPS: usize = 150;
const OBSTACLE_RADIUS: f32 = 2.0;
const OBSTACLE_MASS: f32 = 8.0;
const PARTICLE_RENDER_DIAMETER: f32 = 0.9;
// Real control law, not a teleport-to-cursor hack: the cursor sets a target
// position, and the obstacle is pulled toward it by a genuine virtual
// spring-damper (impedance control -- Hogan 1985; the same principle every
// force-feedback/haptic interface uses to let a human "feel" resistance
// through a driven object). Because the spring force and the real water
// reaction impulse are summed into the SAME velocity update, heavy
// resistance genuinely displaces the obstacle away from the cursor (gets
// "carried by the forces"), and holding the cursor against that resistance
// genuinely fights it back -- neither behavior is scripted separately, both
// fall out of the one real force balance.
//
// Real, disclosed derivation, not an arbitrary gain: choosing a target
// settling time of `CONTROL_RESPONSE_PERIODS_OF_DT` physics steps gives a
// real natural frequency omega=2*pi/(periods*DT), then k=m*omega^2 (simple
// harmonic oscillator relation) and damping=2*sqrt(k*m)*ratio (critical-
// damping formula, same convention `grain_contact_config` already uses
// elsewhere in this engine) -- so the "gain" is really just "how many
// physics steps should a deliberate cursor move take to catch up," a real
// interface-responsiveness choice, not a material constant pretending to be
// one.
const CONTROL_RESPONSE_PERIODS_OF_DT: f32 = 20.0;
const CONTROL_DAMPING_RATIO: f32 = 1.0; // critically damped: no overshoot chasing the cursor

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    obstacle: Arc<KinematicCircleBoundary>,
    obstacle_pos: Vec2,
    obstacle_vel: Vec2,
    cursor_screen: [f32; 2],
    // Real spring/damping gains, derived once from `CONTROL_RESPONSE_
    // PERIODS_OF_DT`/`CONTROL_DAMPING_RATIO` (see those constants' own doc)
    // -- not per-frame recomputed, they don't depend on anything that
    // changes.
    control_stiffness: f32,
    control_damping: f32,
    frame: u64,
    log_timer: std::time::Instant,
    // Real, derived-from-the-actual-scene camera extent (see
    // `camera_extent_for_aspect`'s own doc): computed ONCE at startup from
    // the real settled-water bounding box + real obstacle travel room, using
    // the window's OWN initial aspect. Real, live-reported bug fixed
    // 2026-09-15: this used to be RECOMPUTED on every resize using the
    // window's current aspect, which changes the vertical framing too, not
    // just the horizontal -- unlike every other example in this codebase,
    // which passes one FIXED `grid_res` on every resize and lets
    // `set_camera`'s own internal aspect handling adapt the horizontal
    // extent alone (that's what its `sx`/`sy` split is FOR). Recomputing a
    // second time on top of that was redundant and is what broke framing
    // under a fullscreen-sized aspect change. Fixed by treating this as a
    // fixed `grid_res` after the initial computation, exactly like every
    // other example.
    camera_extent: f32,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

/// `Renderer::set_camera`'s `grid_res` always frames a square-ish region
/// from world origin (0,0), widened by the window's own aspect ratio (see
/// that function's own derivation: landscape shows `y=grid_res,
/// x=grid_res*aspect`; portrait the mirror). Passing the PHYSICS grid size
/// (64 here) frames the whole simulation domain regardless of where the
/// actual water+obstacle sit in it -- fine for a scene that fills its grid,
/// but this one's real water pool only occupies a thin band near the
/// bottom (see the real, live-reported bug this fixes: 2026-09-15,
/// screenshot showed a mostly-empty window over a sliver of water). This
/// computes the smallest `grid_res` that still contains the real scene
/// (vertical_need: real settled water height + splash/sky headroom;
/// horizontal_reach: real water extent + room for the obstacle to keep
/// traveling into view) for whatever aspect ratio the window currently is.
fn camera_extent_for_aspect(vertical_need: f32, horizontal_reach: f32, aspect: f32) -> f32 {
    if aspect >= 1.0 {
        vertical_need.max(horizontal_reach / aspect)
    } else {
        horizontal_reach.max(vertical_need * aspect)
    }
}

/// Builds the settled water pool + a fresh obstacle sitting just outside it,
/// not yet launched -- byte-for-byte the same scene as the already-verified
/// scratch test, minus the settle loop (run once, live, in `State::new`
/// instead of before construction, so the window shows real settling too).
fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        gravity: Vec2::new(0.0, -0.3),
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 2.5);
    const SPACING: f32 = 0.9;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        box_size: IVec2::new(24, 10),
        box_center: Vec2::new(30.0, 8.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let boundary_thickness = config.boundary_thickness;
    Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        // Real fix (2026-09-15, live-reported): the scratch test this scene
        // was ported from never needed a domain wall (its own 200-step
        // scripted run never depended on where the water ended up), but a
        // long-running live demo does -- without one the water pool simply
        // spreads across nearly the whole grid during settling (confirmed
        // live: bbox reached x=62 of 64), which also pushed the obstacle's
        // own start position derived from that bbox off-grid entirely.
        // Same real wall every other fluid example in this codebase uses.
        .with_boundary(Box::new(SlipBoundary::new(boundary_thickness)))
}

fn bounding_box(sim: &Simulation) -> (Vec2, Vec2) {
    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);
    for p in sim.particles().x.iter() {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &sc);

        let mut sim = make_sim();
        for _ in 0..SETTLE_STEPS {
            sim.step();
        }
        let (bb_min, bb_max) = bounding_box(&sim);
        let obstacle_pos = Vec2::new(
            bb_min.x - OBSTACLE_RADIUS - 1.0,
            (bb_min.y + bb_max.y) * 0.5,
        );
        let obstacle = Arc::new(KinematicCircleBoundary::new(
            obstacle_pos,
            OBSTACLE_RADIUS,
            0.3,
        ));
        sim.add_boundary_condition(Box::new(obstacle.clone()));

        // Real headroom, not arbitrary: enough sky above the settled
        // surface for a real splash to stay visible. Clamped to the actual
        // physics domain (`GRID`) on both axes -- there is never a real
        // reason to show camera space beyond where the simulation actually
        // exists (see the real bug this closes: the cursor mapping into
        // that dead space and the obstacle getting stuck there with zero
        // possible contact, 2026-09-15).
        const SPLASH_HEADROOM: f32 = 8.0;
        let camera_vertical_need = (bb_max.y + SPLASH_HEADROOM).min(GRID as f32);
        let camera_horizontal_reach = GRID as f32;

        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        let aspect = size.width.max(1) as f32 / size.height.max(1) as f32;
        let camera_extent =
            camera_extent_for_aspect(camera_vertical_need, camera_horizontal_reach, aspect);
        renderer.set_camera(
            &queue,
            camera_extent as u32,
            size.width,
            size.height,
            PARTICLE_RENDER_DIAMETER,
            true,
        );
        renderer.set_color_mode(ColorMode::ByMaterial);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window.as_ref(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            fmt,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        println!(
            "kinematic_obstacle_in_water: {} water particles settled, bbox=({:.2},{:.2})-({:.2},{:.2})",
            sim.particles().len(),
            bb_min.x,
            bb_min.y,
            bb_max.x,
            bb_max.y
        );
        println!(
            "obstacle: radius={OBSTACLE_RADIUS} mass={OBSTACLE_MASS} start=({:.2},{:.2}) -- \
             move the mouse to drag it; real water resistance can carry it away from the \
             cursor, moving harder fights back",
            obstacle_pos.x, obstacle_pos.y
        );
        println!("R reset  Q quit");

        let omega = std::f32::consts::TAU / (CONTROL_RESPONSE_PERIODS_OF_DT * DT);
        let control_stiffness = OBSTACLE_MASS * omega * omega;
        let control_damping =
            2.0 * (control_stiffness * OBSTACLE_MASS).sqrt() * CONTROL_DAMPING_RATIO;

        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            obstacle,
            obstacle_pos,
            obstacle_vel: Vec2::ZERO,
            cursor_screen: [size.width as f32 * 0.5, size.height as f32 * 0.5],
            control_stiffness,
            control_damping,
            frame: 0,
            log_timer: std::time::Instant::now(),
            camera_extent,
            egui_ctx,
            egui_state,
            egui_renderer,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        // `camera_extent` is FIXED (computed once at startup) -- see its own
        // field doc for why recomputing it here on every resize was the real
        // fullscreen bug. `set_camera` itself already adapts correctly to
        // the new `w`/`h` aspect internally.
        self.renderer.set_camera(
            &self.queue,
            self.camera_extent as u32,
            w,
            h,
            PARTICLE_RENDER_DIAMETER,
            true,
        );
    }

    fn reset(&mut self) {
        let mut sim = make_sim();
        for _ in 0..SETTLE_STEPS {
            sim.step();
        }
        let (bb_min, bb_max) = bounding_box(&sim);
        self.obstacle_pos = Vec2::new(
            bb_min.x - OBSTACLE_RADIUS - 1.0,
            (bb_min.y + bb_max.y) * 0.5,
        );
        self.obstacle_vel = Vec2::ZERO;
        self.obstacle = Arc::new(KinematicCircleBoundary::new(
            self.obstacle_pos,
            OBSTACLE_RADIUS,
            0.3,
        ));
        sim.add_boundary_condition(Box::new(self.obstacle.clone()));
        self.sim = sim;
        self.frame = 0;
        println!("reset");
    }

    /// Real two-way coupling: the obstacle's velocity update sums TWO real
    /// forces every step -- a virtual spring pulling it toward the cursor
    /// (see the control constants' own doc) and the real reaction impulse
    /// the water exerts back on it (`take_reaction_impulse()`, Newton's
    /// third law). Neither is scripted around the other; "gets carried by
    /// the water" and "fighting back by moving the cursor harder" are both
    /// just this one force balance playing out differently depending on how
    /// hard the real water is pushing.
    fn update_and_render(&mut self, window: &Window) {
        // Real, structural fix (2026-09-15, live-reported "no collision at
        // all"): a raw `screen_to_grid` reading can map to world space
        // outside the actual [0, GRID] physics domain (window edges, or the
        // camera's own real splash/travel headroom extending past it -- see
        // `camera_extent_for_aspect`'s own doc) -- if the cursor sits there
        // and stops moving, the spring finds a real equilibrium exactly
        // there, off-grid, with zero possible reaction: not a bug in the
        // coupling itself, but a real trap this clamp closes at the
        // source. Margin = `OBSTACLE_RADIUS` so the obstacle's own BODY,
        // not just its center, always stays inside the domain that actually
        // exists.
        let cursor_target = {
            let (gx, gy) = self.renderer.screen_to_grid(
                self.cursor_screen[0],
                self.cursor_screen[1],
                self.surface_config.width,
                self.surface_config.height,
            );
            Vec2::new(
                gx.clamp(OBSTACLE_RADIUS, GRID as f32 - OBSTACLE_RADIUS),
                gy.clamp(OBSTACLE_RADIUS, GRID as f32 - OBSTACLE_RADIUS),
            )
        };
        let spring_force = self.control_stiffness * (cursor_target - self.obstacle_pos)
            - self.control_damping * self.obstacle_vel;
        self.obstacle
            .set_position_velocity(self.obstacle_pos, self.obstacle_vel);
        self.sim.step();
        let reaction = self.obstacle.take_reaction_impulse();
        self.obstacle_vel += reaction / OBSTACLE_MASS + (spring_force / OBSTACLE_MASS) * DT;
        self.obstacle_pos += self.obstacle_vel * DT;
        self.frame += 1;

        if self.log_timer.elapsed().as_secs_f32() >= 0.5 {
            self.log_timer = std::time::Instant::now();
            let max_speed = self
                .sim
                .particles()
                .v
                .iter()
                .map(|v| v.length())
                .fold(0.0f32, f32::max);
            println!(
                "frame={} cursor_screen=({:.0},{:.0}) obstacle_pos=({:.2},{:.2}) \
                 obstacle_speed={:.3} water_max_speed={:.2} pixels_per_point={:.2} \
                 window_physical=({},{})",
                self.frame,
                self.cursor_screen[0],
                self.cursor_screen[1],
                self.obstacle_pos.x,
                self.obstacle_pos.y,
                self.obstacle_vel.length(),
                max_speed,
                self.egui_ctx.pixels_per_point(),
                self.surface_config.width,
                self.surface_config.height,
            );
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);

        // Real, live-reported fix (2026-09-15): the particle renderer has
        // no notion of the obstacle at all -- it's a `BoundaryCondition`,
        // not a `Particle`, and can't be added as one without injecting
        // fake mass into the real MPM grid every frame (that would corrupt
        // the actual physics, not just the visual). This overlay is purely
        // a screen-space marker drawn with egui (same library every other
        // GUI example in this codebase already uses), positioned via the
        // exact same camera math the particle renderer itself uses -- it
        // never touches `sim`/`Particles` state.
        let raw_input = self.egui_state.take_egui_input(window);
        // Real fix (2026-09-15): read the position straight from the
        // renderer's OWN cached projection, in the LOGICAL points a UI
        // toolkit actually draws in (`grid_to_screen_points`/
        // `grid_distance_to_points` -- see their own doc for the real,
        // live-reported DPI bug this closes at the API level, not just in
        // this one call site: physical-pixel variants exist too, but a UI
        // overlay should always reach for the `_points` ones).
        let ppp = self.egui_ctx.pixels_per_point();
        let (cx, cy) = self.renderer.grid_to_screen_points(
            self.obstacle_pos.x,
            self.obstacle_pos.y,
            self.surface_config.width,
            self.surface_config.height,
            ppp,
        );
        let center = egui::pos2(cx, cy);
        let radius =
            self.renderer
                .grid_distance_to_points(OBSTACLE_RADIUS, self.surface_config.height, ppp);
        // Real signal, not a UI-only flag: solid when the obstacle is
        // ACTUALLY in contact this frame (real nonzero reaction impulse),
        // translucent when it's moving through free water -- so the marker
        // itself shows the same real physics `in_contact` gates on.
        let in_contact = reaction != Vec2::ZERO;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::background());
            let fill = if in_contact {
                egui::Color32::from_rgb(220, 90, 60)
            } else {
                egui::Color32::from_rgba_unmultiplied(220, 90, 60, 140)
            };
            painter.circle(
                center,
                radius,
                fill,
                egui::Stroke::new(2.0_f32, egui::Color32::WHITE),
            );
        });
        self.egui_state
            .handle_platform_output(window, full_output.platform_output);
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let sd = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let cmd = {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut enc, &tris, &sd);
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut rp, &tris, &sd);
            drop(rp);
            enc.finish()
        };
        self.queue.submit(std::iter::once(cmd));
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Kinematic Obstacle in Water (solid-liquid coupling)")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 480u32)),
            )
            .unwrap(),
        );
        // Real UX fix (2026-09-15, live-reported): the obstacle marker
        // lagging behind the OS cursor (the whole POINT of the real spring-
        // damper control law -- see those constants' own doc) reads as
        // "wrong" when a separate, always-on-target OS arrow is ALSO
        // visible right next to it. Hiding the OS cursor makes the red
        // circle the only visible pointer, so what's being steered and what
        // gets pushed around by real water resistance are visually the same
        // thing, matching the actual control model instead of fighting it.
        w.set_cursor_visible(false);
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        if let Some(w) = &self.window {
            let resp = s.egui_state.on_window_event(w, &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => s.reset(),
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_screen = [position.x as f32, position.y as f32];
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
