extern crate emerge_engine as emerge;

#[path = "../gui_common/render_mode.rs"]
mod render_mode;

use egui_wgpu::ScreenDescriptor;
use emerge::materials::optical::pure_water;
use emerge::render::{ColorMode, CpuRenderBridge, Renderer};
use emerge::{
    KinematicCircleBoundary, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};
/// Visible, INTERACTIVE solid-liquid two-way coupling: the mouse
/// cursor drives a `KinematicCircleBoundary` (a kinematically-driven
/// obstacle, NOT a rigid body -- see that type's doc for why this
/// engine's "no rigid bodies" scope rule doesn't apply to it) through
/// settled water via a virtual spring-damper control law (see the
/// `CONTROL_*` constants' doc for the derivation -- impedance
/// control, Hogan 1985, not a teleport-to-cursor hack). The obstacle's
/// velocity every step sums TWO real forces: that spring pulling it toward
/// the cursor, and the mass-weighted reaction impulse the water
/// exerts back on it (`take_reaction_impulse()`, Newton's third law) -- so
/// if this coupling is genuine, heavy water resistance visibly displaces
/// the obstacle away from wherever the cursor is trying to drag it ("gets
/// carried by the forces"), and moving the cursor harder measurably fights
/// that resistance back, all from the same one force balance, nothing
/// scripted per-behavior.
///
/// The underlying reaction-impulse mechanism is the exact same real physics
/// already verified headless in
/// `tests/probes/kinematic_reactive_obstacle_water_verify.rs` (mass
/// conserved, obstacle decelerates on contact, reaction hits zero
/// out of contact) -- this demo drives that same mechanism from live cursor
/// input instead of a scripted initial velocity.
///
/// G cycles the view: particles, the grid-volume view, the curvature-flow
/// surface. The water carries pure water's measured absorption
/// (`materials::optical::pure_water`). The obstacle is a boundary, not
/// matter, so every view shows it only through the marker drawn over it.
///
///   cargo run --example kinematic_obstacle_in_water --features render
use render_mode::RenderMode;
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
/// Particle pitch of the pool, in grid cells.
const SPACING: f32 = 0.9;
/// Surface grid resolution as a multiple of the physics grid.
const SURFACE_RES_MULTIPLIER: u32 = 4;
// Control law: the cursor sets a target position, and the obstacle is pulled toward it
// by a virtual spring-damper (impedance control, Hogan 1985, the principle haptic
// interfaces use to let a human feel resistance through a driven object). The spring
// force and the water's reaction impulse are summed into one velocity update, so heavy
// resistance displaces the obstacle away from the cursor and holding the cursor
// against it fights back; neither behavior is scripted, both come from one force
// balance.
//
// Gains from a target settling time of `CONTROL_RESPONSE_PERIODS_OF_DT` physics steps:
// natural frequency omega=2*pi/(periods*DT), k=m*omega^2 (harmonic oscillator) and
// damping=2*sqrt(k*m)*ratio (critical damping, as `grain_contact_config` uses). The
// "gain" is how many physics steps a deliberate cursor move takes to catch up, an
// interface-responsiveness choice, not a material constant.
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
    // Spring and damping gains, derived once from `CONTROL_RESPONSE_PERIODS_OF_DT`/
    // `CONTROL_DAMPING_RATIO` (see those constants); nothing they depend on changes.
    control_stiffness: f32,
    control_damping: f32,
    frame: u64,
    log_timer: std::time::Instant,
    // Camera extent derived from the scene (see `camera_extent_for_aspect`), computed
    // once at startup from the settled-water bounding box and the obstacle's travel
    // room, with the window's initial aspect, then used as a fixed `grid_res` on every
    // resize like the other examples: `set_camera` already adapts the horizontal extent
    // to the aspect (its `sx`/`sy` split), and recomputing the extent per resize also
    // changes the vertical framing.
    camera_extent: f32,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    /// Which render path draws the frame, cycled with G.
    render_mode: RenderMode,
    /// GPU buffers the grid-volume and surface modes read, rebuilt from the
    /// CPU solver on the frames those modes are shown.
    render_bridge: CpuRenderBridge,
}

/// `Renderer::set_camera`'s `grid_res` frames a square-ish region from world origin
/// (0,0), widened by the window's aspect ratio (see its derivation: landscape shows
/// `y=grid_res, x=grid_res*aspect`; portrait the mirror). The physics grid size (64
/// here) frames the whole domain, but this scene's water pool is a thin band near the
/// bottom, which would leave a mostly empty window over a sliver of water. This
/// computes the smallest `grid_res` that contains the scene (vertical_need: settled
/// water height + splash headroom; horizontal_reach: water extent + room for the
/// obstacle to travel into view) for the window's aspect ratio.
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
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let mut water = NewtonianFluidMaterial::low_viscosity(0.1, 2.5);
    water.optics = Some(pure_water());
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
        // A domain wall, as in the other fluid examples: without one the pool spreads
        // across nearly the whole grid while settling (bbox reaching x=62 of 64),
        // which also pushes the obstacle's start position, derived from that bbox,
        // off the grid.
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

        // Enough sky above the settled surface for a splash to stay visible, clamped
        // to the physics domain (`GRID`) on both axes: camera space beyond the
        // simulation would let the cursor map there and strand the obstacle with no
        // possible contact.
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
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.adopt_material_optics(&queue, sim.materials());
        // The grid-volume and surface modes threshold on cell mass as a
        // fraction of a full cell, which holds 1/SPACING^2 particles.
        renderer.set_grid_reference_cell_mass(sim.particles().mass[0] / (SPACING * SPACING));
        renderer.set_surface_res_multiplier(SURFACE_RES_MULTIPLIER);
        let render_bridge = CpuRenderBridge::new(&device, GRID);

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
        println!("G render mode  R reset  Q quit");

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
            render_mode: RenderMode::Particles,
            render_bridge,
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

    /// Two-way coupling: the obstacle's velocity update sums two forces every step --
    /// a virtual spring toward the cursor (see the control constants) and the reaction
    /// impulse the water exerts back on it (`take_reaction_impulse()`, Newton's third
    /// law). Being carried by the water and fighting back by moving the cursor harder
    /// are the same force balance playing out differently with how hard the water
    /// pushes.
    fn update_and_render(&mut self, window: &Window) {
        // Clamp the target to the physics domain: a raw `screen_to_grid` reading can
        // map outside [0, GRID] (window edges, or the camera's splash/travel headroom,
        // see `camera_extent_for_aspect`), and a cursor resting there gives the spring
        // an equilibrium off the grid with no possible reaction. Margin =
        // `OBSTACLE_RADIUS` so the obstacle's body, not just its center, stays inside.
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
        match self.render_mode {
            RenderMode::Particles => {
                self.renderer
                    .render(&self.device, &self.queue, self.sim.particles(), &view, true)
            }
            RenderMode::GridVolume => {
                self.render_bridge
                    .upload_grid(&self.queue, self.sim.particles(), self.sim.grid());
                self.renderer.render_grid_volume(
                    &self.device,
                    &self.queue,
                    self.render_bridge.grid_volume_source(),
                    &view,
                    true,
                );
            }
            RenderMode::Surface => {
                self.render_bridge.upload_particles(
                    &self.device,
                    &self.queue,
                    self.sim.particles(),
                );
                // One material, so one optics slot colours the whole surface.
                self.renderer.render_surface_reconstruction(
                    &self.device,
                    &self.queue,
                    self.render_bridge
                        .surface_source(MAT_WATER, false, self.sim.mean_substep_dt()),
                    &view,
                    true,
                );
            }
        }

        // The particle renderer has no notion of the obstacle: it is a
        // `BoundaryCondition`, not a `Particle`, and making it one would inject mass
        // into the MPM grid every frame. This overlay is a screen-space marker drawn
        // with egui, positioned with the particle renderer's camera math; it never
        // touches `sim`/`Particles` state.
        let raw_input = self.egui_state.take_egui_input(window);
        // Position from the renderer's cached projection, in logical points, the unit
        // a UI toolkit draws in (`grid_to_screen_points`/`grid_distance_to_points`;
        // see their doc for the DPI issue with the physical-pixel variants).
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
        // Solid when the obstacle is in contact this frame (nonzero reaction impulse),
        // translucent when it moves through free water: the marker shows what
        // `in_contact` gates on.
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
        // OS cursor hidden: the obstacle marker lags the cursor by design (the
        // spring-damper control law), and an always-on-target OS arrow next to it
        // reads as wrong. With the red circle the only pointer, what is steered and
        // what the water pushes around are the same thing on screen.
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
                KeyCode::KeyG => {
                    s.render_mode = s.render_mode.next();
                    println!("render mode: {}", s.render_mode.label());
                }
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
