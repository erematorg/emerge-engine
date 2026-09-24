//! Real, general, opt-in window+event-loop runner for a winit/wgpu demo --
//! extracted from the identical ~80-line boilerplate every example in this
//! repo was hand-duplicating (window creation, adapter/device/surface setup,
//! the `ApplicationHandler` impl, resize handling, the `WindowEvent` match).
//! Real engine capability (behind `feature = "render"`, same gate `Renderer`
//! itself uses), not example-only tooling -- any consumer of this crate gets
//! it, not just this repo's own `examples/`.
//!
//! Deliberately NOT the only way to use [`super::Renderer`]: a consumer with
//! their own event loop (LP's own Bevy-based one, for instance) uses
//! `Renderer` directly and never touches this module -- purely additive,
//! opt-in convenience for quick visual demos/prototyping, not a forced
//! architecture. `winit`/`pollster` are real (non-dev) but `optional`
//! dependencies, gated the same way `wgpu` already is -- zero cost for any
//! consumer that never enables `render`.
//!
//! # Example
//! ```rust,no_run
//! # extern crate emerge_engine as emerge;
//! use emerge::render::demo_harness::{DemoApp, run_demo};
//!
//! struct MyDemo;
//! impl DemoApp for MyDemo {
//!     const TITLE: &'static str = "my demo";
//!     fn new(_device: &wgpu::Device, _queue: &wgpu::Queue, _format: wgpu::TextureFormat, _width: u32, _height: u32) -> Self {
//!         MyDemo
//!     }
//!     fn resize(&mut self, _queue: &wgpu::Queue, _width: u32, _height: u32) {}
//!     fn update_and_render(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue, _view: &wgpu::TextureView) {}
//! }
//!
//! fn main() {
//!     run_demo::<MyDemo>();
//! }
//! ```

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// What a demo provides; the harness owns everything else (window, adapter,
/// device, surface, event loop, resize, Escape/Q-to-quit). Real, minimal
/// surface -- three required methods, matching what every existing demo's
/// own `State`/`App` pair already implemented by hand.
pub trait DemoApp: 'static {
    /// Window title.
    const TITLE: &'static str;
    /// Initial window size, logical pixels. Most existing demos use 480x480
    /// -- kept as the default so a minimal `impl` doesn't need to restate it.
    const SIZE: (u32, u32) = (480, 480);

    /// Build the demo's own state (simulation, renderer, whatever else it
    /// needs) once the GPU device/queue/surface format are ready. `width`/
    /// `height` are the REAL initial physical-pixel surface size (may
    /// differ from `SIZE` under DPI scaling, same value `resize` would
    /// receive for this same window state) -- most demos need this
    /// immediately for their own initial `Renderer::set_camera` call.
    /// Mirrors every existing demo's own `State::new` body, minus the wgpu
    /// setup that body always duplicated.
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self;

    /// Called on every real `WindowEvent::Resized` (already filtered for
    /// zero width/height, matching every existing demo's own guard).
    fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32);

    /// Called once per real frame (`WindowEvent::RedrawRequested`, after the
    /// harness has already acquired the surface texture and view) --
    /// step the simulation and draw into `view`. The harness presents the
    /// frame immediately after this returns; do not call `present()` here.
    fn update_and_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    );

    /// Called for every real key press OTHER than Escape/Q (which the
    /// harness itself always treats as quit, matching every existing demo's
    /// own convention). Default: no-op -- override for demo-specific keys
    /// (R to reset, a slider toggle, etc.).
    fn key_pressed(&mut self, _key: KeyCode) {}

    /// Called for every real key release. Default: no-op -- override for
    /// hold-to-drive input (arrow keys held down to steer a body), which
    /// needs both edges, not just the press.
    fn key_released(&mut self, _key: KeyCode) {}

    /// Called on every real cursor move, as a FRACTION of the current
    /// window size (`x`/`y` in `[0,1]`, `y` NOT flipped -- `0` is the top of
    /// the window, matching winit's own `CursorMoved` convention directly).
    /// A demo that needs grid coordinates does its own
    /// `x * GRID as f32`/`(1.0 - y) * GRID as f32` (whichever convention it
    /// needs) -- the harness doesn't know the demo's own grid resolution or
    /// Y convention, so it hands back the one thing it DOES know (window
    /// size) already applied, not a half-converted value. Default: no-op.
    fn cursor_moved(&mut self, _x_frac: f32, _y_frac: f32) {}

    /// Called on every real mouse button press/release. Default: no-op --
    /// override for click-to-drop/push-pull style interactions.
    fn mouse_button(&mut self, _button: MouseButton, _pressed: bool) {}
}

struct HarnessState<A: DemoApp> {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    app: A,
}

impl<A: DemoApp> HarnessState<A> {
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
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);
        let app = A::new(&device, &queue, format, size.width, size.height);
        Self {
            surface,
            surface_config,
            device,
            queue,
            app,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.app.resize(&self.queue, width, height);
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.app.update_and_render(&self.device, &self.queue, &view);
        output.present();
    }
}

struct Harness<A: DemoApp> {
    window: Option<Arc<Window>>,
    state: Option<HarnessState<A>>,
}

impl<A: DemoApp> ApplicationHandler for Harness<A> {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let window = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title(A::TITLE)
                    .with_inner_size(winit::dpi::LogicalSize::new(A::SIZE.0, A::SIZE.1)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(HarnessState::new(window.clone())));
        self.window = Some(window);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                let w = state.surface_config.width.max(1) as f32;
                let h = state.surface_config.height.max(1) as f32;
                state
                    .app
                    .cursor_moved(position.x as f32 / w, position.y as f32 / h);
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                state
                    .app
                    .mouse_button(button, btn_state == ElementState::Pressed);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: key_state,
                        ..
                    },
                ..
            } => match (key, key_state) {
                (KeyCode::Escape | KeyCode::KeyQ, ElementState::Pressed) => el.exit(),
                (other, ElementState::Pressed) => state.app.key_pressed(other),
                (other, ElementState::Released) => state.app.key_released(other),
            },
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Runs a [`DemoApp`] with a real winit+wgpu window -- handles adapter/
/// device/surface setup, the event loop, resize, and Escape/Q-to-quit, all
/// of which every existing demo in this repo was hand-duplicating. Blocks
/// until the window closes.
pub fn run_demo<A: DemoApp>() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut harness = Harness::<A> {
        window: None,
        state: None,
    };
    event_loop.run_app(&mut harness).unwrap();
}
