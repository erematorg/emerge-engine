use super::GpuSimulation;

impl GpuSimulation {
    /// Why this instance's device was lost, if it was; `None` in ordinary
    /// operation. Wired automatically for `new()` instances; `with_device()`
    /// instances need one call to `enable_device_lost_detection()` first (see
    /// that method for why). Once set, `step_frame` and the blocking sync
    /// methods are no-ops instead of panicking on a dead device; poll this
    /// rather than assume silence means healthy.
    pub fn device_lost_reason(&self) -> Option<String> {
        self.device_lost.lock().ok().and_then(|g| g.clone())
    }

    /// Opt in to device-lost detection (issue #10's cause: an `Out of Memory`
    /// device loss under sustained load on slow or software GPU backends).
    /// Called automatically by `new()`, which owns its device. Not automatic
    /// for `with_device()` (a shared device, e.g. with a renderer): a wgpu
    /// device holds one lost-callback and one uncaptured-error handler
    /// (single-slot storage, wgpu-27.0.1's `ErrorSinkRaw`), so registering
    /// here could silently replace the caller's. Call this after
    /// `with_device()` if you (like LP) have no device-lost handling of your
    /// own; if you have registered your own callback or handler, don't, since
    /// the second registration replaces the first (wgpu's behavior).
    ///
    /// Also installs an uncaptured-error handler. wgpu's default for any
    /// uncaptured error is an unconditional panic (`panic!("wgpu error:
    /// {err}")`, wgpu-27.0.1's `default_error_handler`); this handler replaces
    /// it and never panics, whatever the error.
    ///
    /// "Never" matters: a handler that treated errors naming a destroyed or
    /// lost resource as device loss but still panicked on anything else
    /// crashed with `STATUS_STACK_BUFFER_OVERRUN` on the D3D12 WARP backend
    /// (the one windows-latest CI uses), because unwinding a panic from inside
    /// `wgpu_core::Queue::submit`'s error path is unsafe there. Do not add a
    /// "still panic for real bugs" branch. Every uncaptured error sets
    /// `device_lost` (so `is_device_lost()`'s no-op guards take over) and is
    /// printed in full with `eprintln!`.
    pub fn enable_device_lost_detection(&self) {
        let flag = self.device_lost.clone();
        self.device
            .set_device_lost_callback(move |reason, message| {
                *flag.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(format!("{reason:?}: {message}"));
            });

        let flag = self.device_lost.clone();
        self.device
            .on_uncaptured_error(std::sync::Arc::new(move |error: wgpu::Error| {
                let message = error.to_string();
                let mut guard = flag.lock().unwrap_or_else(|e| e.into_inner());
                if guard.is_none() {
                    *guard = Some(format!("(uncaptured wgpu error) {message}"));
                }
                drop(guard);
                eprintln!(
                    "emerge: uncaptured wgpu error, treating device as unusable from \
                     here (see GpuSimulation::enable_device_lost_detection's doc for \
                     why this never panics): {message}"
                );
            }));
    }

    pub(super) fn is_device_lost(&self) -> bool {
        self.device_lost
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }
}
