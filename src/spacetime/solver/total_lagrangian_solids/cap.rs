//! A wall-clock cap for the probes: a run must end and say
//! why. The body runs on its own thread; past the cap the process prints
//! "stopped at cap" with the run's name and exits with status 2, ending
//! every thread (and every other test of the binary). The cap is a
//! parameter: each run's own, or `EMERGE_PROBE_CAP_S` seconds when set.

use std::sync::mpsc;
use std::time::Duration;

/// Runs `body`, ending the process with an explicit "stopped at cap" line
/// when it takes longer than `seconds` (or `EMERGE_PROBE_CAP_S`).
pub(crate) fn capped(name: &str, seconds: u64, body: impl FnOnce() + Send + 'static) {
    let limit = std::env::var("EMERGE_PROBE_CAP_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(seconds);
    let (done, finished) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        body();
        let _ = done.send(());
    });
    match finished.recv_timeout(Duration::from_secs(limit)) {
        Ok(()) => worker.join().expect("the run's thread"),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            println!("{name}: stopped at cap ({limit} s)");
            std::process::exit(2);
        }
        // The body panicked: join to pass its panic on.
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            if let Err(panic) = worker.join() {
                std::panic::resume_unwind(panic);
            }
        }
    }
}
