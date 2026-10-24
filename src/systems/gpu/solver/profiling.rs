use super::{GpuProfiling, GpuSimulation, PROFILE_PASS_LABELS};

/// First of the two frame-wide timestamp slots, right after the per-pass pairs.
const PROFILE_FRAME_SLOT: u32 = PROFILE_PASS_LABELS.len() as u32 * 2;

impl GpuSimulation {
    /// Turns on per-stage GPU timing for `encode_substep`'s labeled stages. Returns false
    /// (no-op) if this device wasn't created with `TIMESTAMP_QUERY` and
    /// `TIMESTAMP_QUERY_INSIDE_PASSES` (all stages share one compute pass) -- `new()`
    /// requests it opportunistically when the adapter supports it; `with_device()` depends
    /// on whatever device the caller already built. Call once after construction; read
    /// results back with `last_pass_timings_ns()` after stepping a few frames.
    pub fn enable_profiling(&mut self) -> bool {
        if !self.device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        ) {
            return false;
        }
        let n = PROFILE_PASS_LABELS.len() as u32;
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("emerge_profile_queries"),
            ty: wgpu::QueryType::Timestamp,
            // begin+end per pass, plus one frame-wide begin/end pair
            // (`PROFILE_FRAME_SLOT`) spanning every substep of the frame.
            count: n * 2 + 2,
        });
        let resolve_size = (n * 2 + 2) as u64 * 8; // 8 bytes per u64 timestamp
        let resolve_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emerge_profile_resolve"),
            size: resolve_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emerge_profile_readback"),
            size: resolve_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.profiling = Some(GpuProfiling {
            query_set,
            resolve_buf,
            readback_buf,
            timestamp_period_ns: self.queue.get_timestamp_period(),
        });
        true
    }

    /// Reads back the last substep's per-pass GPU timings (label, nanoseconds), in
    /// `encode_substep`'s pass order. Blocks until the GPU work + readback completes -- a
    /// diagnostic call, not for the hot path. Returns None if `enable_profiling()` wasn't
    /// called or wasn't supported on this device.
    pub fn last_pass_timings_ns(&mut self) -> Option<Vec<(&'static str, f32)>> {
        Some(
            self.last_pass_timeline_ns()?
                .into_iter()
                .map(|(label, begin, end)| (label, end - begin))
                .collect(),
        )
    }

    /// Like `last_pass_timings_ns`, but keeps each pass's begin/end offsets (ns) relative
    /// to the earliest recorded begin, so the idle gaps BETWEEN passes are visible too.
    /// Passes that did not run this substep report (label, 0, 0).
    pub fn last_pass_timeline_ns(&mut self) -> Option<Vec<(&'static str, f32, f32)>> {
        let profiling = self.profiling.as_ref()?;
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let slice = profiling.readback_buf.slice(..);
        let flag = std::sync::Arc::new(std::sync::Mutex::new(None));
        let flag2 = flag.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            *flag2.lock().unwrap() = Some(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        flag.lock().unwrap().take()?.ok()?;
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        let period = profiling.timestamp_period_ns;
        let origin = (0..PROFILE_PASS_LABELS.len())
            .filter(|&i| timestamps[i * 2 + 1] > timestamps[i * 2])
            .map(|i| timestamps[i * 2])
            .min()
            .unwrap_or(0);
        let result = PROFILE_PASS_LABELS
            .iter()
            .enumerate()
            .map(|(i, &label)| {
                let begin = timestamps[i * 2];
                let end = timestamps[i * 2 + 1];
                if end <= begin {
                    return (label, 0.0, 0.0);
                }
                (
                    label,
                    begin.saturating_sub(origin) as f32 * period,
                    end.saturating_sub(origin) as f32 * period,
                )
            })
            .collect();
        drop(data);
        profiling.readback_buf.unmap();
        Some(result)
    }

    /// GPU wall time (ns) from the first substep's first pass to the last substep's last
    /// pass of the most recent `step_frame()` -- includes everything between substeps,
    /// which the per-pass timeline cannot see. Same blocking readback as
    /// `last_pass_timeline_ns`.
    pub fn last_frame_gpu_span_ns(&mut self) -> Option<f32> {
        let profiling = self.profiling.as_ref()?;
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let slice = profiling.readback_buf.slice(..);
        let flag = std::sync::Arc::new(std::sync::Mutex::new(None));
        let flag2 = flag.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            *flag2.lock().unwrap() = Some(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        flag.lock().unwrap().take()?.ok()?;
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        let slot = PROFILE_FRAME_SLOT as usize;
        let span = timestamps[slot + 1].saturating_sub(timestamps[slot]) as f32
            * profiling.timestamp_period_ns;
        drop(data);
        profiling.readback_buf.unmap();
        Some(span)
    }

    /// Records the frame-wide begin (`end == false`) or end (`end == true`) timestamp
    /// as an empty compute pass, and on `end` resolves every query into the readback
    /// buffer. No-op unless profiling is enabled.
    pub(super) fn profile_frame_marker(&self, encoder: &mut wgpu::CommandEncoder, end: bool) {
        let Some(p) = &self.profiling else {
            return;
        };
        let (begin_idx, end_idx) = if end {
            (None, Some(PROFILE_FRAME_SLOT + 1))
        } else {
            (Some(PROFILE_FRAME_SLOT), None)
        };
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(if end {
                "profile_frame_end"
            } else {
                "profile_frame_begin"
            }),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &p.query_set,
                beginning_of_pass_write_index: begin_idx,
                end_of_pass_write_index: end_idx,
            }),
        }));
        if end {
            let count = PROFILE_FRAME_SLOT + 2;
            encoder.resolve_query_set(&p.query_set, 0..count, &p.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&p.resolve_buf, 0, &p.readback_buf, 0, count as u64 * 8);
        }
    }

    /// Writes the begin (`end == false`) or end timestamp of labeled stage `i` (in
    /// `PROFILE_PASS_LABELS` order) inside the shared substep compute pass -- only for the
    /// frame's last substep (`profile_this_substep`), and only when profiling is enabled.
    pub(super) fn profile_stamp(&self, pass: &mut wgpu::ComputePass<'_>, i: u32, end: bool) {
        if !self.profile_this_substep.get() {
            return;
        }
        if let Some(p) = &self.profiling {
            pass.write_timestamp(&p.query_set, i * 2 + u32::from(end));
        }
    }
}
