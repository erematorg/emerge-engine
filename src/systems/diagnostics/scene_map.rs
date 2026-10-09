//! A text picture of a scene, so a log can show where things are, not only
//! how much of them there is.
//!
//! Per-frame statistics say "40 percent of this body is at yield"; the
//! screen says which 40 percent. [`scene_map`] cuts a rectangle of the grid
//! into character cells and draws each by the mean of a per-particle value
//! over the particles inside it, through bands chosen to match what the
//! renderer shows. [`crate::FrameLogger::log_map`] writes it into the same
//! NDJSON file as the frame statistics, so one run gives both.

use glam::Vec2;

use crate::particle::Particles;

/// The renderer's `heat` colour map as characters. A colour is named by its
/// strongest channel, and as a mix of two when the second reaches 0.6 of the
/// strongest (a declared reading of "looks like both"). Through `heat`'s own
/// channel formulas that gives blue below 0.197 (`.`), teal to 0.299 (`:`),
/// green to 0.619 (`-`), yellow to orange to 0.716 (`+`), red above (`#`);
/// the bands below round those, and a test in the renderer sweeps `heat`
/// against them, so the map cannot drift from the screen unnoticed. The
/// first band catches everything below the second.
pub const HEAT_BANDS: [(f32, char); 5] = [
    (f32::NEG_INFINITY, '.'),
    (0.2, ':'),
    (0.3, '-'),
    (0.62, '+'),
    (0.72, '#'),
];

/// Where there is material and where there is not, whatever its value.
pub const OCCUPANCY_BANDS: [(f32, char); 1] = [(f32::NEG_INFINITY, 'o')];

/// A text picture of the grid rectangle from `region.0` to `region.1`, in
/// cells: `rows` strings of `cols` characters, top row first, as the screen
/// shows it. A character cell holds the mean of `value` over the particles
/// whose position falls inside it, drawn with the last band whose threshold
/// that mean reaches, or a space where no particle is.
///
/// `value` receives a particle's index into `particles`.
pub fn scene_map(
    particles: &Particles,
    region: (Vec2, Vec2),
    cols: usize,
    rows: usize,
    value: impl Fn(usize) -> f32,
    bands: &[(f32, char)],
) -> Vec<String> {
    let (min, max) = region;
    let size = (max - min).max(Vec2::splat(f32::MIN_POSITIVE));
    let (cols, rows) = (cols.max(1), rows.max(1));
    let mut sum = vec![0.0f32; cols * rows];
    let mut count = vec![0u32; cols * rows];
    for i in 0..particles.len() {
        let t = (particles.x[i] - min) / size;
        if !(0.0..1.0).contains(&t.x) || !(0.0..1.0).contains(&t.y) {
            continue;
        }
        let col = (t.x * cols as f32) as usize;
        // Row 0 is the top of the picture, the largest y.
        let row = rows - 1 - ((t.y * rows as f32) as usize).min(rows - 1);
        let cell = row * cols + col.min(cols - 1);
        sum[cell] += value(i);
        count[cell] += 1;
    }
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    let cell = row * cols + col;
                    if count[cell] == 0 {
                        return ' ';
                    }
                    let mean = sum[cell] / count[cell] as f32;
                    bands
                        .iter()
                        .rev()
                        .find(|(threshold, _)| mean >= *threshold)
                        .map_or('?', |(_, c)| *c)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;

    fn particle_at(x: f32, y: f32) -> Particle {
        let mut p = Particle::zeroed();
        p.x = Vec2::new(x, y);
        p
    }

    /// Two particles in opposite corners of a 4 x 2 picture of an 8 x 4
    /// region: the one near the origin lands bottom left, the other top
    /// right, each drawn by its own value's band; the rest stays blank.
    #[test]
    fn a_particle_lands_in_its_cell_top_row_first() {
        let particles = Particles::from(vec![particle_at(0.5, 0.5), particle_at(7.5, 3.5)]);
        let values = [0.1f32, 0.9];
        let map = scene_map(
            &particles,
            (Vec2::ZERO, Vec2::new(8.0, 4.0)),
            4,
            2,
            |i| values[i],
            &HEAT_BANDS,
        );
        assert_eq!(map, vec!["   #".to_string(), ".   ".to_string()]);
    }

    /// A cell is drawn by the mean of its particles, and anything outside
    /// the region is left out.
    #[test]
    fn a_cell_takes_the_mean_and_the_outside_is_ignored() {
        let particles = Particles::from(vec![
            particle_at(0.2, 0.2),
            particle_at(0.8, 0.8),
            particle_at(-1.0, 0.5),
        ]);
        let values = [0.2f32, 0.3, 5.0];
        let map = scene_map(
            &particles,
            (Vec2::ZERO, Vec2::ONE),
            1,
            1,
            |i| values[i],
            &HEAT_BANDS,
        );
        assert_eq!(map, vec![":".to_string()]);
    }
}
