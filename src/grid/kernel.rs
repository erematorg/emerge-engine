use glam::{IVec2, Vec2};

#[derive(Clone, Copy, Debug)]
pub struct QuadraticWeights {
    pub base_cell: IVec2,
    pub wx: [f32; 3],
    pub wy: [f32; 3],
}

pub fn quadratic_weights(position: Vec2) -> QuadraticWeights {
    let base_cell = position.floor().as_ivec2();
    let diff = position - base_cell.as_vec2() - Vec2::splat(0.5);
    QuadraticWeights {
        base_cell,
        wx: axis_weights(diff.x),
        wy: axis_weights(diff.y),
    }
}

pub fn axis_weights(d: f32) -> [f32; 3] {
    let w0 = 0.5 * (0.5 - d).powi(2);
    let w1 = 0.75 - d.powi(2);
    let w2 = 0.5 * (0.5 + d).powi(2);
    [w0, w1, w2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_weights_sum_to_one() {
        for sample in [0.0f32, 0.1, 0.35, -0.2] {
            let ws = axis_weights(sample);
            let sum = ws[0] + ws[1] + ws[2];
            assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        }
    }
}
