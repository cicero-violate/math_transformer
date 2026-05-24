#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.re * rhs.re - self.im * rhs.im, self.re * rhs.im + self.im * rhs.re)
    }
}

pub fn fft(values: &mut [Complex], invert: bool) {
    let n = values.len();
    assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = 2.0 * std::f64::consts::PI / len as f64 * if invert { -1.0 } else { 1.0 };
        let wlen = Complex::new(angle.cos(), angle.sin());
        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = values[i + k];
                let v = values[i + k + len / 2] * w;
                values[i + k] = u + v;
                values[i + k + len / 2] = u - v;
                w = w * wlen;
            }
        }
        len <<= 1;
    }
    if invert {
        for x in values {
            x.re /= n as f64;
            x.im /= n as f64;
        }
    }
}

pub fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let inner = b.len();
    let cols = b.first().map_or(0, Vec::len);
    let mut c = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for k in 0..inner {
            for j in 0..cols {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

pub fn gaussian_elimination(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let div = a[col][col];
        for j in col..n {
            a[col][j] /= div;
        }
        b[col] /= div;
        for i in 0..n {
            if i == col {
                continue;
            }
            let factor = a[i][col];
            for j in col..n {
                a[i][j] -= factor * a[col][j];
            }
            b[i] -= factor * b[col];
        }
    }
    Some(b)
}

pub fn simplex_two_variable(c: [f64; 2], constraints: &[([f64; 2], f64)]) -> Option<([f64; 2], f64)> {
    let mut points = vec![[0.0, 0.0]];
    for &([a1, a2], rhs) in constraints {
        if a1.abs() > 1e-12 {
            points.push([rhs / a1, 0.0]);
        }
        if a2.abs() > 1e-12 {
            points.push([0.0, rhs / a2]);
        }
    }
    for i in 0..constraints.len() {
        for j in i + 1..constraints.len() {
            let ([a, b], r1) = constraints[i];
            let ([c1, d], r2) = constraints[j];
            let det = a * d - b * c1;
            if det.abs() > 1e-12 {
                points.push([(r1 * d - b * r2) / det, (a * r2 - r1 * c1) / det]);
            }
        }
    }
    points.into_iter()
        .filter(|p| p[0] >= -1e-9 && p[1] >= -1e-9)
        .filter(|p| constraints.iter().all(|&([a, b], rhs)| a * p[0] + b * p[1] <= rhs + 1e-9))
        .map(|p| (p, c[0] * p[0] + c[1] * p[1]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_kernels_work() {
        assert_eq!(matrix_multiply(&[vec![1.0, 2.0]], &[vec![3.0], vec![4.0]]), vec![vec![11.0]]);
        let x = gaussian_elimination(vec![vec![2.0, 1.0], vec![1.0, -1.0]], vec![5.0, 1.0]).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-9);
        assert!((x[1] - 1.0).abs() < 1e-9);
        let (_, value) = simplex_two_variable([3.0, 2.0], &[([1.0, 1.0], 4.0), ([1.0, 0.0], 2.0)]).unwrap();
        assert!((value - 10.0).abs() < 1e-9);
        let mut xs = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        fft(&mut xs, false);
        assert!(xs.iter().all(|x| (x.re - 1.0).abs() < 1e-9));
    }
}
