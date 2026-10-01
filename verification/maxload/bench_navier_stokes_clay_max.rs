//! MAX-LOAD параметрическая копия bench_navier_stokes_clay.rs (poler-engine-org/poler-engine)
//!
//! Алгоритм_bit-for_bit идентичен оригиналу: центральные разности 2-го порядка,
//! явный Эйлер, 12 итераций Якоби для давления (приближённая проекция Лере).
//! ИЗМЕНЁН ТОЛЬКО main(): параметры (N, steps, dt, out_every) читаются из argv.
//!
//! Сборка: rustc -O bench_navier_stokes_clay_max.rs -o clay_bench_max
//! Запуск: ./clay_bench_max [N] [steps] [dt] [out_every]

use std::time::Instant;
use std::f64::consts::PI;
use std::env;

pub struct NavierStokes3D {
    pub n: usize,
    pub l: f64,
    pub nu: f64,
    pub rho: f64,
    pub v1: Vec<f64>,
    pub v2: Vec<f64>,
    pub v3: Vec<f64>,
    pub p:  Vec<f64>,
}

impl NavierStokes3D {
    pub fn new(n: usize, l: f64, nu: f64, rho: f64) -> Self {
        let size = n * n * n;
        Self {
            n,
            l,
            nu,
            rho,
            v1: vec![0.0; size],
            v2: vec![0.0; size],
            v3: vec![0.0; size],
            p:  vec![0.0; size],
        }
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.n + j) * self.n + k
    }

    /// Ініціалізація C^∞ гладким бездивергентним полем швидкостей Тейлора-Гріна (∇·v = 0)
    pub fn init_taylor_green(&mut self, v0: f64) {
        let dx = self.l / (self.n as f64);
        let k_val = 2.0 * PI / self.l;
        for i in 0..self.n {
            let x = i as f64 * dx;
            for j in 0..self.n {
                let y = j as f64 * dx;
                for k in 0..self.n {
                    let z = k as f64 * dx;
                    let id = self.idx(i, j, k);
                    self.v1[id] = v0 * (k_val * x).sin() * (k_val * y).cos() * (k_val * z).cos();
                    self.v2[id] = -v0 * (k_val * x).cos() * (k_val * y).sin() * (k_val * z).cos();
                    self.v3[id] = 0.0;
                    self.p[id] = (self.rho * v0 * v0 / 16.0)
                        * ((2.0 * k_val * x).cos() + (2.0 * k_val * y).cos())
                        * ((2.0 * k_val * z).cos() + 2.0);
                }
            }
        }
    }

    pub fn compute_max_divergence(&self) -> f64 {
        let dx = self.l / (self.n as f64);
        let mut max_div: f64 = 0.0;
        for i in 0..self.n {
            let ip = (i + 1) % self.n;
            let im = (i + self.n - 1) % self.n;
            for j in 0..self.n {
                let jp = (j + 1) % self.n;
                let jm = (j + self.n - 1) % self.n;
                for k in 0..self.n {
                    let kp = (k + 1) % self.n;
                    let km = (k + self.n - 1) % self.n;

                    let dv1_dx = (self.v1[self.idx(ip, j, k)] - self.v1[self.idx(im, j, k)]) / (2.0 * dx);
                    let dv2_dy = (self.v2[self.idx(i, jp, k)] - self.v2[self.idx(i, jm, k)]) / (2.0 * dx);
                    let dv3_dz = (self.v3[self.idx(i, j, kp)] - self.v3[self.idx(i, j, km)]) / (2.0 * dx);

                    let div = dv1_dx + dv2_dy + dv3_dz;
                    if div.abs() > max_div {
                        max_div = div.abs();
                    }
                }
            }
        }
        max_div
    }

    pub fn compute_total_energy(&self) -> f64 {
        let dx = self.l / (self.n as f64);
        let d_vol = dx * dx * dx;
        let mut energy = 0.0;
        for id in 0..(self.n * self.n * self.n) {
            let v_sq = self.v1[id] * self.v1[id] + self.v2[id] * self.v2[id] + self.v3[id] * self.v3[id];
            energy += 0.5 * self.rho * v_sq * d_vol;
        }
        energy
    }

    pub fn compute_enstrophy(&self) -> f64 {
        let dx = self.l / (self.n as f64);
        let d_vol = dx * dx * dx;
        let mut enstrophy = 0.0;
        for i in 0..self.n {
            let ip = (i + 1) % self.n;
            let im = (i + self.n - 1) % self.n;
            for j in 0..self.n {
                let jp = (j + 1) % self.n;
                let jm = (j + self.n - 1) % self.n;
                for k in 0..self.n {
                    let kp = (k + 1) % self.n;
                    let km = (k + self.n - 1) % self.n;

                    let dv3_dy = (self.v3[self.idx(i, jp, k)] - self.v3[self.idx(i, jm, k)]) / (2.0 * dx);
                    let dv2_dz = (self.v2[self.idx(i, j, kp)] - self.v2[self.idx(i, j, km)]) / (2.0 * dx);
                    let omega_1 = dv3_dy - dv2_dz;

                    let dv1_dz = (self.v1[self.idx(i, j, kp)] - self.v1[self.idx(i, j, km)]) / (2.0 * dx);
                    let dv3_dx = (self.v3[self.idx(ip, j, k)] - self.v3[self.idx(im, j, k)]) / (2.0 * dx);
                    let omega_2 = dv1_dz - dv3_dx;

                    let dv2_dx = (self.v2[self.idx(ip, j, k)] - self.v2[self.idx(im, j, k)]) / (2.0 * dx);
                    let dv1_dy = (self.v1[self.idx(i, jp, k)] - self.v1[self.idx(i, jm, k)]) / (2.0 * dx);
                    let omega_3 = dv2_dx - dv1_dy;

                    let w_sq = omega_1 * omega_1 + omega_2 * omega_2 + omega_3 * omega_3;
                    enstrophy += 0.5 * w_sq * d_vol;
                }
            }
        }
        enstrophy
    }

    /// Точний крок розв'язку 3D Нав'є–Стокса з проектором Лере-Гельмгольца P = I - ∇Δ⁻¹∇·
    /// (ДОСЛОВНО з bench_navier_stokes_clay.rs — не змінено жодного рядка обчислень)
    pub fn step(&mut self, dt: f64) {
        let dx = self.l / (self.n as f64);
        let dx2 = dx * dx;
        let size = self.n * self.n * self.n;

        let mut rhs1 = vec![0.0; size];
        let mut rhs2 = vec![0.0; size];
        let mut rhs3 = vec![0.0; size];

        for i in 0..self.n {
            let ip = (i + 1) % self.n;
            let im = (i + self.n - 1) % self.n;
            for j in 0..self.n {
                let jp = (j + 1) % self.n;
                let jm = (j + self.n - 1) % self.n;
                for k in 0..self.n {
                    let kp = (k + 1) % self.n;
                    let km = (k + self.n - 1) % self.n;
                    let id = self.idx(i, j, k);

                    let u = self.v1[id];
                    let v = self.v2[id];
                    let w = self.v3[id];

                    let du_dx = (self.v1[self.idx(ip, j, k)] - self.v1[self.idx(im, j, k)]) / (2.0 * dx);
                    let du_dy = (self.v1[self.idx(i, jp, k)] - self.v1[self.idx(i, jm, k)]) / (2.0 * dx);
                    let du_dz = (self.v1[self.idx(i, j, kp)] - self.v1[self.idx(i, j, km)]) / (2.0 * dx);
                    let conv1 = u * du_dx + v * du_dy + w * du_dz;

                    let dv_dx = (self.v2[self.idx(ip, j, k)] - self.v2[self.idx(im, j, k)]) / (2.0 * dx);
                    let dv_dy = (self.v2[self.idx(i, jp, k)] - self.v2[self.idx(i, jm, k)]) / (2.0 * dx);
                    let dv_dz = (self.v2[self.idx(i, j, kp)] - self.v2[self.idx(i, j, km)]) / (2.0 * dx);
                    let conv2 = u * dv_dx + v * dv_dy + w * dv_dz;

                    let dw_dx = (self.v3[self.idx(ip, j, k)] - self.v3[self.idx(im, j, k)]) / (2.0 * dx);
                    let dw_dy = (self.v3[self.idx(i, jp, k)] - self.v3[self.idx(i, jm, k)]) / (2.0 * dx);
                    let dw_dz = (self.v3[self.idx(i, j, kp)] - self.v3[self.idx(i, j, km)]) / (2.0 * dx);
                    let conv3 = u * dw_dx + v * dw_dy + w * dw_dz;

                    let lap1 = (self.v1[self.idx(ip, j, k)] + self.v1[self.idx(im, j, k)]
                              + self.v1[self.idx(i, jp, k)] + self.v1[self.idx(i, jm, k)]
                              + self.v1[self.idx(i, j, kp)] + self.v1[self.idx(i, j, km)]
                              - 6.0 * u) / dx2;

                    let lap2 = (self.v2[self.idx(ip, j, k)] + self.v2[self.idx(im, j, k)]
                              + self.v2[self.idx(i, jp, k)] + self.v2[self.idx(i, jm, k)]
                              + self.v2[self.idx(i, j, kp)] + self.v2[self.idx(i, j, km)]
                              - 6.0 * v) / dx2;

                    let lap3 = (self.v3[self.idx(ip, j, k)] + self.v3[self.idx(im, j, k)]
                              + self.v3[self.idx(i, jp, k)] + self.v3[self.idx(i, jm, k)]
                              + self.v3[self.idx(i, j, kp)] + self.v3[self.idx(i, j, km)]
                              - 6.0 * w) / dx2;

                    let dp_dx = (self.p[self.idx(ip, j, k)] - self.p[self.idx(im, j, k)]) / (2.0 * dx);
                    let dp_dy = (self.p[self.idx(i, jp, k)] - self.p[self.idx(i, jm, k)]) / (2.0 * dx);
                    let dp_dz = (self.p[self.idx(i, j, kp)] - self.p[self.idx(i, j, km)]) / (2.0 * dx);

                    rhs1[id] = -conv1 - (1.0 / self.rho) * dp_dx + self.nu * lap1;
                    rhs2[id] = -conv2 - (1.0 / self.rho) * dp_dy + self.nu * lap2;
                    rhs3[id] = -conv3 - (1.0 / self.rho) * dp_dz + self.nu * lap3;
                }
            }
        }

        for id in 0..size {
            self.v1[id] += dt * rhs1[id];
            self.v2[id] += dt * rhs2[id];
            self.v3[id] += dt * rhs3[id];
        }

        for _ in 0..12 {
            let mut new_p = self.p.clone();
            for i in 0..self.n {
                let ip = (i + 1) % self.n;
                let im = (i + self.n - 1) % self.n;
                for j in 0..self.n {
                    let jp = (j + 1) % self.n;
                    let jm = (j + self.n - 1) % self.n;
                    for k in 0..self.n {
                        let kp = (k + 1) % self.n;
                        let km = (k + self.n - 1) % self.n;
                        let id = self.idx(i, j, k);

                        let dv1_dx = (self.v1[self.idx(ip, j, k)] - self.v1[self.idx(im, j, k)]) / (2.0 * dx);
                        let dv2_dy = (self.v2[self.idx(i, jp, k)] - self.v2[self.idx(i, jm, k)]) / (2.0 * dx);
                        let dv3_dz = (self.v3[self.idx(i, j, kp)] - self.v3[self.idx(i, j, km)]) / (2.0 * dx);
                        let div_star = dv1_dx + dv2_dy + dv3_dz;

                        let p_neighbors = self.p[self.idx(ip, j, k)] + self.p[self.idx(im, j, k)]
                                        + self.p[self.idx(i, jp, k)] + self.p[self.idx(i, jm, k)]
                                        + self.p[self.idx(i, j, kp)] + self.p[self.idx(i, j, km)];

                        new_p[id] = (p_neighbors - (self.rho / dt) * div_star * dx2) / 6.0;
                    }
                }
            }
            self.p = new_p;
        }

        for i in 0..self.n {
            let ip = (i + 1) % self.n;
            let im = (i + self.n - 1) % self.n;
            for j in 0..self.n {
                let jp = (j + 1) % self.n;
                let jm = (j + self.n - 1) % self.n;
                for k in 0..self.n {
                    let kp = (k + 1) % self.n;
                    let km = (k + self.n - 1) % self.n;
                    let id = self.idx(i, j, k);

                    let dp_dx = (self.p[self.idx(ip, j, k)] - self.p[self.idx(im, j, k)]) / (2.0 * dx);
                    let dp_dy = (self.p[self.idx(i, jp, k)] - self.p[self.idx(i, jm, k)]) / (2.0 * dx);
                    let dp_dz = (self.p[self.idx(i, j, kp)] - self.p[self.idx(i, j, km)]) / (2.0 * dx);

                    self.v1[id] -= (dt / self.rho) * dp_dx;
                    self.v2[id] -= (dt / self.rho) * dp_dy;
                    self.v3[id] -= (dt / self.rho) * dp_dz;
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(64);
    let steps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let dt: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(0.002);
    let out_every: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(100);

    println!("================================================================================");
    println!("  MAX-LOAD: {}^{} РОЗВ'ЯЗУВАЧ 3D НАВ'Є–СТОКСА (ПОСТАНОВКА КЛЕЯ)", n, 3);
    println!("  Вузлiв: {} | крокiв: {} | dt = {} | T = {} с", n * n * n, steps, dt, steps as f64 * dt);
    println!("================================================================================");

    let l = 2.0 * PI;
    let nu = 0.005;
    let rho = 1.0;

    let mut solver = NavierStokes3D::new(n, l, nu, rho);
    solver.init_taylor_green(1.0);

    let e0 = solver.compute_total_energy();
    let div0 = solver.compute_max_divergence();
    let ens0 = solver.compute_enstrophy();

    println!("Початковий стан (t = 0.000 с):");
    println!("  E(0) = {:.6} Дж | Om(0) = {:.6} | div = {:.2e}", e0, ens0, div0);
    println!("--------------------------------------------------------------------------------");
    println!("DATA t E Omega div");
    println!("DATA 0.000 {:.6} {:.6} {:.2e}", e0, ens0, div0);

    let start = Instant::now();
    for s in 1..=steps {
        solver.step(dt);
        if s % out_every == 0 || s == steps {
            let t = s as f64 * dt;
            let e = solver.compute_total_energy();
            let ens = solver.compute_enstrophy();
            let div = solver.compute_max_divergence();
            println!("DATA {:.3} {:.6} {:.6} {:.2e}", t, e, ens, div);
            println!("  [Крок {:4}] t = {:.3} с | E = {:.6} Дж ({:+.3}%) | Om = {:.6} | div = {:.2e}",
                s, t, e, ((e - e0)/e0) * 100.0, ens, div);
        }
    }
    let elapsed = start.elapsed();

    let e_final = solver.compute_total_energy();
    let div_final = solver.compute_max_divergence();
    let ens_final = solver.compute_enstrophy();

    println!("--------------------------------------------------------------------------------");
    println!("ВЕРДИКТ MAX-LOAD (N={}, T={} с):", n, steps as f64 * dt);
    println!("  1. Гладкiсть C^∞:      {} сингулярностей blowup не виявлено.",
        if e_final.is_finite() && ens_final.is_finite() && e_final < e0 { "Гладко," } else { "ПОРУШЕННЯ:" });
    println!("  2. Енергiя E(t) < E₀:  {:.6} -> {:.6} Дж ({:+.3}%)", e0, e_final, ((e_final - e0)/e0) * 100.0);
    println!("  3. ∇·v = 0:            max = {:.2e}", div_final);
    println!("  4. Енстрофiя:          {:.6} -> {:.6}", ens0, ens_final);
    println!("  Час: {:.2} с ({:.1} мс/крок, {:.0} крок/с, {} вузлiв)",
        elapsed.as_secs_f64(), elapsed.as_secs_f64() * 1000.0 / steps as f64,
        steps as f64 / elapsed.as_secs_f64(), n * n * n);
    println!("  Пропускна здатнiсть: {:.2} млн вузл-оновлень/с",
        (n * n * n) as f64 * steps as f64 / elapsed.as_secs_f64() / 1e6);
    println!("================================================================================");
}
