#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Спектральный эталон 3D Навье-Стокса: Тейлор-Грин на торе T^3 = [0, 2*pi]^3.

Назначение: независимая высокоточная ссылка для кросс-сравнения с нативным
решателем poler-engine/crates/pqc/examples/bench_navier_stokes_clay.rs
(32^3, центральные разности 2-го порядка, явный Эйлер, 12 итераций Якоби
для давления -> ПРИБЛИЖЁННАЯ проекция Лере).

Здесь: Фурье-псевдоспектральный метод
    dv/dt = -P[(v.grad)v] + nu * Laplacian v,
    P = I - k k^T / |k|^2   (ТОЧНАЯ спектральная проекция Лере),
    2/3-правило деалиасинга, интегрирование RK4.

Энергия считается той же квадратурой, что у пользователя:
    E = (rho/2) * dx^3 * sum |v|^2   (для T^3 трапеция точна).
Аналитика: E(0) = pi^3, Omega(0) = 3*pi^3.

Выход: таблицы t | E | Omega | max|div|, баланс dE/dt = -2*nu*Omega,
сходимость по dt и по разрешению N.
"""
import numpy as np

PI = np.pi


def build(N, L=2 * PI):
    k_int = np.fft.fftfreq(N, d=1.0 / N)          # целые волновые числа
    kx = k_int[:, None, None]
    ky = k_int[None, :, None]
    kz = k_int[None, None, :]
    KX = (2 * PI / L) * kx
    KY = (2 * PI / L) * ky
    KZ = (2 * PI / L) * kz
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2s = np.where(K2 == 0, 1.0, K2)
    dealias = (np.abs(kx) <= N / 3) & (np.abs(ky) <= N / 3) & (np.abs(kz) <= N / 3)
    dealias = dealias.astype(np.float64)
    x = np.arange(N) * (L / N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return KX, KY, KZ, K2, K2s, dealias, X, Y, Z


def tg_initial(X, Y, Z, v0=1.0):
    """Поле Тейлора-Грина в точности как в bench_navier_stokes_clay.rs."""
    vx = v0 * np.sin(X) * np.cos(Y) * np.cos(Z)
    vy = -v0 * np.cos(X) * np.sin(Y) * np.cos(Z)
    vz = np.zeros_like(X)
    return [np.fft.fftn(v) for v in (vx, vy, vz)]


def project(vh, KX, KY, KZ, K2, K2s):
    kdv = (KX * vh[0] + KY * vh[1] + KZ * vh[2]) / K2s
    kdv = np.where(K2 == 0, 0.0, kdv)
    return [vh[0] - KX * kdv, vh[1] - KY * kdv, vh[2] - KZ * kdv]


def rhs(vh, KX, KY, KZ, K2, dealias, nu):
    # -P[(v.grad)v] с деалиасингом
    vx, vy, vz = [np.fft.ifftn(c).real for c in vh]
    conv = []
    for comp in vh:
        dx_i = np.fft.ifftn(1j * KX * comp).real
        dy_i = np.fft.ifftn(1j * KY * comp).real
        dz_i = np.fft.ifftn(1j * KZ * comp).real
        conv.append(vx * dx_i + vy * dy_i + vz * dz_i)
    ch = [np.fft.fftn(c) * dealias for c in conv]
    nl = [-p for p in project(ch, KX, KY, KZ, K2, np.where(K2 == 0, 1.0, K2))]
    # + nu * Laplacian v  (спектрально, v уже деалиасировано)
    return [nl[i] - nu * K2 * vh[i] for i in range(3)]


def diagnostics(vh, KX, KY, KZ, dx3, rho):
    vp = [np.fft.ifftn(c).real for c in vh]
    E = 0.5 * rho * dx3 * sum(np.sum(v * v) for v in vp)
    wx = np.fft.ifftn(1j * (KY * vh[2] - KZ * vh[1])).real
    wy = np.fft.ifftn(1j * (KZ * vh[0] - KX * vh[2])).real
    wz = np.fft.ifftn(1j * (KX * vh[1] - KY * vh[0])).real
    Om = 0.5 * dx3 * (np.sum(wx * wx) + np.sum(wy * wy) + np.sum(wz * wz))
    div = np.fft.ifftn(1j * (KX * vh[0] + KY * vh[1] + KZ * vh[2])).real
    return E, Om, float(np.max(np.abs(div)))


def run(N=32, dt=0.001, t_end=0.4, nu=0.005, rho=1.0, L=2 * PI,
        sample_ts=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)):
    KX, KY, KZ, K2, K2s, dealias, X, Y, Z = build(N, L)
    dx3 = (L / N) ** 3
    vh = tg_initial(X, Y, Z)
    vh = project(vh, KX, KY, KZ, K2, K2s)
    vh = [c * dealias for c in vh]

    rows = []
    n_steps = int(round(t_end / dt))
    out_every = max(1, int(round(0.05 / dt)))
    E, Om, div = diagnostics(vh, KX, KY, KZ, dx3, rho)
    rows.append((0.0, E, Om, div))
    for s in range(1, n_steps + 1):
        k1 = rhs(vh, KX, KY, KZ, K2, dealias, nu)
        v2 = [vh[i] + 0.5 * dt * k1[i] for i in range(3)]
        k2 = rhs(v2, KX, KY, KZ, K2, dealias, nu)
        v3 = [vh[i] + 0.5 * dt * k2[i] for i in range(3)]
        k3 = rhs(v3, KX, KY, KZ, K2, dealias, nu)
        v4 = [vh[i] + dt * k3[i] for i in range(3)]
        k4 = rhs(v4, KX, KY, KZ, K2, dealias, nu)
        vh = [vh[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
              for i in range(3)]
        if s % out_every == 0 or s == n_steps:
            E, Om, div = diagnostics(vh, KX, KY, KZ, dx3, rho)
            rows.append((s * dt, E, Om, div))
    return rows


def main():
    print("=" * 78)
    print("  СПЕКТРАЛЬНЫЙ ЭТАЛОН 3D NSE (Тейлор-Грин, T^3=[0,2pi]^3, nu=0.005, rho=1)")
    print("  Псевдоспектральный метод: точная проекция Лере + 2/3-деалиасинг + RK4")
    print("=" * 78)
    print("Аналитика: E(0) = pi^3 = %.6f Дж | Omega(0) = 3*pi^3 = %.6f" % (PI ** 3, 3 * PI ** 3))
    print()

    configs = [(32, 0.002), (32, 0.001), (48, 0.001), (64, 0.001)]
    results = {}
    for (N, dt) in configs:
        rows = run(N=N, dt=dt)
        results[(N, dt)] = rows
        print("--- N=%d, dt=%.4f ---" % (N, dt))
        print("   t      E(t) [Дж]        Omega(t)      max|div|")
        for (t, E, Om, div) in rows:
            if abs(t * 100 - round(t * 100)) < 1e-9 and round(t, 2) % 0.1 < 1e-9:
                print("  %.2f   %.6f      %.4f      %.2e" % (t, E, Om, div))
        print()

    # Сходимость по t=0.4
    print("=" * 78)
    print("СХОДИМОСТЬ E(0.4) [Дж]:")
    ref = results[(64, 0.001)]
    E_ref = [E for (t, E, _, _) in ref if abs(t - 0.4) < 1e-9][0]
    for (N, dt) in configs:
        E04 = [E for (t, E, _, _) in results[(N, dt)] if abs(t - 0.4) < 1e-9][0]
        print("  N=%2d dt=%.4f : E(0.4) = %.6f   (отклонение от N=64: %+.2e)"
              % (N, dt, E04, E04 - E_ref))

    # Баланс энергии dE/dt = -2*nu*Omega (на сетке N=64)
    print()
    print("БАЛАНС ЭНЕРГИИ dE/dt = -2*nu*Omega (N=64, dt=0.001):")
    rows = results[(64, 0.001)]
    for i in range(1, len(rows)):
        t0, E0, Om0, _ = rows[i - 1]
        t1, E1, Om1, _ = rows[i]
        dtl = t1 - t0
        dEdt_num = (E1 - E0) / dtl
        dEdt_theory = -2 * 0.005 * 0.5 * (Om0 + Om1)
        print("  t=%.2f..%.2f : числ. %+.4f  теор. %+.4f  |отн.ош| %.1e"
              % (t0, t1, dEdt_num, dEdt_theory,
                 abs(dEdt_num - dEdt_theory) / max(1e-12, abs(dEdt_theory))))

    # Кросс-сравнение с нативным решателем пользователя (bench_navier_stokes_clay.rs)
    # Значения -- дословный запуск его бинарника (rustc -O, 32^3, dt=0.002):
    USER = {
        0.0:  (31.006277, 91.829569, 2.66e-15),
        0.1:  (30.914423, 91.645671, 7.64e-5),
        0.2:  (30.823303, 91.638476, 8.07e-5),
        0.3:  (30.732854, 91.806165, 8.66e-5),
        0.4:  (30.642942, 92.147914, 9.51e-5),
    }
    print()
    print("=" * 78)
    print("КРОСС-СРАВНЕНИЕ: решатель пользователя (32^3, Эйлер, Якоби-12) vs эталон")
    print("=" * 78)
    print("   t  |  E_user     E_ref     dE       |  Om_user   Om_ref    |  div_user  div_ref")
    for (t, E, Om, div) in results[(64, 0.001)]:
        if any(abs(t - tt) < 1e-9 for tt in (0.0, 0.1, 0.2, 0.3, 0.4)):
            Eu, Ou, Du = USER[round(t, 1)]
            print("  %.1f  | %.6f  %.6f  %+.2e | %.4f  %.4f | %.2e  %.2e"
                  % (t, Eu, E, Eu - E, Ou, Om, Du, div))
    e_u = USER[0.4][0]
    print()
    print("  Отклонение E(0.4): %+.6f Дж (%+.3f%% от E(0))"
          % (e_u - E_ref, (e_u - E_ref) / PI ** 3 * 100))
    print("  Отношение диссипации (user/ref): %.4f"
          % ((PI ** 3 - e_u) / (PI ** 3 - E_ref)))
    print("  Затухание энстрофии решателя (sinc^2-эффект разностей): %.5f"
          % (USER[0.0][1] / (3 * PI ** 3)))


if __name__ == "__main__":
    main()
