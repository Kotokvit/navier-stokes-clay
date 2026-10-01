#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Спектральный эталон MAX-LOAD: 3D Навье-Стокс, Тейлор-Грин на торе [0, 2*pi]^3.

Метод бит-в-бит идентичен верифицированному ns_spectral_reference.py:
    dv/dt = -P[(v.grad)v] + nu * Laplacian v,
    P = I - k k^T / |k|^2  (ТОЧНАЯ спектральная проекция Лере),
    2/3-правило деалиасинга, интегрирование RK4.
Изменение ТОЛЬКО одно: горизонт T и параметры из argv.

Запуск: python3 ns_spectral_max.py [N] [dt] [T] [out_every]
Выход: строки DATA t E Omega div  (парсятся кросс-анализом).
"""
import sys
import numpy as np

PI = np.pi


def build(N, L=2 * PI):
    k_int = np.fft.fftfreq(N, d=1.0 / N)
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
    vx = v0 * np.sin(X) * np.cos(Y) * np.cos(Z)
    vy = -v0 * np.cos(X) * np.sin(Y) * np.cos(Z)
    vz = np.zeros_like(X)
    return [np.fft.fftn(v) for v in (vx, vy, vz)]


def project(vh, KX, KY, KZ, K2, K2s):
    kdv = (KX * vh[0] + KY * vh[1] + KZ * vh[2]) / K2s
    kdv = np.where(K2 == 0, 0.0, kdv)
    return [vh[0] - KX * kdv, vh[1] - KY * kdv, vh[2] - KZ * kdv]


def rhs(vh, KX, KY, KZ, K2, dealias, nu):
    vx, vy, vz = [np.fft.ifftn(c).real for c in vh]
    conv = []
    for comp in vh:
        dx_i = np.fft.ifftn(1j * KX * comp).real
        dy_i = np.fft.ifftn(1j * KY * comp).real
        dz_i = np.fft.ifftn(1j * KZ * comp).real
        conv.append(vx * dx_i + vy * dy_i + vz * dz_i)
    ch = [np.fft.fftn(c) * dealias for c in conv]
    nl = [-p for p in project(ch, KX, KY, KZ, K2, np.where(K2 == 0, 1.0, K2))]
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


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.004
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    out_every = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    nu = 0.005
    rho = 1.0
    L = 2 * PI

    print("REF N=%d dt=%.4f T=%.2f nu=%.3f (Leray-exact + 2/3-dealias + RK4)"
          % (N, dt, T, nu))
    print("Analytic: E(0)=pi^3=%.6f  Om(0)=3pi^3=%.6f" % (PI ** 3, 3 * PI ** 3))

    KX, KY, KZ, K2, K2s, dealias, X, Y, Z = build(N, L)
    dx3 = (L / N) ** 3
    vh = tg_initial(X, Y, Z)
    vh = project(vh, KX, KY, KZ, K2, K2s)
    vh = [c * dealias for c in vh]

    E, Om, div = diagnostics(vh, KX, KY, KZ, dx3, rho)
    print("DATA %.3f %.6f %.6f %.2e" % (0.0, E, Om, div))

    n_steps = int(round(T / dt))
    out_step = max(1, int(round(out_every / dt)))
    import time
    t0w = time.time()
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
        if s % out_step == 0 or s == n_steps:
            E, Om, div = diagnostics(vh, KX, KY, KZ, dx3, rho)
            print("DATA %.3f %.6f %.6f %.2e" % (s * dt, E, Om, div))
            sys.stdout.flush()
    wall = time.time() - t0w
    print("REF_DONE steps=%d wall=%.1fs (%.3f s/step)" % (n_steps, wall, wall / n_steps))


if __name__ == "__main__":
    main()
