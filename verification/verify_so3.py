#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Численная верификация модели P^3 -> so(3) для Навье--Стокса.
=============================================================
Каждая проверка соответствует теореме PDF-документа
«Математическая модель P^3 -> so(3) для Навье--Стокса».

Система (EP-D):   dOmega/dt = Omega x (I^{-1} Omega) - nu L Omega
Штрейн-вариант:   dOmega/dt = Omega x (I^{-1} Omega) + S(t) Omega - nu L Omega

Запуск (через суверенный движок):
    poler-engine --exec 'python3 /home/z/my-project/scripts/so3_paper/verify_so3.py'
"""
import numpy as np

rng = np.random.default_rng(20261002)

# ----------------------------------------------------------------------------
# Ядро: правые части и интегратор RK4
# ----------------------------------------------------------------------------

def rhs(Om, Iinv, nu, L, S=None):
    """Правая часть (EP-D); S(t) -- опциональный тензор деформации."""
    out = np.cross(Om, Iinv @ Om) - nu * (L @ Om)
    if S is not None:
        out = out + S @ Om
    return out


def rk4(f, y0, T, n):
    """Классический RK4 с постоянным шагом. Возвращает (ts, traj[n+1,3])."""
    h = T / n
    ts = np.linspace(0.0, T, n + 1)
    y = np.array(y0, dtype=float)
    traj = np.empty((n + 1, 3))
    traj[0] = y
    for k in range(n):
        t = ts[k]
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[k + 1] = y
    return ts, traj


def golden_min(fun, a, b, iters=80):
    """Золотое сечение: минимум скалярной функции на [a,b]."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = fun(c), fun(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = fun(d)
    return 0.5 * (a + b)


# ----------------------------------------------------------------------------
# Журнал результатов
# ----------------------------------------------------------------------------
ROWS = []


def report(idx, name, measured, expect, ok):
    ROWS.append((idx, name, measured, expect, ok))
    flag = "PASS" if ok else "FAIL"
    print(f"{idx:<4} {name:<62} {measured:<18} {expect:<24} {flag}")


def head(title):
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)


print("=" * 118)
print("ВЕРИФИКАЦИЯ МОДЕЛИ P^3 -> so(3) (НАВЬЕ--СТОКС, РЕДУЦИРОВАННАЯ ДИНАМИКА)")
print("интегратор: RK4, двойная точность, seed=20261002, numpy %s" % np.__version__)
print("=" * 118)

# ----------------------------------------------------------------------------
# Базовые параметры (символьные значения зафиксированы численно)
# ----------------------------------------------------------------------------
I_asym = np.diag([0.90, 0.62, 0.37])          # различные главные моменты
Iinv_asym = np.linalg.inv(I_asym)
OM0 = np.array([2.2, -1.1, 0.7])              # начальная завихренность

# ============================================================================
# C1: Казимир при nu=0 -- ||Omega||^2 сохраняется тождественно (Т.4.1)
# ============================================================================
head("C1--C2: инварианты при nu = 0 (T=200)")


def run_inv(n):
    ts, tr = rk4(lambda t, y: rhs(y, Iinv_asym, 0.0, np.zeros((3, 3))), OM0, 200.0, n)
    C = np.sum(tr ** 2, axis=1)
    H = 0.5 * np.einsum("ij,jk,ik->i", tr, Iinv_asym, tr)
    return np.max(np.abs(C - C[0])) / C[0], np.max(np.abs(H - H[0])) / H[0]


dC_h, dH_h = run_inv(10000)        # h = 2e-2
dC_h2, dH_h2 = run_inv(20000)      # h = 1e-2
rich_c = dC_h / max(dC_h2, 1e-300)
p_c = np.log2(rich_c)
report("C1", f"Дрейф Казимира C=||Om||^2, h=1e-2 (порядок p={p_c:.2f})",
       f"{dC_h2:.2e}", "<= 2e-8, p in [3.5, 5.5]", dC_h2 <= 2e-8 and 3.5 <= p_c <= 5.5)

report("C2", "Дрейф гамильтониана H, h=1e-2", f"{dH_h2:.2e}",
       "<= 2e-8", dH_h2 <= 2e-8)

# ============================================================================
# P1: периодичность всех неравновесных орбит при nu=0 (Т.4.5)
# ============================================================================
head("P1: замкнутость орбиты (Лиувилль), nu = 0")
ts, tr = rk4(lambda t, y: rhs(y, Iinv_asym, 0.0, np.zeros((3, 3))), OM0, 100.0, 200000)
d = np.linalg.norm(tr - OM0, axis=1)
cand = None
for j in range(4000, len(d) - 2):        # пропускаем старт; ищем первый глубокий минимум
    if d[j] < d[j - 1] and d[j] <= d[j + 1] and d[j] < 0.05 * np.linalg.norm(OM0):
        cand = j
        break
if cand is None:
    report("P1", "Возвращение орбиты за T=100", "не найдено", "d(T_p) < 1e-6*||Om0||", False)
else:
    t_lo, t_hi = ts[cand - 1], ts[cand + 1]

    # уточняем период локальным интегрированием от t_lo
    def dist_at(t):
        n_loc = max(int((t - ts[cand - 2]) / 5e-4), 40)
        _, yy = rk4(lambda s, y: rhs(y, Iinv_asym, 0.0, np.zeros((3, 3))), tr[cand - 2],
                    t - ts[cand - 2], n_loc)
        return np.linalg.norm(yy[-1] - OM0)

    Tp = golden_min(dist_at, ts[cand - 1], ts[cand + 1])
    dTp = dist_at(Tp)
    report("P1", f"Замкнутость: d(T_p)/||Om0||, T_p={Tp:.4f}", f"{dTp / np.linalg.norm(OM0):.2e}",
           "<= 1e-5", dTp <= 1e-5 * np.linalg.norm(OM0))

# ============================================================================
# D1--D2: диссипация при nu>0, L>0, [L,I]=0 (Т.4.1, Т.4.2, С.4.3)
# ============================================================================
head("D1--D2: диссипация (nu=0.1, L=diag(1,1.7,2.3), T=40)")
nu1, L1 = 0.10, np.diag([1.0, 1.7, 2.3])
ts, tr = rk4(lambda t, y: rhs(y, Iinv_asym, nu1, L1), OM0, 40.0, 20000)
C = np.sum(tr ** 2, axis=1)
# невязка тождества dC/dt = -2 nu <Om, L Om> вдоль траектории (алгебраически)
Cdot_alg = -2.0 * nu1 * np.einsum("ij,jk,ik->i", tr, L1, tr)
Cdot_fd = np.gradient(C, ts)
res_id = np.max(np.abs(Cdot_fd - Cdot_alg)) / np.max(np.abs(Cdot_alg))
report("D1", "Невязка dC/dt = -2nu<Om,L Om> (аналитика vs числ.)", f"{res_id:.2e}",
       "<= 5e-4 (шаг fd)", res_id <= 5e-4)
report("D2", "Монотонность C(t): max (C-C0)/C0", f"{(C.max() - C[0]) / C[0]:.2e}",
       "<= 0 (строго <= 0)", C.max() <= C[0] + 1e-12 * C[0])
lam_min = min(np.linalg.eigvalsh(L1))
viol = np.max(C / (C[0] * np.exp(-2.0 * nu1 * lam_min * ts)))
report("D3", "Экспонента: max C/(C0*exp(-2nu*lam_min*t))", f"{viol:.6f}",
       "<= 1.0000001", viol <= 1.0000001)
H = 0.5 * np.einsum("ij,jk,ik->i", tr, Iinv_asym, tr)
report("D4", "Монотонность H при [L,I]=0: max (H-H0)/H0", f"{(H.max() - H[0]) / H[0]:.2e}",
       "<= 0", H.max() <= H[0] + 1e-12 * H[0])

# ============================================================================
# E1: тонкость -- некоммутирующие L и I: H растёт, C убывает (Зам.4.3)
# ============================================================================
head("E1: контрпример [L,I]!=0 (I=diag(.1,1,1), L=[[1,1,0],[1,1.5,0],[0,0,1]])")
I_nc = np.diag([0.10, 1.0, 1.0])
Iinv_nc = np.linalg.inv(I_nc)
L_nc = np.array([[1.0, 1.0, 0.0], [1.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
eig_L = np.linalg.eigvalsh(L_nc)
OM0_nc = np.array([-0.5, 1.0, 0.0])
Hdot0 = -0.10 * OM0_nc @ Iinv_nc @ L_nc @ OM0_nc
ts, tr = rk4(lambda t, y: rhs(y, Iinv_nc, 0.10, L_nc), OM0_nc, 4.0, 40000)
C = np.sum(tr ** 2, axis=1)
H = 0.5 * np.einsum("ij,jk,ik->i", tr, Iinv_nc, tr)
report("E1a", "L >= 0: мин. собственное значение L", f"{eig_L.min():.4f}", "> 0", eig_L.min() > 0)
report("E1b", "dH/dt(0) = -nu<Om,I^-1 L Om> > 0 (аналитика)", f"{Hdot0:+.4f}",
       "= +0.1500", abs(Hdot0 - 0.15) < 1e-12)
report("E1c", "Рост H: max H/H0 при строго убывающем C", f"{H.max() / H[0]:.4f}",
       "> 1.005 и max C/C0 <= 1", (H.max() / H[0] > 1.005) and (C.max() <= C[0] * (1 + 1e-12)))

# ============================================================================
# B1--B2: отсутствие blow-up (Т.6.1)
# ============================================================================
head("B1--B2: априорная оценка ||Om(t)|| <= ||Om0|| (горизонты T=1000 / T=60)")
ts, tr = rk4(lambda t, y: rhs(y, Iinv_asym, 0.0, np.zeros((3, 3))), OM0, 1000.0, 100000)
nrm = np.linalg.norm(tr, axis=1)
sup_ratio = nrm.max() / np.linalg.norm(OM0)
report("B1", "nu=0, T=1000, h=1e-2: max ||Om(t)||/||Om0||", f"{sup_ratio:.10f}",
       "<= 1 + 1e-7", sup_ratio <= 1.0 + 1e-7)
ts, tr = rk4(lambda t, y: rhs(y, Iinv_asym, nu1, L1), OM0, 60.0, 30000)
nrm = np.linalg.norm(tr, axis=1)
bnd = np.linalg.norm(OM0) * np.exp(-nu1 * lam_min * ts)
sup2 = (nrm / bnd).max()
report("B2", "nu>0, L>0: max ||Om||/(||Om0||e^{-nu*lam*t}), T=60", f"{sup2:.10f}",
       "<= 1 + 1e-9", sup2 <= 1.0 + 1e-9)
report("B2b", "Финальное затухание ||Om(T)||/||Om0||, T=60", f"{nrm[-1] / np.linalg.norm(OM0):.2e}",
       "<= e^{-nu*lam*T} = e^{-6} ~ 2.48e-3", nrm[-1] / np.linalg.norm(OM0) <= 2.48e-3)

# ============================================================================
# A1--A2: явные осесимметричные решения (П.6.2)
# ============================================================================
head("A1--A2: осесимметрия I1=I2=1, I3=0.5 (порядок сходимости RK4)")


def exact_axisym_nu0(z0, O30, t):
    kap = (1.0 / 0.5 - 1.0) * O30
    z = z0 * np.exp(-1j * kap * t)
    return np.array([z.real, z.imag, O30])


def err_axisym(nu, n, T=200.0):
    Iinv_ax = np.diag([1.0, 1.0, 2.0])
    L_ax = np.diag([1.0, 1.0, 2.0])
    z0 = complex(1.3, -0.8)
    O30 = 0.9
    y0 = np.array([z0.real, z0.imag, O30])
    ts, tr = rk4(lambda t, y: rhs(y, Iinv_ax, nu, L_ax), y0, T, n)
    if nu == 0.0:
        ex = np.array([exact_axisym_nu0(z0, O30, T)])
    else:
        lam33 = nu * 2.0
        O3t = O30 * np.exp(-lam33 * T)
        phase = (1.0 / 0.5 - 1.0) * O30 * (1.0 - np.exp(-lam33 * T)) / lam33
        z = z0 * np.exp(-nu * 1.0 * T) * np.exp(-1j * phase)
        ex = np.array([[z.real, z.imag, O3t]])
    return np.max(np.abs(tr[-1] - ex[0]))


e1 = err_axisym(0.0, 800)       # h = 0.25
e2 = err_axisym(0.0, 1600)     # h = 0.125
order0 = np.log2(e1 / e2)
report("A1", "nu=0: ошибка при h=0.25 / порядок", f"{e1:.2e} / p={order0:.2f}",
       "p = 4.00 +- 0.3", e1 < 0.02 and abs(order0 - 4.0) < 0.3)
e1 = err_axisym(0.15, 800)
e2 = err_axisym(0.15, 1600)
orderv = np.log2(e1 / e2)
report("A2", "nu=0.15 (прецессия с затуханием): ошибка / порядок", f"{e1:.2e} / p={orderv:.2f}",
       "p = 4.00 +- 0.3", e1 < 0.02 and abs(orderv - 4.0) < 0.3)

# ============================================================================
# M1--M2: проектор МакВини (Л.5.6--5.8, П.5.9)
# ============================================================================
head("M1--M2: P=Om Om^T/||Om||^2 и Q(P)=3P^2-2P^3")
worst_P, worst_QP = 0.0, 0.0
for _ in range(500):
    Om = rng.normal(size=3) * 3.0
    P = np.outer(Om, Om) / (Om @ Om)
    QP = 3.0 * P @ P - 2.0 * P @ P @ P
    worst_P = max(worst_P, np.linalg.norm(P @ P - P))
    worst_QP = max(worst_QP, np.linalg.norm(QP - P))
report("M1", "Точный ранг-1 проектор: max||P^2-P||, max||Q(P)-P|| (500 случайных)",
       f"{worst_P:.1e}/{worst_QP:.1e}", "<= 1e-13 / <= 1e-13", worst_P <= 1e-13 and worst_QP <= 1e-13)

ratio_max, ratio_at = 0.0, 0.0
for _ in range(10000):
    lam = rng.uniform(-0.08, 1.08, size=3)
    Qm, _Rm = np.linalg.qr(rng.normal(size=(3, 3)))
    lam = np.clip(lam, 0.0, 1.0)
    P = Qm @ np.diag(lam) @ Qm.T
    dP = np.linalg.norm(P @ P - P)                     # = max lam(1-lam)
    QP = 3.0 * P @ P - 2.0 * P @ P @ P
    dQ = np.linalg.norm(QP @ QP - QP)
    if dP > 1e-12:
        r = dQ / (4.0 * dP ** 2)
        if r > ratio_max:
            ratio_max, ratio_at = r, np.sort(lam)
report("M2", "Острота: max dQ/(4 dP^2) по 10^4 возмущённых P", f"{ratio_max:.6f}",
       "<= 1, достигается ~1 (lam=1/2)", ratio_max <= 1.0 + 1e-12)
print(f"     спектр в точке максимума: lam = {np.sort(ratio_at)}")

# ============================================================================
# H1: Хопф--Коул (Т.5.4): u=-2nu d_x ln phi, phi_t = nu phi_xx
# ============================================================================
head("H1: Бюргерс, u0=sin(x), периодическая область (N=2048), phi -- спектрально точное")
N, Ldom = 2048, 2.0 * np.pi
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
dx = x[1] - x[0]
k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)


def make_phi0(nu_b):
    U0 = 1.0 - np.cos(x)
    return np.exp(-U0 / (2.0 * nu_b))


def burgers_residual(t, nu_b, ph0h):
    ph = np.fft.irfft(np.exp(-nu_b * k ** 2 * t) * ph0h)
    phx = np.fft.irfft(1j * k * np.fft.rfft(ph))
    phxx = np.fft.irfft(-(k ** 2) * np.fft.rfft(ph))
    g = phx / ph
    u = -2.0 * nu_b * g
    ux = -2.0 * nu_b * np.fft.irfft(1j * k * np.fft.rfft(g))
    uxx = -2.0 * nu_b * np.fft.irfft(-(k ** 2) * np.fft.rfft(g))
    r_ = nu_b * phxx / ph
    ut = -2.0 * nu_b * np.fft.irfft(1j * k * np.fft.rfft(r_))
    res = ut + u * ux - nu_b * uxx
    scale = np.max(np.abs(ut) + np.abs(u * ux) + np.abs(nu_b * uxx))
    return u, np.max(np.abs(res)), np.max(np.abs(res)) / scale, np.max(np.abs(ux))


# H1a: невязка тождества Коула--Хопфа при nu=0.1 (диапазон phi: e^5)
nu_a = 0.10
ph0h_a = np.fft.rfft(make_phi0(nu_a))
res_abs_a, res_rel_a = 0.0, 0.0
for t in (0.5, 1.0, 5.0):
    _u, ra, rr, _ux = burgers_residual(t, nu_a, ph0h_a)
    res_abs_a, res_rel_a = max(res_abs_a, ra), max(res_rel_a, rr)
report("H1a", "nu=0.1: невязка Бюргерса, абс. / относит.", f"{res_abs_a:.2e}/{res_rel_a:.2e}",
       "относит. <= 1e-3", res_rel_a <= 1e-3)

# H1b: nu=0.05 -- ограниченность градиента и вязкое сглаживание после t~1
nu_b = 0.05
ph0h_b = np.fft.rfft(make_phi0(nu_b))
uxs = {}
for t in (1e-8, 0.5, 1.0, 2.0, 5.0):
    _u, _ra, _rr, uxs[t] = burgers_residual(t, nu_b, ph0h_b)
bounded = max(uxs.values()) < 10.0
smoothing = uxs[5.0] < uxs[1.0]
report("H1b", "nu=0.05: max|u_x| при t=0,.5,1,2,5", 
       "/".join(f"{uxs[t]:.2f}" for t in (1e-8, 0.5, 1.0, 2.0, 5.0)),
       "< 10 всюду и убывание после t~1", bounded and smoothing)

# ============================================================================
# S1: условная теорема со штрейном (Т.7.1)
# ============================================================================
head("S1: штрейн-расширение: dOm/dt = Om x I^-1 Om + S(t) Om - nu L Om")
Iinv_s, L_s, nu_s = Iinv_asym, np.diag([1.0, 1.0, 1.0]), 0.10
Rm = rng.normal(size=(3, 3))
Rm = 0.5 * (Rm + Rm.T)
Rm /= np.linalg.norm(Rm, 2)
s0_int = 0.80
ts, tr = rk4(lambda t, y: rhs(y, Iinv_s, nu_s, L_s, S=s0_int * np.exp(-t) * Rm), OM0, 10.0, 20000)
nrm = np.linalg.norm(tr, axis=1)
bnd = np.linalg.norm(OM0) * np.exp(s0_int * (1.0 - np.exp(-ts)) - nu_s * ts)
viol = (nrm / bnd).max()
report("S1a", "Интегрируемый штрейн s0*e^-t: max ||Om||/bound", f"{viol:.10f}",
       "<= 1 + 1e-9", viol <= 1.0 + 1e-9)
# изотропный постоянный штрейн: S = s0*E --> рост e^{(s0-nu)t} РОВНО по границе
s0_iso = 0.50
ts, tr = rk4(lambda t, y: rhs(y, Iinv_s, nu_s, L_s, S=s0_iso * np.eye(3)), OM0, 10.0, 20000)
nrm = np.linalg.norm(tr, axis=1)
bnd = np.linalg.norm(OM0) * np.exp((s0_iso - nu_s) * ts)
ratio_end = nrm[-1] / bnd[-1]
report("S1b", "S=s0*E (s0>nu*lam_min): рост и точность границы, ratio(T)", f"{ratio_end:.8f}",
       "= 1.000000 (оценка достигается)", abs(ratio_end - 1.0) <= 1e-6)
print(f"     ||Om(0)||={np.linalg.norm(OM0):.3f}, ||Om(T)||={nrm[-1]:.3f}, "
      f"bound={bnd[-1]:.3f} --> взрывной рост без условия теоремы 7.1")

# ============================================================================
# L1: 2D-сечение (П.5.2): Omega_3 = 0 инвариантно при I1=I2
# ============================================================================
head("L1: 2D-редукция (I1=I2=1, I3=0.5, nu=0.1, L=diag(1,1,2), T=20)")
Iinv_2d = np.diag([1.0, 1.0, 2.0])
L_2d = np.diag([1.0, 1.0, 2.0])
OM0_2d = np.array([1.3, -0.8, 0.0])
ts, tr = rk4(lambda t, y: rhs(y, Iinv_2d, 0.10, L_2d), OM0_2d, 20.0, 20000)
w3 = np.max(np.abs(tr[:, 2]))
report("L1a", "Инвариантность плоскости: max |Omega_3(t)|", f"{w3:.2e}",
       "<= 1e-12", w3 <= 1e-12)
C2 = tr[:, 0] ** 2 + tr[:, 1] ** 2
Cdot_alg = -0.2 * (tr[:, 0] ** 2 + tr[:, 1] ** 2)  # -2 nu L11 (Om1^2+Om2^2)
Cdot_fd = np.gradient(C2, ts)
res2 = np.max(np.abs(Cdot_fd - Cdot_alg)) / np.max(np.abs(Cdot_alg))
report("L1b", "Невязка dC_2/dt = -2 nu L11 (Om1^2+Om2^2)", f"{res2:.2e}",
       "<= 5e-4", res2 <= 5e-4)

# ----------------------------------------------------------------------------
# Итог
# ----------------------------------------------------------------------------
print("\n" + "=" * 118)
n_pass = sum(1 for r in ROWS if r[4])
print(f"ИТОГ: {n_pass}/{len(ROWS)} проверок пройдено.")
if n_pass == len(ROWS):
    print("Все тождества и оценки теорем 3.1--7.1 подтверждены численно;")
    print("границы метода (штрейн-член, S1b) воспроизведены точно -- модель верифицирована.")
print("=" * 118)
