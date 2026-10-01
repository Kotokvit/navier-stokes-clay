#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Штурм-бисекция через КАЛЬКУЛЯТОР poler-engine (v0.61.0).

Задача: собственные значения трёхдиагонали T_n = tridiag(2,-1,n), n=128.
Встроенная `eigen` ломается на n>=28 (взрыв коэффициентов charpoly
Фаддеева-Леврерье -- феномен Уилкинсона). Граница расширяется БЕЗ изменения
движка: та же арифметика калькулятора (переменные, +, *, sign), тот же REPL,
но структурно-осознанный алгоритм:

  * последовательность Штурма ведущих миноров T_k - xI:
      d_0 = 1,  d_1 = 2-x,  d_k = (2-x)*d_{k-1} - d_{k-2}
  * N(x) = число перемен знака в {d_0..d_n} = число собственных значений < x
  * бисекция каждого собственного значения по монотонности N(x)

Каждая строка скрипта -- инструкция `calc ...` РЕПЛа движка; весь тяжёлый
счёт делает САМ калькулятор. Python только генерирует текст и читает ответ.
Кросс-чек аналитикой: lambda_k = 2 - 2cos(k*pi/(n+1)).
"""
import subprocess
import sys
import math
import time

ENGINE = "poler-engine"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
TOL = 1e-12          # абсолютная точность бисекции
BATCH = 16           # штурм-счётов за одну REPL-сессию
TAGS = "abcdefghijklmnop"
PROMPT = "poler> "


def sturm_lines(tag, x):
    """REPL-строки: полная последовательность Штурма + счёт для одного x."""
    lines = [
        f"calc x{tag} = {x!r}",
        f"calc a{tag} = 2 - x{tag}",
        f"calc d{tag}0 = 1",
        f"calc d{tag}1 = a{tag}",
    ]
    for k in range(2, N + 1):
        lines.append(f"calc d{tag}{k} = a{tag}*d{tag}{k-1} - d{tag}{k-2}")
    terms = " + ".join(f"(1-sign(d{tag}{k}*d{tag}{k-1}))/2" for k in range(1, N + 1))
    lines.append(f"calc n{tag} = {terms}")
    return lines


def sturm_counts(xs):
    """Пакетный запрос: до BATCH штурм-счётов за один запуск движка."""
    xs = list(xs)
    assert len(xs) <= BATCH
    script_lines = []
    for tag, x in zip(TAGS, xs):
        script_lines += sturm_lines(tag, x)
    script = "\n".join(script_lines) + "\nquit\n"
    out = subprocess.run([ENGINE, "--shell"], input=script.encode(),
                         capture_output=True, timeout=1200).stdout.decode()
    vals = [l[len(PROMPT):].strip() for l in out.splitlines()
            if l.startswith(PROMPT)]
    # последний элемент -- "до свидания"; отбросим его
    if vals and "до свидания" in vals[-1]:
        vals = vals[:-1]
    assert len(vals) == len(xs) * (N + 4), \
        f"ожидалось {len(xs)*(N+4)} значений, получено {len(vals)}"
    counts = []
    for i in range(len(xs)):
        pos = (i + 1) * (N + 4) - 1
        counts.append(int(round(float(vals[pos]))))
    return counts


def main():
    t0 = time.time()
    print(f"=== ШТУРМ-БИСЕКЦИЯ n={N} через калькулятор poler-engine ===")
    # дымовой тест: N(2.5) против аналитики
    theo = sum(1 for k in range(1, N + 1)
               if 2 - 2 * math.cos(k * math.pi / (N + 1)) < 2.5)
    got = sturm_counts([2.5])[0]
    print(f"дымовой тест: N(2.5)={got}, теория={theo}",
          "OK" if got == theo else "FAIL")
    assert got == theo

    # параллельная бисекция всех n собственных значений
    lo = [0.0] * N
    hi = [4.0] * N
    active = list(range(N))          # индексы 0..N-1 -> собственное значение k=i+1
    queries = 0
    round_no = 0
    while active:
        round_no += 1
        # средние точки активных -- батчами по BATCH
        mids = [0.5 * (lo[i] + hi[i]) for i in active]
        cnts = {}
        for s in range(0, len(mids), BATCH):
            chunk = mids[s:s + BATCH]
            cs = sturm_counts(chunk)
            queries += len(chunk)
            for j, c in zip(range(s, s + len(chunk)), cs):
                cnts[j] = c
        nxt = []
        for idx, i in enumerate(active):
            k = i + 1                     # k-е собственное значение (снизу)
            mid = mids[idx]
            c = cnts[idx]
            if c >= k:
                hi[i] = mid
            else:
                lo[i] = mid
            if hi[i] - lo[i] > TOL:
                nxt.append(i)
        active = nxt
        if round_no % 10 == 0:
            worst = max(hi[i] - lo[i] for i in range(N))
            print(f"  раунд {round_no}: активно {len(active)}, "
                  f"худший интервал {worst:.2e}, запросов {queries}")

    lam = [0.5 * (lo[i] + hi[i]) for i in range(N)]
    dt = time.time() - t0

    # кросс-чек аналитикой
    errs = [abs(lam[k - 1] - (2 - 2 * math.cos(k * math.pi / (N + 1))))
            for k in range(1, N + 1)]
    mx = max(errs)
    print(f"\nитераций бисекции: {round_no}; штурм-запросов: {queries}; "
          f"время {dt:.1f} с")
    print(f"МАКС. ОШИБКА против 2-2cos(k*pi/(n+1)): {mx:.3e}")
    print("первые 5:", [f"{v:.12f}" for v in lam[:5]])
    print("последние 5:", [f"{v:.12f}" for v in lam[-5:]])
    prod = math.prod(lam)
    print(f"prod(lambda)={prod:.6f}; det(T_{N})={N+1} "
          f"(откл. {abs(prod-(N+1))/(N+1):.2e})")
    out_path = f"/home/z/my-project/download/sturm_eigen_n{N}.txt"
    with open(out_path, "w") as f:
        f.write(f"Sturm bisection n={N} via poler-engine calc (REPL)\n")
        f.write(f"rounds={round_no} queries={queries} time={dt:.1f}s\n")
        f.write(f"max err vs analytics: {mx:.3e}\n")
        f.write("k lambda_sturm lambda_theory err\n")
        for k in range(1, N + 1):
            t = 2 - 2 * math.cos(k * math.pi / (N + 1))
            f.write(f"{k} {lam[k-1]:.15f} {t:.15f} {errs[k-1]:.2e}\n")
    print(f"сохранено: {out_path}")


if __name__ == "__main__":
    main()
