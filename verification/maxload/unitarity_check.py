#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Квантовая унитарность калькулятора: |Psi(t)|^2 = 1 для всех t.
Вызывает poler-engine calc schrodinger(...) и парсит комплексный вектор честно."""
import re
import subprocess
import os

ENGINE = os.path.expanduser("~/.local/bin/poler-engine")


def engine_calc(expr):
    out = subprocess.run([ENGINE, "--exec", "calc " + expr],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip().splitlines()[0] if out.stdout else ""


def parse_cplx_vector(s):
    """'[a; b i; c + d i; ...]' -> список Complex."""
    body = s.strip().lstrip("[").rstrip("]")
    vec = []
    for item in body.split(";"):
        item = item.strip()
        # форма: x, x i, x + y i, x - y i, i, -i
        m = re.match(r"^([-+]?[\d.eE+-]*)?\s*(?:([+-])\s*((?:[\d.eE+-]*)?)i)?$|^([-+]?[\d.]*)i$", item)
        if item.endswith("i") and not item[:-1].strip() in ("", "+", "-"):
            # чисто мнимое: 'b i' или 'bi'
            b = item[:-1].strip()
            vec.append(complex(0.0, float(b)))
        elif "i" in item:
            # x + y i
            mm = re.match(r"^([-+]?[\d.eE+-]+)\s*([+-])\s*([\d.eE+-]*)i$", item)
            if mm:
                x = float(mm.group(1))
                y = float((mm.group(2) + (mm.group(3) or "1")))
                vec.append(complex(x, y))
            else:
                vec.append(complex(0.0, float(item.replace("i", "").strip())))
        else:
            vec.append(complex(float(item), 0.0))
    return vec


def norm2(vec):
    return sum(abs(z) ** 2 for z in vec)


print("КВАНТОВАЯ УНИТАРНОСТЬ КАЛЬКУЛЯТОРА (expm-Паде [6/6], H = лапласиан-графа 3x3)")
print("Теория: |Psi(t)> = expm(-i H t) |Psi0>,  |Psi(t)|^2 = 1 для всех t")
print("-" * 62)
worst = 0.0
for t in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
    raw = engine_calc("schrodinger(tridiag(0,-1,3), [1;0;0], %s)" % t)
    try:
        v = parse_cplx_vector(raw)
        n2 = norm2(v)
        dev = abs(n2 - 1.0)
        worst = max(worst, dev)
        print("  t=%5.1f : |Psi|^2 = %.8f   |отклонение| = %.2e" % (t, n2, dev))
    except Exception as e:
        print("  t=%5.1f : PARSE FAIL (%s) raw=%r" % (t, e, raw[:60]))
print("-" * 62)
print("ХУДШЕЕ ОТКЛОНЕНИЕ УНИТАРНОСТИ: %.2e  (ошибка Паде [6/6])" % worst)
