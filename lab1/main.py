import itertools
import json
import math
import sys
from pathlib import Path

EPS = 1e-7


def _flatten(node):
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return (), [float(node)]
    if not isinstance(node, list) or not node:
        raise ValueError("Некорректный вложенный список.")
    children = [_flatten(c) for c in node]
    if len({s for s, _ in children}) != 1:
        raise ValueError("Несогласованная размерность (рваный массив).")
    return (len(node),) + children[0][0], [x for _, f in children for x in f]


def read_ensemble(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
        lines = [line for line in lines if line]
    if not lines:
        raise ValueError("Файл пуст.")

    text = "\n".join(lines)
    if "[" in text or "]" in text:
        shape, probs = _flatten(json.loads(text))
    elif lines[0].lower().startswith("shape:"):
        shape = tuple(int(x) for x in lines[0].split(":", 1)[1].split())
        probs = [float(x) for line in lines[1:] for x in line.split()]
    else:
        rows = [[float(x) for x in line.split()] for line in lines]
        if len({len(r) for r in rows}) != 1:
            raise ValueError("Строки матрицы имеют разную длину.")
        shape = (len(rows), len(rows[0]))
        probs = [x for r in rows for x in r]

    expected = math.prod(shape)
    if len(shape) < 2 or any(d <= 0 for d in shape) or len(probs) != expected:
        raise ValueError(f"Некорректные размеры {shape} или число элементов ({len(probs)} != {expected}).")
    if any(p < -EPS for p in probs):
        raise ValueError("Вероятности не могут быть отрицательными.")
    probs = [0.0 if abs(p) <= EPS else p for p in probs]
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"Сумма вероятностей = {sum(probs):.8f}, должна быть 1.")
    return shape, probs


def entropy(p_list):
    h = -sum(p * math.log2(p) for p in p_list if p > 0.0)
    return 0.0 if abs(h) < 1e-15 else h


def get_marginal(probs, shape, keep_axes):
    res = [0.0] * math.prod(shape[a] for a in keep_axes)
    for idx, p in enumerate(probs):
        if p == 0.0:
            continue
        rem = idx
        coords = [0] * len(shape)
        for i in range(len(shape) - 1, -1, -1):
            coords[i] = rem % shape[i]
            rem //= shape[i]
        sub_idx = 0
        for a in keep_axes:
            sub_idx = sub_idx * shape[a] + coords[a]
        res[sub_idx] += p
    return res


def solve_two_events(shape, probs):
    rows, cols = shape
    px, py = get_marginal(probs, shape, (0,)), get_marginal(probs, shape, (1,))
    hx, hy, hxy = entropy(px), entropy(py), entropy(probs)

    print(f"Размерность: {rows}x{cols}")
    print(f"H(X) = {hx:.6f} бит, H(Y) = {hy:.6f} бит, H(X,Y) = {hxy:.6f} бит\n")

    print("Частные условные энтропии:")
    hy_x = 0.0
    for i in range(rows):
        h_part = entropy([probs[i * cols + j] / px[i] for j in range(cols)]) if px[i] > 0 else 0.0
        hy_x += px[i] * h_part
        print(f"  H(Y | x{i + 1}) = {h_part:.6f} бит")

    hx_y = 0.0
    for j in range(cols):
        h_part = entropy([probs[i * cols + j] / py[j] for i in range(rows)]) if py[j] > 0 else 0.0
        hx_y += py[j] * h_part
        print(f"  H(X | y{j + 1}) = {h_part:.6f} бит")

    print(f"\nH(Y | X) = {hy_x:.6f} бит\nH(X | Y) = {hx_y:.6f} бит\nI(X, Y)  = {max(0.0, hx + hy - hxy):.6f} бит")
    indep = all(abs(probs[i * cols + j] - px[i] * py[j]) <= EPS for i in range(rows) for j in range(cols))
    print("Независимость:", "ДА" if indep else "НЕТ")


def solve_n_events(shape, probs):
    n = len(shape)
    margs = [get_marginal(probs, shape, (i,)) for i in range(n)]
    h_single = [entropy(m) for m in margs]
    h_joint = entropy(probs)

    for i in range(n):
        print(f"H(X{i + 1}) = {h_single[i]:.6f} бит")
    print(f"H(X1..X{n}) = {h_joint:.6f} бит\n")

    print("Цепное правило:")
    print(f"  H(X1) = {h_single[0]:.6f} бит")
    for k in range(1, n):
        h_prev = entropy(get_marginal(probs, shape, tuple(range(k))))
        h_curr = entropy(get_marginal(probs, shape, tuple(range(k + 1))))
        given = "X1" if k == 1 else f"X1..X{k}"
        print(f"  H(X{k + 1} | {given}) = {max(0.0, h_curr - h_prev):.6f} бит")

    print("\nПри известных остальных:")
    for k in range(n):
        h_others = entropy(get_marginal(probs, shape, tuple(i for i in range(n) if i != k)))
        print(f"  H(X{k + 1} | остальные) = {max(0.0, h_joint - h_others):.6f} бит")

    print(f"\nОбщая взаимная информация I(X1;..;X{n}) = {max(0.0, sum(h_single) - h_joint):.6f} бит")

    # Проверка независимости в совокупности и попарно
    mut_indep = all(
        abs(p - math.prod(margs[a][(idx // math.prod(shape[a + 1:])) % shape[a]] for a in range(n))) <= EPS
        for idx, p in enumerate(probs)
    )
    dep_pairs = []
    for i, j in itertools.combinations(range(n), 2):
        p_ij = get_marginal(probs, shape, (i, j))
        if any(abs(p_ij[a * shape[j] + b] - margs[i][a] * margs[j][b]) > EPS for a in range(shape[i]) for b in range(shape[j])):
            dep_pairs.append(f"(X{i + 1}, X{j + 1})")

    print("Попарная независимость:      ", "ДА" if not dep_pairs else f"НЕТ ({', '.join(dep_pairs)})")
    print("Независимость в совокупности:", "ДА" if mut_indep else "НЕТ")


def main():
    if len(sys.argv) > 1:
        filename = Path(sys.argv[1])
    else:
        raw = input("Введите имя файла (Enter — example_2d_dependent.txt): ").strip()
        name = raw or "example_2d_dependent.txt"
        filename = Path(name)
        if not filename.exists():
            filename = Path(__file__).parent / name

    shape, probs = read_ensemble(filename)
    print(f"Файл: {filename.name}\n" + "-" * 40)
    solve_two_events(shape, probs) if len(shape) == 2 else solve_n_events(shape, probs)


if __name__ == "__main__":
    main()
