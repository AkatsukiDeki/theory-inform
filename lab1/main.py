import math
import sys
from pathlib import Path

EPS = 1e-7


def read_ensemble(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
        lines = [line for line in lines if line]

    if not lines:
        raise ValueError("Файл пуст.")

    # Если задана форма (для N событий): первая строка "shape: 2 2 2"
    if lines[0].lower().startswith("shape:"):
        shape = tuple(int(x) for x in lines[0].split(":", 1)[1].split())
        probs = [float(x) for line in lines[1:] for x in line.split()]
    else:
        # Обычная 2D-матрица
        rows = [[float(x) for x in line.split()] for line in lines]
        shape = (len(rows), len(rows[0]))
        probs = [x for row in rows for x in row]

    expected = 1
    for d in shape:
        expected *= d
    if len(probs) != expected:
        raise ValueError("Количество вероятностей не совпадает с размерами.")
    if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError("Вероятности должны быть >= 0 и в сумме давать 1.")

    return shape, probs


def entropy(p_list):
    h = -sum(p * math.log2(p) for p in p_list if p > 0)
    return 0.0 if h == 0 else h


def get_coords(idx, shape):
    coords = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        coords[i] = idx % shape[i]
        idx //= shape[i]
    return tuple(coords)


def get_marginal(probs, shape, keep_axes):
    sub_shape = tuple(shape[a] for a in keep_axes)
    size = 1
    for d in sub_shape:
        size *= d
    res = [0.0] * size

    for idx, p in enumerate(probs):
        if p == 0:
            continue
        coords = get_coords(idx, shape)
        sub_idx = 0
        for a in keep_axes:
            sub_idx = sub_idx * shape[a] + coords[a]
        res[sub_idx] += p

    return res


def solve_two_events(shape, probs):
    rows, cols = shape
    px = get_marginal(probs, shape, (0,))
    py = get_marginal(probs, shape, (1,))

    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(probs)

    hy_given_x = 0.0
    print(f"H(X)   = {hx:.6f} бит")
    print(f"H(Y)   = {hy:.6f} бит")
    print(f"H(X,Y) = {hxy:.6f} бит\n")

    print("Частные условные энтропии:")
    for i in range(rows):
        if px[i] > 0:
            cond = [probs[i * cols + j] / px[i] for j in range(cols)]
            h_part = entropy(cond)
            hy_given_x += px[i] * h_part
        else:
            h_part = 0.0
        print(f"  H(Y | x{i + 1}) = {h_part:.6f} бит")

    hx_given_y = 0.0
    for j in range(cols):
        if py[j] > 0:
            cond = [probs[i * cols + j] / py[j] for i in range(rows)]
            h_part = entropy(cond)
            hx_given_y += py[j] * h_part
        else:
            h_part = 0.0
        print(f"  H(X | y{j + 1}) = {h_part:.6f} бит")

    ixy = hx + hy - hxy
    print(f"\nH(Y | X) = {hy_given_x:.6f} бит")
    print(f"H(X | Y) = {hx_given_y:.6f} бит")
    print(f"I(X, Y)  = {max(0.0, ixy):.6f} бит")

    independent = True
    for i in range(rows):
        for j in range(cols):
            if abs(probs[i * cols + j] - px[i] * py[j]) > EPS:
                independent = False
                break

    print("\nНезависимость:", "ДА (события независимы)" if independent else "НЕТ (события зависимы)")


def solve_n_events(shape, probs):
    n = len(shape)
    marginals = [get_marginal(probs, shape, (i,)) for i in range(n)]
    h_single = [entropy(m) for m in marginals]
    h_joint = entropy(probs)

    for i in range(n):
        print(f"H(X{i + 1}) = {h_single[i]:.6f} бит")
    print(f"H(X1..X{n}) = {h_joint:.6f} бит\n")

    print("Условные энтропии (цепное правило):")
    print(f"  H(X1) = {h_single[0]:.6f} бит")
    for k in range(1, n):
        h_prefix = entropy(get_marginal(probs, shape, tuple(range(k))))
        h_with_k = entropy(get_marginal(probs, shape, tuple(range(k + 1))))
        given = "X1" if k == 1 else f"X1..X{k}"
        print(f"  H(X{k + 1} | {given}) = {max(0.0, h_with_k - h_prefix):.6f} бит")

    print("\nУсловные энтропии при известных остальных:")
    for k in range(n):
        others = tuple(i for i in range(n) if i != k)
        h_others = entropy(get_marginal(probs, shape, others))
        print(f"  H(X{k + 1} | остальные) = {max(0.0, h_joint - h_others):.6f} бит")

    total_info = max(0.0, sum(h_single) - h_joint)
    print(f"\nОбщая взаимная информация I(X1..X{n}) = {total_info:.6f} бит")

    independent = True
    for idx, p_joint in enumerate(probs):
        coords = get_coords(idx, shape)
        p_prod = 1.0
        for axis, c in enumerate(coords):
            p_prod *= marginals[axis][c]
        if abs(p_joint - p_prod) > EPS:
            independent = False
            break

    print("Независимость в совокупности:", "ДА" if independent else "НЕТ")


def main():
    if len(sys.argv) > 1:
        filename = Path(sys.argv[1])
    else:
        filename = Path(__file__).parent / "example_2d_dependent.txt"

    shape, probs = read_ensemble(filename)
    if len(shape) == 2:
        solve_two_events(shape, probs)
    else:
        solve_n_events(shape, probs)


if __name__ == "__main__":
    main()
