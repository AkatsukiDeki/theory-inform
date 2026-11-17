import itertools
import math
import sys
from pathlib import Path

EPS = 1e-7


def _parse_nested_brackets(text: str) -> tuple[tuple[int, ...], list[float]]:
    """Безопасный рекурсивный парсер вложенных списков вида [[[0.04, 0.06], ...]] без eval."""
    cleaned = text.replace(",", " ").replace("[", " [ ").replace("]", " ] ")
    tokens = cleaned.split()
    pos = 0

    def parse_node():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Неожиданный конец файла при разборе скобок.")
        tok = tokens[pos]
        if tok == "[":
            pos += 1
            items = []
            while pos < len(tokens) and tokens[pos] != "]":
                items.append(parse_node())
            if pos >= len(tokens) or tokens[pos] != "]":
                raise ValueError("Незакрытая квадратная скобка '['.")
            pos += 1
            if not items:
                raise ValueError("Обнаружен пустой список '[]'.")
            return items
        if tok == "]":
            raise ValueError("Лишняя закрывающая скобка ']'.")
        pos += 1
        return float(tok)

    tree = parse_node()
    if pos != len(tokens):
        raise ValueError("Лишние данные после закрытия корневого списка.")

    def validate_and_flatten(node) -> tuple[tuple[int, ...], list[float]]:
        if isinstance(node, float):
            return (), [node]
        sub_shapes = []
        flat: list[float] = []
        for child in node:
            s, f = validate_and_flatten(child)
            sub_shapes.append(s)
            flat.extend(f)
        if len(set(sub_shapes)) != 1:
            raise ValueError("Несогласованная размерность вложенных списков (рваный массив).")
        return (len(node),) + sub_shapes[0], flat

    shape, probs = validate_and_flatten(tree)
    if len(shape) < 2:
        raise ValueError("Ансамбль должен содержать минимум 2 события.")
    return shape, probs


def read_ensemble(filename: Path) -> tuple[tuple[int, ...], list[float]]:
    with open(filename, "r", encoding="utf-8") as f:
        raw_lines = [line.split("#", 1)[0].strip() for line in f]
        lines = [line for line in raw_lines if line]

    if not lines:
        raise ValueError("Файл пуст.")

    full_text = "\n".join(lines)

    # Вариант 1: вложенные списки со скобками
    if "[" in full_text or "]" in full_text:
        shape, probs = _parse_nested_brackets(full_text)
    # Вариант 2: явное указание размерности "shape: d1 d2 ... dn"
    elif lines[0].lower().startswith("shape:"):
        shape = tuple(int(x) for x in lines[0].split(":", 1)[1].split())
        if len(shape) < 2 or any(d <= 0 for d in shape):
            raise ValueError("Некорректный заголовок shape (нужно >= 2 положительных размерностей).")
        probs = [float(x) for line in lines[1:] for x in line.split()]
    # Вариант 3: классическая 2D-матрица
    else:
        rows = [[float(x) for x in line.split()] for line in lines]
        row_lengths = {len(r) for r in rows}
        if len(row_lengths) != 1:
            raise ValueError("Строки матрицы имеют разную длину.")
        shape = (len(rows), len(rows[0]))
        probs = [x for row in rows for x in row]

    expected = 1
    for d in shape:
        expected *= d

    if len(probs) != expected:
        raise ValueError(f"Ожидалось {expected} вероятностей для формы {shape}, получено {len(probs)}.")
    if any(p < -EPS for p in probs):
        raise ValueError("Вероятности не могут быть отрицательными.")
    probs = [0.0 if abs(p) <= EPS else p for p in probs]
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"Сумма вероятностей равна {sum(probs):.8f}, а должна быть равна 1.")

    return shape, probs


def entropy(p_list: list[float]) -> float:
    """Шенноновская энтропия H(Z) = -sum(p_i * log2(p_i)) в битах."""
    h = -sum(p * math.log2(p) for p in p_list if p > 0.0)
    return 0.0 if abs(h) < 1e-15 else h


def get_coords(idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        coords[i] = idx % shape[i]
        idx //= shape[i]
    return tuple(coords)


def get_marginal(probs: list[float], shape: tuple[int, ...], keep_axes: tuple[int, ...]) -> list[float]:
    """Суммирование тензора вероятностей по всем осям, кроме keep_axes."""
    sub_shape = tuple(shape[a] for a in keep_axes)
    size = 1
    for d in sub_shape:
        size *= d
    res = [0.0] * size

    for idx, p in enumerate(probs):
        if p == 0.0:
            continue
        coords = get_coords(idx, shape)
        sub_idx = 0
        for a in keep_axes:
            sub_idx = sub_idx * shape[a] + coords[a]
        res[sub_idx] += p

    return res


def solve_two_events(shape: tuple[int, int], probs: list[float]) -> None:
    rows, cols = shape
    px = get_marginal(probs, shape, (0,))
    py = get_marginal(probs, shape, (1,))

    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(probs)

    print(f"Размерность ансамбля: {rows}x{cols}")
    print(f"Маргинальные P(X): {[round(p, 6) for p in px]}")
    print(f"Маргинальные P(Y): {[round(p, 6) for p in py]}\n")

    print(f"H(X)   = {hx:.6f} бит")
    print(f"H(Y)   = {hy:.6f} бит")
    print(f"H(X,Y) = {hxy:.6f} бит\n")

    print("Частные условные энтропии:")
    hy_given_x = 0.0
    for i in range(rows):
        if px[i] > 0.0:
            cond = [probs[i * cols + j] / px[i] for j in range(cols)]
            h_part = entropy(cond)
            hy_given_x += px[i] * h_part
        else:
            h_part = 0.0
        print(f"  H(Y | x{i + 1}) = {h_part:.6f} бит")

    hx_given_y = 0.0
    for j in range(cols):
        if py[j] > 0.0:
            cond = [probs[i * cols + j] / py[j] for i in range(rows)]
            h_part = entropy(cond)
            hx_given_y += py[j] * h_part
        else:
            h_part = 0.0
        print(f"  H(X | y{j + 1}) = {h_part:.6f} бит")

    ixy = max(0.0, hx + hy - hxy)
    print("\nПолные условные энтропии и взаимная информация:")
    print(f"  H(Y | X) = {hy_given_x:.6f} бит")
    print(f"  H(X | Y) = {hx_given_y:.6f} бит")
    print(f"  I(X, Y)  = {ixy:.6f} бит")

    independent = all(
        abs(probs[i * cols + j] - px[i] * py[j]) <= EPS
        for i in range(rows)
        for j in range(cols)
    )
    print("\nНезависимость:", "ДА (P(x,y) = P(x)P(y), I(X,Y) = 0)" if independent else "НЕТ (события статистически зависимы)")


def solve_n_events(shape: tuple[int, ...], probs: list[float]) -> None:
    n = len(shape)
    marginals = [get_marginal(probs, shape, (i,)) for i in range(n)]
    h_single = [entropy(m) for m in marginals]
    h_joint = entropy(probs)

    print(f"Число событий: {n}, размерность тензора: {shape}\n")
    for i in range(n):
        print(f"H(X{i + 1}) = {h_single[i]:.6f} бит")
    print(f"H(X1..X{n}) = {h_joint:.6f} бит\n")

    print("Условные энтропии (цепное правило H(X1..Xn) = H(X1) + H(X2|X1) + ...):")
    print(f"  H(X1) = {h_single[0]:.6f} бит")
    for k in range(1, n):
        h_prefix = entropy(get_marginal(probs, shape, tuple(range(k))))
        h_with_k = entropy(get_marginal(probs, shape, tuple(range(k + 1))))
        given = "X1" if k == 1 else f"X1..X{k}"
        print(f"  H(X{k + 1} | {given}) = {max(0.0, h_with_k - h_prefix):.6f} бит")

    print("\nУсловные энтропии при известных всех остальных событиях:")
    for k in range(n):
        others = tuple(i for i in range(n) if i != k)
        h_others = entropy(get_marginal(probs, shape, others))
        print(f"  H(X{k + 1} | остальные) = {max(0.0, h_joint - h_others):.6f} бит")

    total_info = max(0.0, sum(h_single) - h_joint)
    print(f"\nОбщая взаимная информация (мультиинформация) I(X1;..;X{n}) = {total_info:.6f} бит")

    # Проверка независимости в совокупности: P(x1,...,xn) == P(x1)*...*P(xn)
    mutually_independent = True
    for idx, p_joint in enumerate(probs):
        coords = get_coords(idx, shape)
        p_prod = 1.0
        for axis, c in enumerate(coords):
            p_prod *= marginals[axis][c]
        if abs(p_joint - p_prod) > EPS:
            mutually_independent = False
            break

    # Проверка попарной независимости для всех пар (Xi, Xj)
    pairwise_independent = True
    dependent_pairs = []
    for i, j in itertools.combinations(range(n), 2):
        p_ij = get_marginal(probs, shape, (i, j))
        pair_ok = True
        for a in range(shape[i]):
            for b in range(shape[j]):
                if abs(p_ij[a * shape[j] + b] - marginals[i][a] * marginals[j][b]) > EPS:
                    pair_ok = False
                    break
            if not pair_ok:
                break
        if not pair_ok:
            pairwise_independent = False
            dependent_pairs.append(f"(X{i + 1}, X{j + 1})")

    print("\nРезультаты проверки независимости:")
    print("  Попарная независимость:      ", "ДА" if pairwise_independent else f"НЕТ (зависимы пары: {', '.join(dependent_pairs)})")
    print("  Независимость в совокупности:", "ДА" if mutually_independent else "НЕТ")


def main() -> None:
    if len(sys.argv) > 1:
        filename = Path(sys.argv[1])
    else:
        filename = Path(__file__).parent / "example_2d_dependent.txt"

    shape, probs = read_ensemble(filename)
    print(f"Файл: {filename.name}")
    print("-" * 50)
    if len(shape) == 2:
        solve_two_events(shape, probs)  # type: ignore[arg-type]
    else:
        solve_n_events(shape, probs)


if __name__ == "__main__":
    main()
