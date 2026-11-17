import sys
from decimal import Decimal, getcontext

# Абсолютные частоты букв русского алфавита с Википедии (НКРЯ),
# без 'ё': частоты 'е' (42691213) и 'ё' (184928) сложены в 'е'
COUNTS = {
    "о": 55414481,
    "е": 42691213 + 184928,
    "а": 40487008,
    "и": 37153142,
    "н": 33838881,
    "т": 31620970,
    "с": 27627040,
    "р": 23916825,
    "в": 22930719,
    "л": 22230174,
    "к": 17653469,
    "м": 16203060,
    "д": 15052118,
    "п": 14201572,
    "у": 13245712,
    "я": 10139085,
    "ы": 9595941,
    "ь": 8784613,
    "г": 8564640,
    "з": 8329904,
    "б": 8051767,
    "ч": 7300193,
    "й": 6106262,
    "х": 4904176,
    "ж": 4746916,
    "ш": 3678738,
    "ю": 3220715,
    "ц": 2438807,
    "щ": 1822476,
    "э": 1610107,
    "ф": 1335747,
    "ъ": 185452,
}

TOTAL = sum(COUNTS.values())
PROBS = {ch: cnt / TOTAL for ch, cnt in COUNTS.items()}


def shannon_fano(probs):
    items = sorted(probs.items(), key=lambda x: -x[1])
    codes = {ch: "" for ch, _ in items}

    def split(group):
        if len(group) <= 1:
            return
        total = sum(p for _, p in group)
        best_i = 1
        best_diff = total
        left_sum = 0.0
        for i in range(1, len(group)):
            left_sum += group[i - 1][1]
            diff = abs(left_sum - (total - left_sum))
            if diff < best_diff:
                best_diff = diff
                best_i = i

        for ch, _ in group[:best_i]:
            codes[ch] += "0"
        for ch, _ in group[best_i:]:
            codes[ch] += "1"

        split(group[:best_i])
        split(group[best_i:])

    split(items)
    return codes


def huffman(probs):
    nodes = [[p, ch] for ch, p in probs.items()]
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x[0])
        left = nodes.pop(0)
        right = nodes.pop(0)
        nodes.append([left[0] + right[0], [left[1], right[1]]])

    codes = {}

    def walk(tree, prefix):
        if isinstance(tree, str):
            codes[tree] = prefix
        else:
            walk(tree[0], prefix + "0")
            walk(tree[1], prefix + "1")

    walk(nodes[0][1], "")
    return codes


def arithmetic_encode(text, counts, total):
    getcontext().prec = max(100, len(text) * 4)

    intervals = {}
    cur = Decimal(0)
    for ch, cnt in counts.items():
        p = Decimal(cnt) / Decimal(total)
        intervals[ch] = (cur, cur + p)
        cur += p

    left = Decimal(0)
    right = Decimal(1)
    for ch in text:
        width = right - left
        low, high = intervals[ch]
        right = left + width * high
        left = left + width * low

    # Находим минимальное число бит m, при котором в [left, right) есть дробь k / 2^m
    m = 1
    denom = Decimal(2)
    while True:
        k = int((left * denom).to_integral_value(rounding="ROUND_CEILING"))
        if Decimal(k) / denom < right:
            return left, right, format(k, f"0{m}b")
        m += 1
        denom *= 2


def main():
    if len(sys.argv) > 1:
        raw_fio = " ".join(sys.argv[1:])
    else:
        raw_fio = input("Введите ФИО: ")

    text = "".join(raw_fio.split()).lower().replace("ё", "е")
    if not text or any(ch not in PROBS for ch in text):
        print("Ошибка: введите непустое ФИО из русских букв.")
        return

    sf_codes = shannon_fano(PROBS)
    huff_codes = huffman(PROBS)

    print("Перечень вероятностей букв и кодов:")
    print(f"{'Буква':<6} {'Вероятность':<12} {'Шеннон-Фано':<14} {'Хаффман'}")
    for ch, p in PROBS.items():
        print(f"{ch:<6} {p:<12.6f} {sf_codes[ch]:<14} {huff_codes[ch]}")

    bits_before = len(text) * 5
    print(f"\nСтрока: {text} ({len(text)} симв.)")
    print(f"Объём ДО кодирования (5 бит/симв.): {bits_before} бит\n")

    sf_encoded = "".join(sf_codes[ch] for ch in text)
    print(f"1. Шеннон-Фано: {sf_encoded}")
    print(f"   Объём ПОСЛЕ: {len(sf_encoded)} бит | Сжатие: {len(sf_encoded) / bits_before * 100:.2f}%\n")

    huff_encoded = "".join(huff_codes[ch] for ch in text)
    print(f"2. Хаффман:     {huff_encoded}")
    print(f"   Объём ПОСЛЕ: {len(huff_encoded)} бит | Сжатие: {len(huff_encoded) / bits_before * 100:.2f}%\n")

    left, right, arith_bits = arithmetic_encode(text, COUNTS, TOTAL)
    print(f"3. Арифметическое кодирование:")
    print(f"   Отрезок: [{left:.12e}, {right:.12e})")
    print(f"   Код: 0.{arith_bits}")
    print(f"   Объём ПОСЛЕ: {len(arith_bits)} бит | Сжатие: {len(arith_bits) / bits_before * 100:.2f}%")


if __name__ == "__main__":
    main()
