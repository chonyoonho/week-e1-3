#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MAC 기반 패턴 판정기 (Cross / X)

패턴(2차원 숫자 배열)이 'Cross(+)' 필터와 'X' 필터 중 어느 쪽과 더 닮았는지
MAC(Multiply-Accumulate: 같은 위치끼리 곱한 뒤 모두 더하기) 점수로 판정한다.
NumPy 같은 외부 라이브러리는 사용하지 않고 반복문으로 직접 구현한다.

실행:
    python main.py              # 같은 폴더의 data.json 사용
    python main.py other.json   # 다른 JSON 파일 사용

코드 구성 (위에서 아래로)
    1. 상수 / 설정
    2. 데이터 구조      : Matrix
    3. 핵심 로직        : mac, compare_scores, judge_scores, normalize_label
    4. 성능 측정        : measure_mac_time, print_performance_table
    5. 모드 1           : 3x3 사용자 입력
    6. 모드 2           : data.json 로드 / 스키마 검증 / 일괄 판정 / 요약
    7. 진입점           : main
"""

import json
import math
import os
import re
import sys
import time
import unicodedata

# ----------------------------------------------------------------------
# 1. 상수 / 설정
# ----------------------------------------------------------------------
EPSILON = 1e-9                  # 동점 판정 허용오차
REPEAT = 10                     # 성능 측정 반복 횟수 (최소 기준 10회)
PERF_SIZES = (3, 5, 13, 25)     # 성능 분석 대상 크기

LABEL_CROSS = "Cross"           # 내부 표준 라벨 2가지 + 판정 불가
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 라벨 정규화 표 (소문자로 바꾼 뒤 조회)
#   expected 값 : '+' -> Cross, 'x' -> X
#   filter 키   : 'cross' -> Cross, 'x' -> X
_LABEL_TABLE = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

# 실패 원인 분류 (README의 "데이터 vs 로직 vs 수치 비교" 진단용)
CAT_SCHEMA = "데이터/스키마"
CAT_LOGIC = "로직(판정 불일치)"
CAT_NUMERIC = "수치 비교(동점)"

DEFAULT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")


# ----------------------------------------------------------------------
# 2. 데이터 구조
# ----------------------------------------------------------------------
class Matrix:
    """n x n 정사각 2차원 배열. 특정 위치의 값을 저장(set)/조회(get)할 수 있다."""

    def __init__(self, n, fill=0.0):
        self.n = n
        self._rows = [[float(fill)] * n for _ in range(n)]

    @classmethod
    def from_rows(cls, rows):
        """검증이 끝난 정사각 리스트로 Matrix를 만든다."""
        n = len(rows)
        m = cls(n)
        for r in range(n):
            for c in range(n):
                m.set(r, c, rows[r][c])
        return m

    def get(self, r, c):
        return self._rows[r][c]

    def set(self, r, c, value):
        self._rows[r][c] = float(value)

    def row(self, r):
        return self._rows[r]


def make_cross(n):
    """성능 측정용 Cross(+) 모양 행렬 (가운데 행 + 가운데 열)."""
    m = Matrix(n)
    mid = n // 2
    for i in range(n):
        m.set(mid, i, 1)
        m.set(i, mid, 1)
    return m


def make_x(n):
    """성능 측정용 X 모양 행렬 (두 대각선)."""
    m = Matrix(n)
    for i in range(n):
        m.set(i, i, 1)
        m.set(i, n - 1 - i, 1)
    return m


# ----------------------------------------------------------------------
# 3. 핵심 로직
# ----------------------------------------------------------------------
def mac(pattern, filt):
    """MAC: 같은 위치의 값끼리 곱하고 전부 더한 점수(float)를 반환한다.

    시간 복잡도 O(N^2): 이중 반복문이 N*N번 돈다.
    """
    if pattern.n != filt.n:
        raise ValueError(
            "크기 불일치: 패턴 {0}×{0} vs 필터 {1}×{1}".format(pattern.n, filt.n)
        )
    n = pattern.n
    total = 0.0
    for r in range(n):
        prow = pattern.row(r)
        frow = filt.row(r)
        for c in range(n):
            total += prow[c] * frow[c]
    return total


def compare_scores(a, b):
    """허용오차(EPSILON) 기반 비교. a>b: 1, a<b: -1, 동점: 0.

    부동소수점은 0.1 + 0.2 != 0.3 처럼 미세한 오차가 생기므로
    '==' 대신 |a-b| < EPSILON 이면 같은 값으로 본다.
    """
    if abs(a - b) < EPSILON:
        return 0
    return 1 if a > b else -1


def judge_scores(score_cross, score_x):
    """Cross 점수와 X 점수로 표준 라벨(Cross / X / UNDECIDED)을 판정한다."""
    result = compare_scores(score_cross, score_x)
    if result == 0:
        return LABEL_UNDECIDED
    return LABEL_CROSS if result > 0 else LABEL_X


def normalize_label(raw):
    """다양한 표기를 표준 라벨(Cross / X)로 바꾼다. 알 수 없으면 None."""
    if not isinstance(raw, str):
        return None
    return _LABEL_TABLE.get(raw.strip().lower())


# ----------------------------------------------------------------------
# 4. 성능 측정
# ----------------------------------------------------------------------
def measure_mac_time(pattern, filt, repeat=REPEAT):
    """MAC 연산만 repeat번 측정해 평균 시간(ms)을 반환한다. (I/O 시간 제외)"""
    mac(pattern, filt)  # 워밍업 1회 (측정 제외)
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        mac(pattern, filt)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


def _display_width(text):
    """한글처럼 폭이 2칸인 문자를 고려한 화면 폭."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text, width):
    return text + " " * max(0, width - _display_width(text))


def print_performance_table(rows):
    """rows: [(N, 평균시간ms), ...] 를 표로 출력한다."""
    print("\n[성능 분석] MAC 연산 시간 (각 크기 {0}회 반복 평균)".format(REPEAT))
    header = (_pad("크기(N×N)", 12), _pad("평균 시간(ms)", 16), "연산 횟수(N²)")
    print("  " + "".join(header))
    print("  " + "-" * 44)
    for n, avg_ms in rows:
        print("  " + _pad("{0}×{0}".format(n), 12)
              + _pad("{0:.6f}".format(avg_ms), 16)
              + str(n * n))
    if len(rows) >= 2 and rows[0][1] > 0:
        (n0, t0), (n1, t1) = rows[0], rows[-1]
        print("  -> {0}×{0} 대비 {1}×{1}: 연산 횟수 {2:.1f}배, 측정 시간 {3:.1f}배"
              .format(n0, n1, (n1 * n1) / (n0 * n0), t1 / t0))


# ----------------------------------------------------------------------
# 5. 모드 1 : 사용자 입력 (3x3)
# ----------------------------------------------------------------------
def parse_row(line, n):
    """'1 0 1' 같은 한 줄을 n개의 float로 변환. 형식이 틀리면 ValueError."""
    tokens = line.split()
    if len(tokens) != n:
        raise ValueError("열 수 불일치")
    values = []
    for token in tokens:
        value = float(token)            # 숫자 파싱 실패 시 ValueError
        if not math.isfinite(value):    # nan / inf 거부
            raise ValueError("유한한 숫자가 아님")
        values.append(value)
    return values


def read_matrix(title, n=3):
    """콘솔에서 n줄을 입력받아 Matrix로 저장. 잘못 입력하면 그 줄을 다시 입력받는다."""
    print("\n{0} 입력 ({1}줄, 한 줄에 {1}개의 숫자를 공백으로 구분)".format(title, n))
    rows = []
    while len(rows) < n:
        line = input("  {0}행> ".format(len(rows) + 1))
        try:
            rows.append(parse_row(line, n))
        except ValueError:
            print("입력 형식 오류: 각 줄에 {0}개의 숫자를 공백으로 구분해 입력하세요.".format(n))
    return Matrix.from_rows(rows)


def print_matrix(title, m):
    print("  {0}:".format(title))
    for r in range(m.n):
        print("    " + " ".join("{0:g}".format(m.get(r, c)) for c in range(m.n)))


def run_user_mode():
    print("\n[모드 1] 사용자 입력 (3×3)")
    filter_a = read_matrix("필터 A")
    filter_b = read_matrix("필터 B")
    print("\n저장 확인 - 필터가 저장되었습니다.")
    print_matrix("필터 A", filter_a)
    print_matrix("필터 B", filter_b)

    pattern = read_matrix("패턴")
    print("\n저장 확인 - 패턴이 저장되었습니다.")
    print_matrix("패턴", pattern)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    time_a = measure_mac_time(pattern, filter_a)
    time_b = measure_mac_time(pattern, filter_b)

    result = compare_scores(score_a, score_b)
    verdict = "판정 불가" if result == 0 else ("A" if result > 0 else "B")

    print("\n[결과]")
    print("  필터 A 점수: {0:.6f}  (연산 시간 {1:.6f} ms)".format(score_a, time_a))
    print("  필터 B 점수: {0:.6f}  (연산 시간 {1:.6f} ms)".format(score_b, time_b))
    print("  판정: {0}".format(verdict))
    if result == 0:
        print("  (두 점수 차이가 {0} 미만이라 동점으로 처리했습니다.)".format(EPSILON))

    print_performance_table([(3, measure_mac_time(pattern, filter_a))])


# ----------------------------------------------------------------------
# 6. 모드 2 : data.json 분석
# ----------------------------------------------------------------------
def load_json_file(path):
    """JSON 파일을 읽는다. 실패하면 사람이 읽을 수 있는 메시지의 ValueError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError("파일을 찾을 수 없습니다: {0}".format(path))
    except json.JSONDecodeError as e:
        raise ValueError("JSON 형식 오류 ({0}): {1}".format(path, e))
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError("파일을 읽을 수 없습니다 ({0}): {1}".format(path, e))


def validate_matrix(raw, name):
    """JSON의 2차원 배열을 검사해 Matrix로 변환. 문제가 있으면 ValueError."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("{0}: 비어 있거나 2차원 배열이 아님".format(name))
    n = len(raw)
    for i, row in enumerate(raw):
        if not isinstance(row, list):
            raise ValueError("{0}: {1}행이 배열이 아님".format(name, i))
        if len(row) != n:
            raise ValueError("{0}: 정사각형이 아님 (행 {1}개, {2}행의 열 {3}개)"
                             .format(name, n, i, len(row)))
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("{0}: [{1}][{2}] 값 {3!r} 은(는) 숫자가 아님".format(name, i, j, v))
    return Matrix.from_rows(raw)


def load_filters(data):
    """filters 를 읽어 크기별 {Cross, X} 필터로 정리한다.

    반환: (filters, filter_errors, warnings)
        filters       {N: {"Cross": Matrix, "X": Matrix}}
        filter_errors {N: 오류 메시지}   - 해당 크기 케이스는 FAIL 처리됨
        warnings      키 형식이 달라 무시한 항목 설명
    """
    raw_filters = data.get("filters")
    if not isinstance(raw_filters, dict):
        raise ValueError("스키마 오류: 최상위에 'filters' 객체가 없습니다.")

    filters, filter_errors, warnings = {}, {}, []
    for key, group in raw_filters.items():
        m = re.fullmatch(r"size_(\d+)", str(key))
        if not m:
            warnings.append("filters 키 '{0}' 는 size_N 형태가 아니라 무시합니다.".format(key))
            continue
        n = int(m.group(1))
        try:
            if not isinstance(group, dict):
                raise ValueError("필터 묶음이 객체가 아님")
            normalized = {}
            for raw_label, raw_matrix in group.items():
                label = normalize_label(raw_label)         # 'cross' -> Cross, 'x' -> X
                if label is None:
                    raise ValueError("필터 키 '{0}' 를 Cross/X 로 정규화할 수 없음".format(raw_label))
                if label in normalized:
                    raise ValueError("필터 키가 정규화 후 중복됨: {0}".format(label))
                matrix = validate_matrix(raw_matrix, "필터 {0}".format(raw_label))
                if matrix.n != n:
                    raise ValueError("필터 {0} 크기가 {1}×{1} 이 아님".format(raw_label, n))
                normalized[label] = matrix
            if set(normalized) != {LABEL_CROSS, LABEL_X}:
                raise ValueError("Cross 와 X 필터가 모두 필요함 (현재: {0})"
                                 .format(sorted(normalized) or "없음"))
            filters[n] = normalized
        except ValueError as e:
            filter_errors[n] = str(e)
    return filters, filter_errors, warnings


def evaluate_case(key, item, filters, filter_errors):
    """패턴 1건을 판정해 결과 dict 를 돌려준다. 예외를 밖으로 던지지 않는다."""
    result = {
        "key": key, "size": None,
        "score_cross": None, "score_x": None,
        "verdict": None, "expected": None,
        "passed": False, "category": None, "reason": "",
    }

    def fail(category, reason):
        result["passed"] = False
        result["category"] = category
        result["reason"] = reason
        return result

    # (1) 키에서 N 추출
    m = re.fullmatch(r"size_(\d+)_(\d+)", str(key))
    if not m:
        return fail(CAT_SCHEMA, "키 형식 오류: size_{N}_{idx} 형태가 아님")
    n = int(m.group(1))
    result["size"] = n

    # (2) 항목 구조 / 라벨 정규화
    if not isinstance(item, dict) or "input" not in item or "expected" not in item:
        return fail(CAT_SCHEMA, "스키마 오류: 'input' 과 'expected' 가 필요함")
    expected = normalize_label(item["expected"])
    if expected is None:
        return fail(CAT_SCHEMA, "라벨 오류: expected {0!r} 를 Cross/X 로 정규화할 수 없음"
                    .format(item["expected"]))
    result["expected"] = expected

    # (3) 패턴 검증
    try:
        pattern = validate_matrix(item["input"], "input")
    except ValueError as e:
        return fail(CAT_SCHEMA, "스키마 오류: {0}".format(e))

    # (4) 필터 선택 + 크기 검증
    if n in filter_errors:
        return fail(CAT_SCHEMA, "필터 오류(size_{0}): {1}".format(n, filter_errors[n]))
    if n not in filters:
        return fail(CAT_SCHEMA, "필터 없음: filters 에 size_{0} 이(가) 없음".format(n))
    fset = filters[n]
    if pattern.n != fset[LABEL_CROSS].n:
        return fail(CAT_SCHEMA, "크기 불일치: 패턴 {0}×{0} vs 필터 {1}×{1}"
                    .format(pattern.n, fset[LABEL_CROSS].n))

    # (5) MAC + 판정
    result["score_cross"] = mac(pattern, fset[LABEL_CROSS])
    result["score_x"] = mac(pattern, fset[LABEL_X])
    verdict = judge_scores(result["score_cross"], result["score_x"])
    result["verdict"] = verdict

    if verdict == expected:
        result["passed"] = True
        return result
    if verdict == LABEL_UNDECIDED:
        return fail(CAT_NUMERIC, "동점(|차이| < {0})이라 판정 불가, expected={1}"
                    .format(EPSILON, expected))
    return fail(CAT_LOGIC, "판정 불일치: expected={0}, 판정={1}".format(expected, verdict))


def _case_sort_key(key):
    """size_5_2 < size_5_10 < size_13_1 순서로 정렬. 이상한 키는 맨 뒤."""
    m = re.fullmatch(r"size_(\d+)_(\d+)", str(key))
    if m:
        return (0, int(m.group(1)), int(m.group(2)), str(key))
    return (1, 0, 0, str(key))


def print_case(res):
    status = "PASS" if res["passed"] else "FAIL"
    if res["score_cross"] is None:
        print("[{0}] {1} | {2}".format(status, res["key"], res["reason"]))
        return
    line = "[{0}] {1} | Cross={2:.6f} X={3:.6f} | 판정={4} | expected={5}".format(
        status, res["key"], res["score_cross"], res["score_x"], res["verdict"], res["expected"])
    if not res["passed"]:
        line += " | " + res["reason"]
    print(line)


def print_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = [r for r in results if not r["passed"]]

    print("\n[결과 요약]")
    print("  전체 테스트 수: {0}".format(total))
    print("  통과: {0}".format(passed))
    print("  실패: {0}".format(len(failed)))
    if not failed:
        print("  실패 케이스 없음")
        return
    by_cat = {}
    for r in failed:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print("  실패 유형별: " + ", ".join("{0} {1}건".format(c, k) for c, k in by_cat.items()))
    print("  실패 케이스 목록:")
    for r in failed:
        print("    - {0}: {1}".format(r["key"], r["reason"]))


def run_json_mode(path):
    print("\n[모드 2] data.json 분석  ({0})".format(path))
    try:
        data = load_json_file(path)
        if not isinstance(data, dict):
            raise ValueError("스키마 오류: 최상위가 객체가 아님")
        filters, filter_errors, warnings = load_filters(data)
        patterns = data.get("patterns")
        if not isinstance(patterns, dict):
            raise ValueError("스키마 오류: 최상위에 'patterns' 객체가 없습니다.")
    except ValueError as e:
        print("데이터 로드 실패: {0}".format(e))
        return

    loaded = ", ".join("size_{0}".format(n) for n in sorted(filters)) or "없음"
    print("필터 로드: {0}".format(loaded))
    for n in sorted(filter_errors):
        print("  [경고] size_{0} 필터 오류: {1}".format(n, filter_errors[n]))
    for w in warnings:
        print("  [경고] {0}".format(w))

    print("\n[케이스별 판정]  (expected 와 비교)")
    results = []
    for key in sorted(patterns, key=_case_sort_key):
        res = evaluate_case(key, patterns[key], filters, filter_errors)
        results.append(res)
        print_case(res)

    rows = [(n, measure_mac_time(make_cross(n), make_x(n))) for n in PERF_SIZES]
    print_performance_table(rows)
    print_summary(results)


# ----------------------------------------------------------------------
# 7. 진입점
# ----------------------------------------------------------------------
def choose_mode():
    while True:
        print("입력 방식을 선택하세요.")
        print("  1. 사용자 입력 (3×3)")
        print("  2. data.json 분석")
        choice = input("선택> ").strip()
        if choice in ("1", "2"):
            return choice
        print("잘못된 선택입니다. 1 또는 2를 입력하세요.\n")


def main(argv):
    json_path = argv[1] if len(argv) > 1 else DEFAULT_JSON
    try:
        if choose_mode() == "1":
            run_user_mode()
        else:
            run_json_mode(json_path)
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 중단되어 프로그램을 종료합니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
