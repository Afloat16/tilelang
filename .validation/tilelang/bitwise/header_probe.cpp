#include "common.h"
#include <cassert>
#include <limits>
#include <iostream>
#include <type_traits>

template <typename T, int N> void integer_checks() {
  using U = std::make_unsigned_t<T>;
  constexpr int bits = sizeof(T) * 8;
  const U patterns[] = {U(0), U(1), std::numeric_limits<U>::max(),
                        U(U(1) << (bits - 1)),
                        U(std::numeric_limits<U>::max() / 3)};
  for (int offset = 0; offset < 5; ++offset) {
    vec_type<T, N> a, b, shift;
    for (int i = 0; i < N; ++i) {
      a.data[i] = static_cast<T>(patterns[(i + offset) % 5]);
      b.data[i] = static_cast<T>(patterns[(i + offset + 2) % 5]);
      shift.data[i] = static_cast<T>(i % 2 ? bits - 1 : 0);
    }
    const auto both = a & b;
    const auto either = a | b;
    const auto exclusive = a ^ b;
    const auto inverted = ~a;
    const auto left = a << shift;
    const auto right = a >> shift;
    const auto broadcast = a ^ vec_type<T, N>(b.data[0]);
    for (int i = 0; i < N; ++i) {
      const U ua = static_cast<U>(a.data[i]);
      const U ub = static_cast<U>(b.data[i]);
      assert(static_cast<U>(both.data[i]) == static_cast<U>(ua & ub));
      assert(static_cast<U>(either.data[i]) == static_cast<U>(ua | ub));
      assert(static_cast<U>(exclusive.data[i]) == static_cast<U>(ua ^ ub));
      assert(static_cast<U>(inverted.data[i]) == static_cast<U>(~ua));
      assert(static_cast<U>(left.data[i]) == static_cast<U>(ua << shift.data[i]));
      assert(right.data[i] == static_cast<T>(a.data[i] >> shift.data[i]));
      assert(broadcast.data[i] == static_cast<T>(a.data[i] ^ b.data[0]));
    }
  }
}

template <typename T> void widths() {
  integer_checks<T, 2>();
  integer_checks<T, 4>();
  integer_checks<T, 8>();
  integer_checks<T, 16>();
}

template <typename T> void arithmetic_control() {
  vec_type<T, 4> a(T(6)), b(T(2));
  const auto add = a + b;
  const auto sub = a - b;
  const auto mul = a * b;
  const auto div = a / b;
  assert(add.data[0] == T(8));
  assert(sub.data[0] == T(4));
  assert(mul.data[0] == T(12));
  assert(div.data[0] == T(3));
}

int main() {
  widths<int8_t>(); widths<uint8_t>();
  widths<int16_t>(); widths<uint16_t>();
  widths<int32_t>(); widths<uint32_t>();
  widths<int64_t>(); widths<uint64_t>();
  arithmetic_control<float>();
  arithmetic_control<double>();
  arithmetic_control<half>();
  std::cout << "32 integer dtype/width combinations and float/double/half controls passed.\n";
}

