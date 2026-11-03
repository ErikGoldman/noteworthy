#pragma once

#include <compare>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace omr {

// Exact non-negative-denominator fraction, used for musical time (in quarter notes).
class Rational {
 public:
  constexpr Rational(int64_t num = 0, int64_t den = 1) : num_(num), den_(den) {
    if (den_ == 0) throw std::invalid_argument("Rational with zero denominator");
    normalize();
  }

  constexpr int64_t num() const { return num_; }
  constexpr int64_t den() const { return den_; }
  constexpr double to_double() const { return static_cast<double>(num_) / static_cast<double>(den_); }

  constexpr Rational operator+(const Rational& o) const { return {num_ * o.den_ + o.num_ * den_, den_ * o.den_}; }
  constexpr Rational operator-(const Rational& o) const { return {num_ * o.den_ - o.num_ * den_, den_ * o.den_}; }
  constexpr Rational operator*(const Rational& o) const { return {num_ * o.num_, den_ * o.den_}; }
  constexpr Rational operator/(const Rational& o) const { return {num_ * o.den_, den_ * o.num_}; }
  constexpr Rational& operator+=(const Rational& o) { return *this = *this + o; }
  constexpr Rational& operator-=(const Rational& o) { return *this = *this - o; }

  constexpr bool operator==(const Rational& o) const { return num_ == o.num_ && den_ == o.den_; }
  constexpr std::strong_ordering operator<=>(const Rational& o) const { return num_ * o.den_ <=> o.num_ * den_; }

 private:
  constexpr void normalize() {
    if (den_ < 0) {
      num_ = -num_;
      den_ = -den_;
    }
    const int64_t g = std::gcd(num_, den_);
    if (g > 1) {
      num_ /= g;
      den_ /= g;
    }
  }

  int64_t num_;
  int64_t den_;
};

inline std::ostream& operator<<(std::ostream& os, const Rational& r) {
  return r.den() == 1 ? os << r.num() : os << r.num() << '/' << r.den();
}

}  // namespace omr
