#pragma once

struct Grad {
  double left, right;
};

inline Grad operator+(const Grad& a, const Grad& b) {
  return Grad{a.left + b.left, a.right + b.right};
}

inline void operator+=(Grad& a, const Grad& b) {
  a.left += b.left;
  a.right += b.right;
}

inline Grad operator*(const Grad& a, double w) {
  return Grad{a.left * w, a.right * w};
}

inline double NegDir(const Grad& g) {
  if (g.left < 0 && g.right > 0) {
    return 0;
  } else if (g.left >= 0) {
    return -g.left;
  } else {
    return -g.right;
  }
}

inline void GradDesc(double &param, const Grad& g) {
  if (g.left < 0 && g.right > 0) {
    return;
  } else if (g.left >= 0) {
    param -= g.left;
  } else {
    param -= g.right;
  }
}
