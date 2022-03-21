#pragma once

#include<bits/stdc++.h>
#include "./Eigen3/Dense"
#include <random>

#define Vec std::vector
#define Pair std::pair
#define RNG std::default_random_engine
#define UniqPtr std::unique_ptr

/// The meta-data of composite feature
struct CpFeatureMeta {
  /// The value cap of each categorical attribute
  Vec<int> enum_cap;
  /// The total number of numerical attribute
  int num_size;
};

struct CpFeature {
  Vec<int> enum_;
  Vec<double> num_;
};

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
