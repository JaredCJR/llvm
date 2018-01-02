#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
#include <iostream>

namespace PassPrediction {
  void PassPeeper(const std::string& file, int line) {
    std::cerr << file << " - " << line << "\n";
  }
}
