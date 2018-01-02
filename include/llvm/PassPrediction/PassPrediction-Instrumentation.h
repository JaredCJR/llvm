#ifndef LLVM_PassPrediction_Instrumentation_H
#define LLVM_PassPrediction_Instrumentation_H
#include <string>

namespace PassPrediction {
  void PassPeeper(const std::string& file, int line);
}

#endif
