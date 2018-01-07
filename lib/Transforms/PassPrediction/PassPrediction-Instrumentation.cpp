#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h>

namespace PassPrediction {
  namespace {
    std::string getDemangledFunctionName(std::string mangledName) {
			int status = -99;
			const char *OrigName = abi::__cxa_demangle(mangledName.c_str(), NULL, NULL, &status);
			if (OrigName == NULL) {
					OrigName = mangledName.c_str();
			}
			return std::string(OrigName);
    }
  }
  void PassPeeper(const std::string& file, unsigned FeatureId) {
    // Get the feature object, which is a singleton
    FeatureRecorder &InstrumentRec = FeatureRecorder::getInstance();
    // if instrumentation is enabled, record it.
    if (InstrumentRec.isInstrumentationActivated()){
      std::string demangledFuncName = getDemangledFunctionName(InstrumentRec.getCurrFuncName());
      // Do not extract features from "std::"
      if (demangledFuncName.find(std::string("std::")) != 0) {
        InstrumentRec.encounterFeatures(FeatureId);
      }
    }  
  }
}
