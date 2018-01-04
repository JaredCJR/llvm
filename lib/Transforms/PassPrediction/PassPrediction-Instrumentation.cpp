#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace PassPrediction {
  void PassPeeper(const std::string& file, unsigned InsertId) {
    FeatureRecorder &InstrumentRec = FeatureRecorder::getInstance();
    if (InstrumentRec.isInstrumentationActivated())
      llvm::errs() << InstrumentRec.getCurrFuncName() << " " << file << " - " << InsertId <<"\n";
    else {
      // Do nothing
    }
  }
}
