#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace PassPrediction {
  void PassPeeper(const std::string& file, unsigned FeatureId) {
    // Get the feature object, which is a singleton
    FeatureRecorder &InstrumentRec = FeatureRecorder::getInstance();
    // if instrumentation is enabled, record it.
    if (InstrumentRec.isInstrumentationActivated()){
      InstrumentRec.encounterFeatures(FeatureId);
    }  
  }
}
