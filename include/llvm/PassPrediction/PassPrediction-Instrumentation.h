#ifndef LLVM_PassPrediction_Instrumentation_H
#define LLVM_PassPrediction_Instrumentation_H
#include <string>

namespace PassPrediction {
  void PassPeeper(const std::string& file, int line);
  class FeatureRecorder {
    public:
      // singleton
      static FeatureRecorder &getInstance() {
        static FeatureRecorder instance;
        return instance;
      }
      void setCurrFuncName(std::string Name) {
        CurrFuncName = Name;
      }
      std::string getCurrFuncName() {
        return CurrFuncName;
      }
      bool isInstrumentationActivated() {
        return ActivateInstrumentation;
      }
      void EnableInstrumentation() {
        ActivateInstrumentation = true;
      }
      void DisableInstrumentation() {
        ActivateInstrumentation = false;
      }
    private:
      bool ActivateInstrumentation;
      std::string CurrFuncName;
      FeatureRecorder() {};// Prevent construction
  };
}


#endif
