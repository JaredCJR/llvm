#ifndef LLVM_PassPrediction_Instrumentation_H
#define LLVM_PassPrediction_Instrumentation_H
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace PassPrediction {
  void PassPeeper(const std::string& file, unsigned FeatureId);
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

      unsigned getFeatureSize() {
        return FeatureSize;
      }

      void encounterFeatures(unsigned FeatureId) {
        if (FeatureMap.find(CurrFuncName) == FeatureMap.end()) {
          // Create feature vector for this function
          FeatureMap[CurrFuncName] = std::vector<unsigned short>(FeatureSize, 0);
        }
        FeatureMap[CurrFuncName][FeatureId] = FeatureMap[CurrFuncName][FeatureId] + 1;
      }

      void printFeatures() {
        for (auto it : FeatureMap) {
          llvm::errs() << it.first << ": ";
          for (auto feature : it.second) {
            llvm::errs() << feature << " ";
          }
          llvm::errs() << "\n\n";
        }
      }

    private:
      bool ActivateInstrumentation;
      unsigned FeatureSize = 0;
      std::string CurrFuncName;
      FeatureRecorder() { // Prevent construction
        ActivateInstrumentation = false;
        // Read the size of features, which is generate from "InstrumentTidiedPasses.sh"
        char *InstrumentPath = getenv("LLVM_THESIS_InstrumentHome");
        if (!InstrumentPath) {
          std::cerr << "$LLVM_THESIS_InstrumentHome is missing.\n";
          exit(EXIT_FAILURE);
        }
        std::string FilePath = std::string(InstrumentPath) + std::string("/Database/FeatureSize");
        std::ifstream fin;
        fin.open(FilePath);
        if(!fin.is_open()) {
          std::cerr << FilePath  << " is missing.\n";
          exit(EXIT_FAILURE);
        }
        fin >> FeatureSize;
        fin.close();
      };
      std::unordered_map<std::string, std::vector<unsigned short> > FeatureMap;
  };
}


#endif
