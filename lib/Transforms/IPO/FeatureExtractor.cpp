#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/FeatureExtractor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/LegacyPassManagers.h"
using namespace llvm;

#define DEBUG_TYPE "FeatureExtractor(DEBUG_TYPE)"

STATISTIC(FeatureExtractorCounter, "Counts number of functions greeted");

namespace {
  struct FeatureExtractor : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    FeatureExtractor() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override;
  };
}

char FeatureExtractor::ID = 0;

/// (passName, arg, name, cfg, analysis)
INITIALIZE_PASS_BEGIN(FeatureExtractor, "Feature-Extractor",
                "PassPrediction Pass", false, false)
INITIALIZE_PASS_END(FeatureExtractor, "Feature-Extractor",
                "PassPrediction Pass", false, false)

FunctionPass *llvm::createFeatureExtractorPass() {
  return new FeatureExtractor();
}

bool FeatureExtractor::runOnFunction(Function &F) {
  ++FeatureExtractorCounter;
  // This pass is a useless and legacy pass for thesis
  // This will be removed when we are available =)
  return false;
}

static RegisterPass<FeatureExtractor> X("FeatureExtractor", "FeatureExtractor Pass");
