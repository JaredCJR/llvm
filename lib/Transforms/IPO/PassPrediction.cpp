#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassPrediction.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/LegacyPassManagers.h"
using namespace llvm;

#define DEBUG_TYPE "PassPrediction(DEBUG_TYPE)"

STATISTIC(PassPredictionCounter, "Counts number of functions greeted");

namespace {
  struct PassPrediction : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    PassPrediction() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override;
  };
}

char PassPrediction::ID = 0;

/// (passName, arg, name, cfg, analysis)
INITIALIZE_PASS_BEGIN(PassPrediction, "passpredict",
                "PassPrediction Pass", false, false)
//INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(PassPrediction, "passpredict",
                "PassPrediction Pass", false, false)

FunctionPass *llvm::createPassPredictionPass() {
  return new PassPrediction();
}

bool PassPrediction::runOnFunction(Function &F) {
  ++PassPredictionCounter;
  errs() << "PassPrediction: ";
  errs().write_escaped(F.getName()) << '\n';

  ///try add pass
  //PMTopLevelManager PM;
  //PM.schedulePass(createDeadCodeEliminationPass());
  return false;
}

static RegisterPass<PassPrediction> X("PassPrediction", "PassPrediction Pass");
