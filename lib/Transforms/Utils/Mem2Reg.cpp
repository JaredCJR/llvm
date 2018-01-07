#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
using namespace llvm;

#define DEBUG_TYPE "mem2reg"

STATISTIC(NumPromoted, "Number of alloca's promoted");

static bool promoteMemoryToRegister(Function &F, DominatorTree &DT,
                                    AssumptionCache &AC) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (1) {
    PassPrediction::PassPeeper(__FILE__, 573); // while
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I) {
      PassPrediction::PassPeeper(__FILE__, 574);      // for
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) { // Is it an alloca?
        PassPrediction::PassPeeper(__FILE__, 575);    // if
        if (isAllocaPromotable(AI)) {
          PassPrediction::PassPeeper(__FILE__, 576); // if
          Allocas.push_back(AI);
        }
      }
    }

    if (Allocas.empty()) {
      PassPrediction::PassPeeper(__FILE__, 577); // if
      break;
    }

    PromoteMemToReg(Allocas, DT, &AC);
    NumPromoted += Allocas.size();
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses PromotePass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  if (!promoteMemoryToRegister(F, DT, AC)) {
    PassPrediction::PassPeeper(__FILE__, 578); // if
    return PreservedAnalyses::all();
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {
struct PromoteLegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  PromoteLegacyPass() : FunctionPass(ID) {
    initializePromoteLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  // runOnFunction - To run this pass, first we calculate the alloca
  // instructions that are safe for promotion, then we promote each one.
  //
  bool runOnFunction(Function &F) override {
    if (skipFunction(F)) {
      PassPrediction::PassPeeper(__FILE__, 579); // if
      return false;
    }

    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    AssumptionCache &AC =
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    return promoteMemoryToRegister(F, DT, AC);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // end of anonymous namespace

char PromoteLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(PromoteLegacyPass, "mem2reg",
                      "Promote Memory to "
                      "Register",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(PromoteLegacyPass, "mem2reg", "Promote Memory to Register",
                    false, false)

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
//
FunctionPass *llvm::createPromoteMemoryToRegisterPass() {
  return new PromoteLegacyPass();
}
