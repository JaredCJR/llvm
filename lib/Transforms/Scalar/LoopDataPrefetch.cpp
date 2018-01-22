#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===-------- LoopDataPrefetch.cpp - Loop Data Prefetching Pass -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a Loop Data Prefetching Pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"

#define DEBUG_TYPE "loop-data-prefetch"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

// By default, we limit this to creating 16 PHIs (which is a little over half
// of the allocatable register set).
static cl::opt<bool> PrefetchWrites("loop-prefetch-writes", cl::Hidden,
                                    cl::init(false),
                                    cl::desc("Prefetch write addresses"));

static cl::opt<unsigned>
    PrefetchDistance("prefetch-distance",
                     cl::desc("Number of instructions to prefetch ahead"),
                     cl::Hidden);

static cl::opt<unsigned>
    MinPrefetchStride("min-prefetch-stride",
                      cl::desc("Min stride to add prefetches"), cl::Hidden);

static cl::opt<unsigned> MaxPrefetchIterationsAhead(
    "max-prefetch-iters-ahead",
    cl::desc("Max number of iterations to prefetch ahead"), cl::Hidden);

STATISTIC(NumPrefetches, "Number of prefetches inserted");

namespace {

/// Loop prefetch implementation class.
class LoopDataPrefetch {
public:
  LoopDataPrefetch(AssumptionCache *AC, LoopInfo *LI, ScalarEvolution *SE,
                   const TargetTransformInfo *TTI,
                   OptimizationRemarkEmitter *ORE)
      : AC(AC), LI(LI), SE(SE), TTI(TTI), ORE(ORE) {}

  bool run();

private:
  bool runOnLoop(Loop *L);

  /// \brief Check if the the stride of the accesses is large enough to
  /// warrant a prefetch.
  bool isStrideLargeEnough(const SCEVAddRecExpr *AR);

  unsigned getMinPrefetchStride() {
    if (MinPrefetchStride.getNumOccurrences() > 0) {
      PassPrediction::PassPeeper(591); // if
      return MinPrefetchStride;
    }
    return TTI->getMinPrefetchStride();
  }

  unsigned getPrefetchDistance() {
    if (PrefetchDistance.getNumOccurrences() > 0) {
      PassPrediction::PassPeeper(592); // if
      return PrefetchDistance;
    }
    return TTI->getPrefetchDistance();
  }

  unsigned getMaxPrefetchIterationsAhead() {
    if (MaxPrefetchIterationsAhead.getNumOccurrences() > 0) {
      PassPrediction::PassPeeper(593); // if
      return MaxPrefetchIterationsAhead;
    }
    return TTI->getMaxPrefetchIterationsAhead();
  }

  AssumptionCache *AC;
  LoopInfo *LI;
  ScalarEvolution *SE;
  const TargetTransformInfo *TTI;
  OptimizationRemarkEmitter *ORE;
};

/// Legacy class for inserting loop data prefetches.
class LoopDataPrefetchLegacyPass : public FunctionPass {
public:
  static char ID; // Pass ID, replacement for typeid
  LoopDataPrefetchLegacyPass() : FunctionPass(ID) {
    initializeLoopDataPrefetchLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    // FIXME: For some reason, preserving SE here breaks LSR (even if
    // this pass changes nothing).
    // AU.addPreserved<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};
} // namespace

char LoopDataPrefetchLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopDataPrefetchLegacyPass, "loop-data-prefetch",
                      "Loop Data Prefetch", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(LoopDataPrefetchLegacyPass, "loop-data-prefetch",
                    "Loop Data Prefetch", false, false)

FunctionPass *llvm::createLoopDataPrefetchPass() {
  return new LoopDataPrefetchLegacyPass();
}

bool LoopDataPrefetch::isStrideLargeEnough(const SCEVAddRecExpr *AR) {
  unsigned TargetMinStride = getMinPrefetchStride();
  // No need to check if any stride goes.
  if (TargetMinStride <= 1) {
    PassPrediction::PassPeeper(594); // if
    return true;
  }

  const auto *ConstStride = dyn_cast<SCEVConstant>(AR->getStepRecurrence(*SE));
  // If MinStride is set, don't prefetch unless we can ensure that stride is
  // larger.
  if (!ConstStride) {
    PassPrediction::PassPeeper(595); // if
    return false;
  }

  unsigned AbsStride = std::abs(ConstStride->getAPInt().getSExtValue());
  return TargetMinStride <= AbsStride;
}

PreservedAnalyses LoopDataPrefetchPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  LoopInfo *LI = &AM.getResult<LoopAnalysis>(F);
  ScalarEvolution *SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  AssumptionCache *AC = &AM.getResult<AssumptionAnalysis>(F);
  OptimizationRemarkEmitter *ORE =
      &AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  const TargetTransformInfo *TTI = &AM.getResult<TargetIRAnalysis>(F);

  LoopDataPrefetch LDP(AC, LI, SE, TTI, ORE);
  bool Changed = LDP.run();

  if (Changed) {
    PassPrediction::PassPeeper(596); // if
    PreservedAnalyses PA;
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<LoopAnalysis>();
    return PA;
  }

  return PreservedAnalyses::all();
}

bool LoopDataPrefetchLegacyPass::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    PassPrediction::PassPeeper(597); // if
    return false;
  }

  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  AssumptionCache *AC =
      &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  OptimizationRemarkEmitter *ORE =
      &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
  const TargetTransformInfo *TTI =
      &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  LoopDataPrefetch LDP(AC, LI, SE, TTI, ORE);
  return LDP.run();
}

bool LoopDataPrefetch::run() {
  // If PrefetchDistance is not set, don't run the pass.  This gives an
  // opportunity for targets to run this pass for selected subtargets only
  // (whose TTI sets PrefetchDistance).
  if (getPrefetchDistance() == 0) {
    PassPrediction::PassPeeper(598); // if
    return false;
  }
  assert(TTI->getCacheLineSize() && "Cache line size is not set for target");

  bool MadeChange = false;

  for (Loop *I : *LI) {
    PassPrediction::PassPeeper(599); // for-range
    for (auto L = df_begin(I), LE = df_end(I); L != LE; ++L) {
      PassPrediction::PassPeeper(600); // for
      MadeChange |= runOnLoop(*L);
    }
  }

  return MadeChange;
}

bool LoopDataPrefetch::runOnLoop(Loop *L) {
  bool MadeChange = false;

  // Only prefetch in the inner-most loop
  if (!L->empty()) {
    PassPrediction::PassPeeper(601); // if
    return MadeChange;
  }

  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, AC, EphValues);

  // Calculate the number of iterations ahead to prefetch
  CodeMetrics Metrics;
  for (const auto BB : L->blocks()) {
    // If the loop already has prefetches, then assume that the user knows
    // what they are doing and don't add any more.
    PassPrediction::PassPeeper(602); // for-range
    for (auto &I : *BB) {
      PassPrediction::PassPeeper(603); // for-range
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        PassPrediction::PassPeeper(604); // if
        if (Function *F = CI->getCalledFunction()) {
          PassPrediction::PassPeeper(605); // if
          if (F->getIntrinsicID() == Intrinsic::prefetch) {
            PassPrediction::PassPeeper(606); // if
            return MadeChange;
          }
        }
      }
    }

    Metrics.analyzeBasicBlock(BB, *TTI, EphValues);
  }
  unsigned LoopSize = Metrics.NumInsts;
  if (!LoopSize) {
    PassPrediction::PassPeeper(607); // if
    LoopSize = 1;
  }

  unsigned ItersAhead = getPrefetchDistance() / LoopSize;
  if (!ItersAhead) {
    PassPrediction::PassPeeper(608); // if
    ItersAhead = 1;
  }

  if (ItersAhead > getMaxPrefetchIterationsAhead()) {
    PassPrediction::PassPeeper(609); // if
    return MadeChange;
  }

  DEBUG(dbgs() << "Prefetching " << ItersAhead
               << " iterations ahead (loop size: " << LoopSize << ") in "
               << L->getHeader()->getParent()->getName() << ": " << *L);

  SmallVector<std::pair<Instruction *, const SCEVAddRecExpr *>, 16> PrefLoads;
  for (const auto BB : L->blocks()) {
    PassPrediction::PassPeeper(610); // for-range
    for (auto &I : *BB) {
      PassPrediction::PassPeeper(611); // for-range
      Value *PtrValue;
      Instruction *MemI;

      if (LoadInst *LMemI = dyn_cast<LoadInst>(&I)) {
        PassPrediction::PassPeeper(612); // if
        MemI = LMemI;
        PtrValue = LMemI->getPointerOperand();
      } else if (StoreInst *SMemI = dyn_cast<StoreInst>(&I)) {
        PassPrediction::PassPeeper(613); // if
        if (!PrefetchWrites) {
          continue;
        }
        MemI = SMemI;
        PtrValue = SMemI->getPointerOperand();
      } else {
        PassPrediction::PassPeeper(614); // else
        continue;
      }

      unsigned PtrAddrSpace = PtrValue->getType()->getPointerAddressSpace();
      if (PtrAddrSpace) {
        PassPrediction::PassPeeper(615); // if
        continue;
      }

      if (L->isLoopInvariant(PtrValue)) {
        PassPrediction::PassPeeper(616); // if
        continue;
      }

      const SCEV *LSCEV = SE->getSCEV(PtrValue);
      const SCEVAddRecExpr *LSCEVAddRec = dyn_cast<SCEVAddRecExpr>(LSCEV);
      if (!LSCEVAddRec) {
        PassPrediction::PassPeeper(617); // if
        continue;
      }

      // Check if the the stride of the accesses is large enough to warrant a
      // prefetch.
      if (!isStrideLargeEnough(LSCEVAddRec)) {
        PassPrediction::PassPeeper(618); // if
        continue;
      }

      // We don't want to double prefetch individual cache lines. If this load
      // is known to be within one cache line of some other load that has
      // already been prefetched, then don't prefetch this one as well.
      bool DupPref = false;
      for (const auto &PrefLoad : PrefLoads) {
        PassPrediction::PassPeeper(619); // for-range
        const SCEV *PtrDiff = SE->getMinusSCEV(LSCEVAddRec, PrefLoad.second);
        if (const SCEVConstant *ConstPtrDiff =
                dyn_cast<SCEVConstant>(PtrDiff)) {
          PassPrediction::PassPeeper(620); // if
          int64_t PD = std::abs(ConstPtrDiff->getValue()->getSExtValue());
          if (PD < (int64_t)TTI->getCacheLineSize()) {
            PassPrediction::PassPeeper(621); // if
            DupPref = true;
            PassPrediction::PassPeeper(622); // break
            break;
          }
        }
      }
      if (DupPref) {
        PassPrediction::PassPeeper(623); // if
        continue;
      }

      const SCEV *NextLSCEV = SE->getAddExpr(
          LSCEVAddRec,
          SE->getMulExpr(SE->getConstant(LSCEVAddRec->getType(), ItersAhead),
                         LSCEVAddRec->getStepRecurrence(*SE)));
      if (!isSafeToExpand(NextLSCEV, *SE)) {
        PassPrediction::PassPeeper(624); // if
        continue;
      }

      PrefLoads.push_back(std::make_pair(MemI, LSCEVAddRec));

      Type *I8Ptr = Type::getInt8PtrTy(BB->getContext(), PtrAddrSpace);
      SCEVExpander SCEVE(*SE, I.getModule()->getDataLayout(), "prefaddr");
      Value *PrefPtrValue = SCEVE.expandCodeFor(NextLSCEV, I8Ptr, MemI);

      IRBuilder<> Builder(MemI);
      Module *M = BB->getParent()->getParent();
      Type *I32 = Type::getInt32Ty(BB->getContext());
      Value *PrefetchFunc = Intrinsic::getDeclaration(M, Intrinsic::prefetch);
      Builder.CreateCall(
          PrefetchFunc,
          {PrefPtrValue,
           ConstantInt::get(I32, MemI->mayReadFromMemory() ? 0 : 1),
           ConstantInt::get(I32, 3), ConstantInt::get(I32, 1)});
      ++NumPrefetches;
      DEBUG(dbgs() << "  Access: " << *PtrValue << ", SCEV: " << *LSCEV
                   << "\n");
      ORE->emit(OptimizationRemark(DEBUG_TYPE, "Prefetched", MemI)
                << "prefetched memory access");

      MadeChange = true;
    }
  }

  return MadeChange;
}
