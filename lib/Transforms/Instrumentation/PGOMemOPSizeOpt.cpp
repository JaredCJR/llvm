#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===-- PGOMemOPSizeOpt.cpp - Optimizations based on value profiling ===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the transformation that optimizes memory intrinsics
// such as memcpy using the size value profile. When memory intrinsic size
// value profile metadata is available, a single memory intrinsic is expanded
// to a sequence of guarded specialized versions that are called with the
// hottest size(s), for later expansion into more optimal inline sequences.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/PGOInstrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pgo-memop-opt"

STATISTIC(NumOfPGOMemOPOpt, "Number of memop intrinsics optimized.");
STATISTIC(NumOfPGOMemOPAnnotate, "Number of memop intrinsics annotated.");

// The minimum call count to optimize memory intrinsic calls.
static cl::opt<unsigned>
    MemOPCountThreshold("pgo-memop-count-threshold", cl::Hidden, cl::ZeroOrMore,
                        cl::init(1000),
                        cl::desc("The minimum count to optimize memory "
                                 "intrinsic calls"));

// Command line option to disable memory intrinsic optimization. The default is
// false. This is for debug purpose.
static cl::opt<bool> DisableMemOPOPT("disable-memop-opt", cl::init(false),
                                     cl::Hidden, cl::desc("Disable optimize"));

// The percent threshold to optimize memory intrinsic calls.
static cl::opt<unsigned>
    MemOPPercentThreshold("pgo-memop-percent-threshold", cl::init(40),
                          cl::Hidden, cl::ZeroOrMore,
                          cl::desc("The percentage threshold for the "
                                   "memory intrinsic calls optimization"));

// Maximum number of versions for optimizing memory intrinsic call.
static cl::opt<unsigned>
    MemOPMaxVersion("pgo-memop-max-version", cl::init(3), cl::Hidden,
                    cl::ZeroOrMore,
                    cl::desc("The max version for the optimized memory "
                             " intrinsic calls"));

// Scale the counts from the annotation using the BB count value.
static cl::opt<bool>
    MemOPScaleCount("pgo-memop-scale-count", cl::init(true), cl::Hidden,
                    cl::desc("Scale the memop size counts using the basic "
                             " block count value"));

// This option sets the rangge of precise profile memop sizes.
extern cl::opt<std::string> MemOPSizeRange;

// This option sets the value that groups large memop sizes
extern cl::opt<unsigned> MemOPSizeLarge;

namespace {
class PGOMemOPSizeOptLegacyPass : public FunctionPass {
public:
  static char ID;

  PGOMemOPSizeOptLegacyPass() : FunctionPass(ID) {
    initializePGOMemOPSizeOptLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "PGOMemOPSize"; }

private:
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
} // end anonymous namespace

char PGOMemOPSizeOptLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(PGOMemOPSizeOptLegacyPass, "pgo-memop-opt",
                      "Optimize memory intrinsic using its size value profile",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(PGOMemOPSizeOptLegacyPass, "pgo-memop-opt",
                    "Optimize memory intrinsic using its size value profile",
                    false, false)

FunctionPass *llvm::createPGOMemOPSizeOptLegacyPass() {
  return new PGOMemOPSizeOptLegacyPass();
}

namespace {
class MemOPSizeOpt : public InstVisitor<MemOPSizeOpt> {
public:
  MemOPSizeOpt(Function &Func, BlockFrequencyInfo &BFI)
      : Func(Func), BFI(BFI), Changed(false) {
    ValueDataArray =
        llvm::make_unique<InstrProfValueData[]>(MemOPMaxVersion + 2);
    // Get the MemOPSize range information from option MemOPSizeRange,
    getMemOPSizeRangeFromOption(MemOPSizeRange, PreciseRangeStart,
                                PreciseRangeLast);
  }
  bool isChanged() const { return Changed; }
  void perform() {
    WorkList.clear();
    visit(Func);

    for (auto &MI : WorkList) {
      PassPrediction::PassPeeper(0); // for-range
      ++NumOfPGOMemOPAnnotate;
      if (perform(MI)) {
        PassPrediction::PassPeeper(1); // if
        Changed = true;
        ++NumOfPGOMemOPOpt;
        DEBUG(dbgs() << "MemOP call: " << MI->getCalledFunction()->getName()
                     << "is Transformed.\n");
      }
    }
  }

  void visitMemIntrinsic(MemIntrinsic &MI) {
    Value *Length = MI.getLength();
    // Not perform on constant length calls.
    if (dyn_cast<ConstantInt>(Length)) {
      PassPrediction::PassPeeper(2); // if
      return;
    }
    WorkList.push_back(&MI);
  }

private:
  Function &Func;
  BlockFrequencyInfo &BFI;
  bool Changed;
  std::vector<MemIntrinsic *> WorkList;
  // Start of the previse range.
  int64_t PreciseRangeStart;
  // Last value of the previse range.
  int64_t PreciseRangeLast;
  // The space to read the profile annotation.
  std::unique_ptr<InstrProfValueData[]> ValueDataArray;
  bool perform(MemIntrinsic *MI);

  // This kind shows which group the value falls in. For PreciseValue, we have
  // the profile count for that value. LargeGroup groups the values that are in
  // range [LargeValue, +inf). NonLargeGroup groups the rest of values.
  enum MemOPSizeKind { PreciseValue, NonLargeGroup, LargeGroup };

  MemOPSizeKind getMemOPSizeKind(int64_t Value) const {
    if (Value == MemOPSizeLarge && MemOPSizeLarge != 0) {
      PassPrediction::PassPeeper(3); // if
      return LargeGroup;
    }
    if (Value == PreciseRangeLast + 1) {
      PassPrediction::PassPeeper(4); // if
      return NonLargeGroup;
    }
    return PreciseValue;
  }
};

static const char *getMIName(const MemIntrinsic *MI) {
  switch (MI->getIntrinsicID()) {
  case Intrinsic::memcpy:
    PassPrediction::PassPeeper(5); // case

    return "memcpy";
  case Intrinsic::memmove:
    PassPrediction::PassPeeper(6); // case

    return "memmove";
  case Intrinsic::memset:
    PassPrediction::PassPeeper(7); // case

    return "memset";
  default:
    return "unknown";
  }
}

static bool isProfitable(uint64_t Count, uint64_t TotalCount) {
  assert(Count <= TotalCount);
  if (Count < MemOPCountThreshold) {
    PassPrediction::PassPeeper(8); // if
    return false;
  }
  if (Count < TotalCount * MemOPPercentThreshold / 100) {
    PassPrediction::PassPeeper(9); // if
    return false;
  }
  return true;
}

static inline uint64_t getScaledCount(uint64_t Count, uint64_t Num,
                                      uint64_t Denom) {
  if (!MemOPScaleCount) {
    PassPrediction::PassPeeper(10); // if
    return Count;
  }
  bool Overflowed;
  uint64_t ScaleCount = SaturatingMultiply(Count, Num, &Overflowed);
  return ScaleCount / Denom;
}

bool MemOPSizeOpt::perform(MemIntrinsic *MI) {
  assert(MI);
  if (MI->getIntrinsicID() == Intrinsic::memmove) {
    PassPrediction::PassPeeper(11); // if
    return false;
  }

  uint32_t NumVals, MaxNumPromotions = MemOPMaxVersion + 2;
  uint64_t TotalCount;
  if (!getValueProfDataFromInst(*MI, IPVK_MemOPSize, MaxNumPromotions,
                                ValueDataArray.get(), NumVals, TotalCount)) {
    PassPrediction::PassPeeper(12); // if
    return false;
  }

  uint64_t ActualCount = TotalCount;
  uint64_t SavedTotalCount = TotalCount;
  if (MemOPScaleCount) {
    PassPrediction::PassPeeper(13); // if
    auto BBEdgeCount = BFI.getBlockProfileCount(MI->getParent());
    if (!BBEdgeCount) {
      PassPrediction::PassPeeper(14); // if
      return false;
    }
    ActualCount = *BBEdgeCount;
  }

  ArrayRef<InstrProfValueData> VDs(ValueDataArray.get(), NumVals);
  DEBUG(dbgs() << "Read one memory intrinsic profile with count " << ActualCount
               << "\n");
  DEBUG(
      for (auto &VD
           : VDs) { dbgs() << "  (" << VD.Value << "," << VD.Count << ")\n"; });

  if (ActualCount < MemOPCountThreshold) {
    PassPrediction::PassPeeper(15); // if
    return false;
  }
  // Skip if the total value profiled count is 0, in which case we can't
  // scale up the counts properly (and there is no profitable transformation).
  if (TotalCount == 0) {
    PassPrediction::PassPeeper(16); // if
    return false;
  }

  TotalCount = ActualCount;
  if (MemOPScaleCount) {
    DEBUG(dbgs() << "Scale counts: numerator = " << ActualCount
                 << " denominator = " << SavedTotalCount << "\n");
  }

  // Keeping track of the count of the default case:
  uint64_t RemainCount = TotalCount;
  uint64_t SavedRemainCount = SavedTotalCount;
  SmallVector<uint64_t, 16> SizeIds;
  SmallVector<uint64_t, 16> CaseCounts;
  uint64_t MaxCount = 0;
  unsigned Version = 0;
  // Default case is in the front -- save the slot here.
  CaseCounts.push_back(0);
  for (auto &VD : VDs) {
    PassPrediction::PassPeeper(17); // for-range
    int64_t V = VD.Value;
    uint64_t C = VD.Count;
    if (MemOPScaleCount) {
      PassPrediction::PassPeeper(18); // if
      C = getScaledCount(C, ActualCount, SavedTotalCount);
    }

    // Only care precise value here.
    if (getMemOPSizeKind(V) != PreciseValue) {
      PassPrediction::PassPeeper(19); // if
      continue;
    }

    // ValueCounts are sorted on the count. Break at the first un-profitable
    // value.
    if (!isProfitable(C, RemainCount)) {
      PassPrediction::PassPeeper(20); // if
      break;
    }

    SizeIds.push_back(V);
    CaseCounts.push_back(C);
    if (C > MaxCount) {
      PassPrediction::PassPeeper(21); // if
      MaxCount = C;
    }

    assert(RemainCount >= C);
    RemainCount -= C;
    assert(SavedRemainCount >= VD.Count);
    SavedRemainCount -= VD.Count;

    if (++Version > MemOPMaxVersion && MemOPMaxVersion != 0) {
      PassPrediction::PassPeeper(22); // if
      break;
    }
  }

  if (Version == 0) {
    PassPrediction::PassPeeper(23); // if
    return false;
  }

  CaseCounts[0] = RemainCount;
  if (RemainCount > MaxCount) {
    PassPrediction::PassPeeper(24); // if
    MaxCount = RemainCount;
  }

  uint64_t SumForOpt = TotalCount - RemainCount;

  DEBUG(dbgs() << "Optimize one memory intrinsic call to " << Version
               << " Versions (covering " << SumForOpt << " out of "
               << TotalCount << ")\n");

  // mem_op(..., size)
  // ==>
  // switch (size) {
  //   case s1:
  //      mem_op(..., s1);
  //      goto merge_bb;
  //   case s2:
  //      mem_op(..., s2);
  //      goto merge_bb;
  //   ...
  //   default:
  //      mem_op(..., size);
  //      goto merge_bb;
  // }
  // merge_bb:

  BasicBlock *BB = MI->getParent();
  DEBUG(dbgs() << "\n\n== Basic Block Before ==\n");
  DEBUG(dbgs() << *BB << "\n");
  auto OrigBBFreq = BFI.getBlockFreq(BB);

  BasicBlock *DefaultBB = SplitBlock(BB, MI);
  BasicBlock::iterator It(*MI);
  ++It;
  assert(It != DefaultBB->end());
  BasicBlock *MergeBB = SplitBlock(DefaultBB, &(*It));
  MergeBB->setName("MemOP.Merge");
  BFI.setBlockFreq(MergeBB, OrigBBFreq.getFrequency());
  DefaultBB->setName("MemOP.Default");

  auto &Ctx = Func.getContext();
  IRBuilder<> IRB(BB);
  BB->getTerminator()->eraseFromParent();
  Value *SizeVar = MI->getLength();
  SwitchInst *SI = IRB.CreateSwitch(SizeVar, DefaultBB, SizeIds.size());

  // Clear the value profile data.
  MI->setMetadata(LLVMContext::MD_prof, nullptr);
  // If all promoted, we don't need the MD.prof metadata.
  if (SavedRemainCount > 0 || Version != NumVals) {
    // Otherwise we need update with the un-promoted records back.
    PassPrediction::PassPeeper(25); // if
    annotateValueSite(*Func.getParent(), *MI, VDs.slice(Version),
                      SavedRemainCount, IPVK_MemOPSize, NumVals);
  }

  DEBUG(dbgs() << "\n\n== Basic Block After==\n");

  for (uint64_t SizeId : SizeIds) {
    PassPrediction::PassPeeper(26); // for-range
    ConstantInt *CaseSizeId = ConstantInt::get(Type::getInt64Ty(Ctx), SizeId);
    BasicBlock *CaseBB = BasicBlock::Create(
        Ctx, Twine("MemOP.Case.") + Twine(SizeId), &Func, DefaultBB);
    Instruction *NewInst = MI->clone();
    // Fix the argument.
    dyn_cast<MemIntrinsic>(NewInst)->setLength(CaseSizeId);
    CaseBB->getInstList().push_back(NewInst);
    IRBuilder<> IRBCase(CaseBB);
    IRBCase.CreateBr(MergeBB);
    SI->addCase(CaseSizeId, CaseBB);
    DEBUG(dbgs() << *CaseBB << "\n");
  }
  setProfMetadata(Func.getParent(), SI, CaseCounts, MaxCount);

  DEBUG(dbgs() << *BB << "\n");
  DEBUG(dbgs() << *DefaultBB << "\n");
  DEBUG(dbgs() << *MergeBB << "\n");

  emitOptimizationRemark(Func.getContext(), "memop-opt", Func,
                         MI->getDebugLoc(),
                         Twine("optimize ") + getMIName(MI) + " with count " +
                             Twine(SumForOpt) + " out of " + Twine(TotalCount) +
                             " for " + Twine(Version) + " versions");

  return true;
}
} // namespace

static bool PGOMemOPSizeOptImpl(Function &F, BlockFrequencyInfo &BFI) {
  if (DisableMemOPOPT) {
    PassPrediction::PassPeeper(27); // if
    return false;
  }

  if (F.hasFnAttribute(Attribute::OptimizeForSize)) {
    PassPrediction::PassPeeper(28); // if
    return false;
  }
  MemOPSizeOpt MemOPSizeOpt(F, BFI);
  MemOPSizeOpt.perform();
  return MemOPSizeOpt.isChanged();
}

bool PGOMemOPSizeOptLegacyPass::runOnFunction(Function &F) {
  BlockFrequencyInfo &BFI =
      getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
  return PGOMemOPSizeOptImpl(F, BFI);
}

namespace llvm {
char &PGOMemOPSizeOptID = PGOMemOPSizeOptLegacyPass::ID;

PreservedAnalyses PGOMemOPSizeOpt::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
  bool Changed = PGOMemOPSizeOptImpl(F, BFI);
  if (!Changed) {
    PassPrediction::PassPeeper(29); // if
    return PreservedAnalyses::all();
  }
  auto PA = PreservedAnalyses();
  PA.preserve<GlobalsAA>();
  return PA;
}
} // namespace llvm
