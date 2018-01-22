#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===----------------------- AlignmentFromAssumptions.cpp -----------------===//
//                  Set Load/Store Alignments From Assumptions
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a ScalarEvolution-based transformation to set
// the alignments of load, stores and memory intrinsics based on the truth
// expressions of assume intrinsics. The primary motivation is to handle
// complex alignment assumptions that apply to vector loads and stores that
// appear after vectorization and unrolling.
//
//===----------------------------------------------------------------------===//

#define AA_NAME "alignment-from-assumptions"
#define DEBUG_TYPE AA_NAME
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
using namespace llvm;

STATISTIC(NumLoadAlignChanged,
          "Number of loads changed by alignment assumptions");
STATISTIC(NumStoreAlignChanged,
          "Number of stores changed by alignment assumptions");
STATISTIC(NumMemIntAlignChanged,
          "Number of memory intrinsics changed by alignment assumptions");

namespace {
struct AlignmentFromAssumptions : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  AlignmentFromAssumptions() : FunctionPass(ID) {
    initializeAlignmentFromAssumptionsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();

    AU.setPreservesCFG();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<ScalarEvolutionWrapperPass>();
  }

  AlignmentFromAssumptionsPass Impl;
};
} // namespace

char AlignmentFromAssumptions::ID = 0;
static const char aip_name[] = "Alignment from assumptions";
INITIALIZE_PASS_BEGIN(AlignmentFromAssumptions, AA_NAME, aip_name, false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AlignmentFromAssumptions, AA_NAME, aip_name, false, false)

FunctionPass *llvm::createAlignmentFromAssumptionsPass() {
  return new AlignmentFromAssumptions();
}

// Given an expression for the (constant) alignment, AlignSCEV, and an
// expression for the displacement between a pointer and the aligned address,
// DiffSCEV, compute the alignment of the displaced pointer if it can be reduced
// to a constant. Using SCEV to compute alignment handles the case where
// DiffSCEV is a recurrence with constant start such that the aligned offset
// is constant. e.g. {16,+,32} % 32 -> 16.
static unsigned getNewAlignmentDiff(const SCEV *DiffSCEV, const SCEV *AlignSCEV,
                                    ScalarEvolution *SE) {
  // DiffUnits = Diff % int64_t(Alignment)
  const SCEV *DiffAlignDiv = SE->getUDivExpr(DiffSCEV, AlignSCEV);
  const SCEV *DiffAlign = SE->getMulExpr(DiffAlignDiv, AlignSCEV);
  const SCEV *DiffUnitsSCEV = SE->getMinusSCEV(DiffAlign, DiffSCEV);

  DEBUG(dbgs() << "\talignment relative to " << *AlignSCEV << " is "
               << *DiffUnitsSCEV << " (diff: " << *DiffSCEV << ")\n");

  if (const SCEVConstant *ConstDUSCEV = dyn_cast<SCEVConstant>(DiffUnitsSCEV)) {
    PassPrediction::PassPeeper(2384); // if
    int64_t DiffUnits = ConstDUSCEV->getValue()->getSExtValue();

    // If the displacement is an exact multiple of the alignment, then the
    // displaced pointer has the same alignment as the aligned pointer, so
    // return the alignment value.
    if (!DiffUnits) {
      PassPrediction::PassPeeper(2385); // if
      return (unsigned)cast<SCEVConstant>(AlignSCEV)
          ->getValue()
          ->getSExtValue();
    }

    // If the displacement is not an exact multiple, but the remainder is a
    // constant, then return this remainder (but only if it is a power of 2).
    uint64_t DiffUnitsAbs = std::abs(DiffUnits);
    if (isPowerOf2_64(DiffUnitsAbs)) {
      PassPrediction::PassPeeper(2386); // if
      return (unsigned)DiffUnitsAbs;
    }
  }

  return 0;
}

// There is an address given by an offset OffSCEV from AASCEV which has an
// alignment AlignSCEV. Use that information, if possible, to compute a new
// alignment for Ptr.
static unsigned getNewAlignment(const SCEV *AASCEV, const SCEV *AlignSCEV,
                                const SCEV *OffSCEV, Value *Ptr,
                                ScalarEvolution *SE) {
  const SCEV *PtrSCEV = SE->getSCEV(Ptr);
  const SCEV *DiffSCEV = SE->getMinusSCEV(PtrSCEV, AASCEV);

  // On 32-bit platforms, DiffSCEV might now have type i32 -- we've always
  // sign-extended OffSCEV to i64, so make sure they agree again.
  DiffSCEV = SE->getNoopOrSignExtend(DiffSCEV, OffSCEV->getType());

  // What we really want to know is the overall offset to the aligned
  // address. This address is displaced by the provided offset.
  DiffSCEV = SE->getMinusSCEV(DiffSCEV, OffSCEV);

  DEBUG(dbgs() << "AFI: alignment of " << *Ptr << " relative to " << *AlignSCEV
               << " and offset " << *OffSCEV << " using diff " << *DiffSCEV
               << "\n");

  unsigned NewAlignment = getNewAlignmentDiff(DiffSCEV, AlignSCEV, SE);
  DEBUG(dbgs() << "\tnew alignment: " << NewAlignment << "\n");

  if (NewAlignment) {
    PassPrediction::PassPeeper(2387); // if
    return NewAlignment;
  } else if (const SCEVAddRecExpr *DiffARSCEV =
                 dyn_cast<SCEVAddRecExpr>(DiffSCEV)) {
    // The relative offset to the alignment assumption did not yield a constant,
    // but we should try harder: if we assume that a is 32-byte aligned, then in
    // for (i = 0; i < 1024; i += 4) r += a[i]; not all of the loads from a are
    // 32-byte aligned, but instead alternate between 32 and 16-byte alignment.
    // As a result, the new alignment will not be a constant, but can still
    // be improved over the default (of 4) to 16.

    PassPrediction::PassPeeper(2388); // if
    const SCEV *DiffStartSCEV = DiffARSCEV->getStart();
    const SCEV *DiffIncSCEV = DiffARSCEV->getStepRecurrence(*SE);

    DEBUG(dbgs() << "\ttrying start/inc alignment using start "
                 << *DiffStartSCEV << " and inc " << *DiffIncSCEV << "\n");

    // Now compute the new alignment using the displacement to the value in the
    // first iteration, and also the alignment using the per-iteration delta.
    // If these are the same, then use that answer. Otherwise, use the smaller
    // one, but only if it divides the larger one.
    NewAlignment = getNewAlignmentDiff(DiffStartSCEV, AlignSCEV, SE);
    unsigned NewIncAlignment = getNewAlignmentDiff(DiffIncSCEV, AlignSCEV, SE);

    DEBUG(dbgs() << "\tnew start alignment: " << NewAlignment << "\n");
    DEBUG(dbgs() << "\tnew inc alignment: " << NewIncAlignment << "\n");

    if (!NewAlignment || !NewIncAlignment) {
      PassPrediction::PassPeeper(2389); // if
      return 0;
    } else if (NewAlignment > NewIncAlignment) {
      PassPrediction::PassPeeper(2390); // if
      if (NewAlignment % NewIncAlignment == 0) {
        DEBUG(dbgs() << "\tnew start/inc alignment: " << NewIncAlignment
                     << "\n");
        return NewIncAlignment;
      }
    } else if (NewIncAlignment > NewAlignment) {
      PassPrediction::PassPeeper(2391); // if
      if (NewIncAlignment % NewAlignment == 0) {
        DEBUG(dbgs() << "\tnew start/inc alignment: " << NewAlignment << "\n");
        return NewAlignment;
      }
    } else if (NewIncAlignment == NewAlignment) {
      DEBUG(dbgs() << "\tnew start/inc alignment: " << NewAlignment << "\n");
      return NewAlignment;
    }
  }

  return 0;
}

bool AlignmentFromAssumptionsPass::extractAlignmentInfo(CallInst *I,
                                                        Value *&AAPtr,
                                                        const SCEV *&AlignSCEV,
                                                        const SCEV *&OffSCEV) {
  // An alignment assume must be a statement about the least-significant
  // bits of the pointer being zero, possibly with some offset.
  ICmpInst *ICI = dyn_cast<ICmpInst>(I->getArgOperand(0));
  if (!ICI) {
    PassPrediction::PassPeeper(2392); // if
    return false;
  }

  // This must be an expression of the form: x & m == 0.
  if (ICI->getPredicate() != ICmpInst::ICMP_EQ) {
    PassPrediction::PassPeeper(2393); // if
    return false;
  }

  // Swap things around so that the RHS is 0.
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);
  const SCEV *CmpLHSSCEV = SE->getSCEV(CmpLHS);
  const SCEV *CmpRHSSCEV = SE->getSCEV(CmpRHS);
  if (CmpLHSSCEV->isZero()) {
    PassPrediction::PassPeeper(2394); // if
    std::swap(CmpLHS, CmpRHS);
  } else if (!CmpRHSSCEV->isZero()) {
    PassPrediction::PassPeeper(2395); // if
    return false;
  }

  BinaryOperator *CmpBO = dyn_cast<BinaryOperator>(CmpLHS);
  if (!CmpBO || CmpBO->getOpcode() != Instruction::And) {
    PassPrediction::PassPeeper(2396); // if
    return false;
  }

  // Swap things around so that the right operand of the and is a constant
  // (the mask); we cannot deal with variable masks.
  Value *AndLHS = CmpBO->getOperand(0);
  Value *AndRHS = CmpBO->getOperand(1);
  const SCEV *AndLHSSCEV = SE->getSCEV(AndLHS);
  const SCEV *AndRHSSCEV = SE->getSCEV(AndRHS);
  if (isa<SCEVConstant>(AndLHSSCEV)) {
    PassPrediction::PassPeeper(2397); // if
    std::swap(AndLHS, AndRHS);
    std::swap(AndLHSSCEV, AndRHSSCEV);
  }

  const SCEVConstant *MaskSCEV = dyn_cast<SCEVConstant>(AndRHSSCEV);
  if (!MaskSCEV) {
    PassPrediction::PassPeeper(2398); // if
    return false;
  }

  // The mask must have some trailing ones (otherwise the condition is
  // trivial and tells us nothing about the alignment of the left operand).
  unsigned TrailingOnes = MaskSCEV->getAPInt().countTrailingOnes();
  if (!TrailingOnes) {
    PassPrediction::PassPeeper(2399); // if
    return false;
  }

  // Cap the alignment at the maximum with which LLVM can deal (and make sure
  // we don't overflow the shift).
  uint64_t Alignment;
  TrailingOnes =
      std::min(TrailingOnes, unsigned(sizeof(unsigned) * CHAR_BIT - 1));
  Alignment = std::min(1u << TrailingOnes, +Value::MaximumAlignment);

  Type *Int64Ty = Type::getInt64Ty(I->getParent()->getParent()->getContext());
  AlignSCEV = SE->getConstant(Int64Ty, Alignment);

  // The LHS might be a ptrtoint instruction, or it might be the pointer
  // with an offset.
  AAPtr = nullptr;
  OffSCEV = nullptr;
  if (PtrToIntInst *PToI = dyn_cast<PtrToIntInst>(AndLHS)) {
    PassPrediction::PassPeeper(2400); // if
    AAPtr = PToI->getPointerOperand();
    OffSCEV = SE->getZero(Int64Ty);
  } else if (const SCEVAddExpr *AndLHSAddSCEV =
                 dyn_cast<SCEVAddExpr>(AndLHSSCEV)) {
    // Try to find the ptrtoint; subtract it and the rest is the offset.
    PassPrediction::PassPeeper(2401); // if
    for (SCEVAddExpr::op_iterator J = AndLHSAddSCEV->op_begin(),
                                  JE = AndLHSAddSCEV->op_end();
         J != JE; ++J) {
      PassPrediction::PassPeeper(2402); // for
      if (const SCEVUnknown *OpUnk = dyn_cast<SCEVUnknown>(*J)) {
        PassPrediction::PassPeeper(2403); // if
        if (PtrToIntInst *PToI = dyn_cast<PtrToIntInst>(OpUnk->getValue())) {
          PassPrediction::PassPeeper(2404); // if
          AAPtr = PToI->getPointerOperand();
          OffSCEV = SE->getMinusSCEV(AndLHSAddSCEV, *J);
          PassPrediction::PassPeeper(2405); // break
          break;
        }
      }
    }
  }

  if (!AAPtr) {
    PassPrediction::PassPeeper(2406); // if
    return false;
  }

  // Sign extend the offset to 64 bits (so that it is like all of the other
  // expressions).
  unsigned OffSCEVBits = OffSCEV->getType()->getPrimitiveSizeInBits();
  if (OffSCEVBits < 64) {
    PassPrediction::PassPeeper(2407); // if
    OffSCEV = SE->getSignExtendExpr(OffSCEV, Int64Ty);
  } else if (OffSCEVBits > 64) {
    PassPrediction::PassPeeper(2408); // if
    return false;
  }

  AAPtr = AAPtr->stripPointerCasts();
  return true;
}

bool AlignmentFromAssumptionsPass::processAssumption(CallInst *ACall) {
  Value *AAPtr;
  const SCEV *AlignSCEV, *OffSCEV;
  if (!extractAlignmentInfo(ACall, AAPtr, AlignSCEV, OffSCEV)) {
    PassPrediction::PassPeeper(2409); // if
    return false;
  }

  // Skip ConstantPointerNull and UndefValue.  Assumptions on these shouldn't
  // affect other users.
  if (isa<ConstantData>(AAPtr)) {
    PassPrediction::PassPeeper(2410); // if
    return false;
  }

  const SCEV *AASCEV = SE->getSCEV(AAPtr);

  // Apply the assumption to all other users of the specified pointer.
  SmallPtrSet<Instruction *, 32> Visited;
  SmallVector<Instruction *, 16> WorkList;
  for (User *J : AAPtr->users()) {
    PassPrediction::PassPeeper(2411); // for-range
    if (J == ACall) {
      PassPrediction::PassPeeper(2412); // if
      continue;
    }

    if (Instruction *K = dyn_cast<Instruction>(J)) {
      PassPrediction::PassPeeper(2413); // if
      if (isValidAssumeForContext(ACall, K, DT)) {
        PassPrediction::PassPeeper(2414); // if
        WorkList.push_back(K);
      }
    }
  }

  while (!WorkList.empty()) {
    PassPrediction::PassPeeper(2415); // while
    Instruction *J = WorkList.pop_back_val();

    if (LoadInst *LI = dyn_cast<LoadInst>(J)) {
      PassPrediction::PassPeeper(2416); // if
      unsigned NewAlignment = getNewAlignment(AASCEV, AlignSCEV, OffSCEV,
                                              LI->getPointerOperand(), SE);

      if (NewAlignment > LI->getAlignment()) {
        PassPrediction::PassPeeper(2417); // if
        LI->setAlignment(NewAlignment);
        ++NumLoadAlignChanged;
      }
    } else if (StoreInst *SI = dyn_cast<StoreInst>(J)) {
      PassPrediction::PassPeeper(2418); // if
      unsigned NewAlignment = getNewAlignment(AASCEV, AlignSCEV, OffSCEV,
                                              SI->getPointerOperand(), SE);

      if (NewAlignment > SI->getAlignment()) {
        PassPrediction::PassPeeper(2419); // if
        SI->setAlignment(NewAlignment);
        ++NumStoreAlignChanged;
      }
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(J)) {
      PassPrediction::PassPeeper(2420); // if
      unsigned NewDestAlignment =
          getNewAlignment(AASCEV, AlignSCEV, OffSCEV, MI->getDest(), SE);

      // For memory transfers, we need a common alignment for both the
      // source and destination. If we have a new alignment for this
      // instruction, but only for one operand, save it. If we reach the
      // other operand through another assumption later, then we may
      // change the alignment at that point.
      if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) {
        PassPrediction::PassPeeper(2421); // if
        unsigned NewSrcAlignment =
            getNewAlignment(AASCEV, AlignSCEV, OffSCEV, MTI->getSource(), SE);

        DenseMap<MemTransferInst *, unsigned>::iterator DI =
            NewDestAlignments.find(MTI);
        unsigned AltDestAlignment =
            (DI == NewDestAlignments.end()) ? 0 : DI->second;

        DenseMap<MemTransferInst *, unsigned>::iterator SI =
            NewSrcAlignments.find(MTI);
        unsigned AltSrcAlignment =
            (SI == NewSrcAlignments.end()) ? 0 : SI->second;

        DEBUG(dbgs() << "\tmem trans: " << NewDestAlignment << " "
                     << AltDestAlignment << " " << NewSrcAlignment << " "
                     << AltSrcAlignment << "\n");

        // Of these four alignments, pick the largest possible...
        unsigned NewAlignment = 0;
        if (NewDestAlignment <= std::max(NewSrcAlignment, AltSrcAlignment)) {
          PassPrediction::PassPeeper(2422); // if
          NewAlignment = std::max(NewAlignment, NewDestAlignment);
        }
        if (AltDestAlignment <= std::max(NewSrcAlignment, AltSrcAlignment)) {
          PassPrediction::PassPeeper(2423); // if
          NewAlignment = std::max(NewAlignment, AltDestAlignment);
        }
        if (NewSrcAlignment <= std::max(NewDestAlignment, AltDestAlignment)) {
          PassPrediction::PassPeeper(2424); // if
          NewAlignment = std::max(NewAlignment, NewSrcAlignment);
        }
        if (AltSrcAlignment <= std::max(NewDestAlignment, AltDestAlignment)) {
          PassPrediction::PassPeeper(2425); // if
          NewAlignment = std::max(NewAlignment, AltSrcAlignment);
        }

        if (NewAlignment > MI->getAlignment()) {
          PassPrediction::PassPeeper(2426); // if
          MI->setAlignment(ConstantInt::get(
              Type::getInt32Ty(MI->getParent()->getContext()), NewAlignment));
          ++NumMemIntAlignChanged;
        }

        NewDestAlignments.insert(std::make_pair(MTI, NewDestAlignment));
        NewSrcAlignments.insert(std::make_pair(MTI, NewSrcAlignment));
      } else if (NewDestAlignment > MI->getAlignment()) {
        assert((!isa<MemIntrinsic>(MI) || isa<MemSetInst>(MI)) &&
               "Unknown memory intrinsic");

        MI->setAlignment(ConstantInt::get(
            Type::getInt32Ty(MI->getParent()->getContext()), NewDestAlignment));
        ++NumMemIntAlignChanged;
      }
    }

    // Now that we've updated that use of the pointer, look for other uses of
    // the pointer to update.
    Visited.insert(J);
    for (User *UJ : J->users()) {
      PassPrediction::PassPeeper(2427); // for-range
      Instruction *K = cast<Instruction>(UJ);
      if (!Visited.count(K) && isValidAssumeForContext(ACall, K, DT)) {
        PassPrediction::PassPeeper(2428); // if
        WorkList.push_back(K);
      }
    }
  }

  return true;
}

bool AlignmentFromAssumptions::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    PassPrediction::PassPeeper(2429); // if
    return false;
  }

  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  ScalarEvolution *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  return Impl.runImpl(F, AC, SE, DT);
}

bool AlignmentFromAssumptionsPass::runImpl(Function &F, AssumptionCache &AC,
                                           ScalarEvolution *SE_,
                                           DominatorTree *DT_) {
  SE = SE_;
  DT = DT_;

  NewDestAlignments.clear();
  NewSrcAlignments.clear();

  bool Changed = false;
  for (auto &AssumeVH : AC.assumptions()) {
    PassPrediction::PassPeeper(2430); // for-range
    if (AssumeVH) {
      PassPrediction::PassPeeper(2431); // if
      Changed |= processAssumption(cast<CallInst>(AssumeVH));
    }
  }

  return Changed;
}

PreservedAnalyses
AlignmentFromAssumptionsPass::run(Function &F, FunctionAnalysisManager &AM) {

  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  ScalarEvolution &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!runImpl(F, AC, &SE, &DT)) {
    PassPrediction::PassPeeper(2432); // if
    return PreservedAnalyses::all();
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<AAManager>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}
