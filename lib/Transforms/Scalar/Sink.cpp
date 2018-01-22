#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===-- Sink.cpp - Code Sinking -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass moves instructions into successor blocks, when possible, so that
// they aren't executed on paths where their results aren't needed.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/Sink.h"
using namespace llvm;

#define DEBUG_TYPE "sink"

STATISTIC(NumSunk, "Number of instructions sunk");
STATISTIC(NumSinkIter, "Number of sinking iterations");

/// AllUsesDominatedByBlock - Return true if all uses of the specified value
/// occur in blocks dominated by the specified block.
static bool AllUsesDominatedByBlock(Instruction *Inst, BasicBlock *BB,
                                    DominatorTree &DT) {
  // Ignoring debug uses is necessary so debug info doesn't affect the code.
  // This may leave a referencing dbg_value in the original block, before
  // the definition of the vreg.  Dwarf generator handles this although the
  // user might not get the right info at runtime.
  for (Use &U : Inst->uses()) {
    // Determine the block of the use.
    PassPrediction::PassPeeper(2180); // for-range
    Instruction *UseInst = cast<Instruction>(U.getUser());
    BasicBlock *UseBlock = UseInst->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(UseInst)) {
      // PHI nodes use the operand in the predecessor block, not the block with
      // the PHI.
      PassPrediction::PassPeeper(2181); // if
      unsigned Num = PHINode::getIncomingValueNumForOperand(U.getOperandNo());
      UseBlock = PN->getIncomingBlock(Num);
    }
    // Check that it dominates.
    if (!DT.dominates(BB, UseBlock)) {
      PassPrediction::PassPeeper(2182); // if
      return false;
    }
  }
  return true;
}

static bool isSafeToMove(Instruction *Inst, AliasAnalysis &AA,
                         SmallPtrSetImpl<Instruction *> &Stores) {

  if (Inst->mayWriteToMemory()) {
    PassPrediction::PassPeeper(2183); // if
    Stores.insert(Inst);
    return false;
  }

  if (LoadInst *L = dyn_cast<LoadInst>(Inst)) {
    PassPrediction::PassPeeper(2184); // if
    MemoryLocation Loc = MemoryLocation::get(L);
    for (Instruction *S : Stores) {
      PassPrediction::PassPeeper(2185); // for-range
      if (AA.getModRefInfo(S, Loc) & MRI_Mod) {
        PassPrediction::PassPeeper(2186); // if
        return false;
      }
    }
  }

  if (isa<TerminatorInst>(Inst) || isa<PHINode>(Inst) || Inst->isEHPad() ||
      Inst->mayThrow()) {
    PassPrediction::PassPeeper(2187); // if
    return false;
  }

  if (auto CS = CallSite(Inst)) {
    // Convergent operations cannot be made control-dependent on additional
    // values.
    PassPrediction::PassPeeper(2188); // if
    if (CS.hasFnAttr(Attribute::Convergent)) {
      PassPrediction::PassPeeper(2189); // if
      return false;
    }

    for (Instruction *S : Stores) {
      PassPrediction::PassPeeper(2190); // for-range
      if (AA.getModRefInfo(S, CS) & MRI_Mod) {
        PassPrediction::PassPeeper(2191); // if
        return false;
      }
    }
  }

  return true;
}

/// IsAcceptableTarget - Return true if it is possible to sink the instruction
/// in the specified basic block.
static bool IsAcceptableTarget(Instruction *Inst, BasicBlock *SuccToSinkTo,
                               DominatorTree &DT, LoopInfo &LI) {
  assert(Inst && "Instruction to be sunk is null");
  assert(SuccToSinkTo && "Candidate sink target is null");

  // It is not possible to sink an instruction into its own block.  This can
  // happen with loops.
  if (Inst->getParent() == SuccToSinkTo) {
    PassPrediction::PassPeeper(2192); // if
    return false;
  }

  // It's never legal to sink an instruction into a block which terminates in an
  // EH-pad.
  if (SuccToSinkTo->getTerminator()->isExceptional()) {
    PassPrediction::PassPeeper(2193); // if
    return false;
  }

  // If the block has multiple predecessors, this would introduce computation
  // on different code paths.  We could split the critical edge, but for now we
  // just punt.
  // FIXME: Split critical edges if not backedges.
  if (SuccToSinkTo->getUniquePredecessor() != Inst->getParent()) {
    // We cannot sink a load across a critical edge - there may be stores in
    // other code paths.
    PassPrediction::PassPeeper(2194); // if
    if (isa<LoadInst>(Inst)) {
      PassPrediction::PassPeeper(2195); // if
      return false;
    }

    // We don't want to sink across a critical edge if we don't dominate the
    // successor. We could be introducing calculations to new code paths.
    if (!DT.dominates(Inst->getParent(), SuccToSinkTo)) {
      PassPrediction::PassPeeper(2196); // if
      return false;
    }

    // Don't sink instructions into a loop.
    Loop *succ = LI.getLoopFor(SuccToSinkTo);
    Loop *cur = LI.getLoopFor(Inst->getParent());
    if (succ != nullptr && succ != cur) {
      PassPrediction::PassPeeper(2197); // if
      return false;
    }
  }

  // Finally, check that all the uses of the instruction are actually
  // dominated by the candidate
  return AllUsesDominatedByBlock(Inst, SuccToSinkTo, DT);
}

/// SinkInstruction - Determine whether it is safe to sink the specified machine
/// instruction out of its current block into a successor.
static bool SinkInstruction(Instruction *Inst,
                            SmallPtrSetImpl<Instruction *> &Stores,
                            DominatorTree &DT, LoopInfo &LI, AAResults &AA) {

  // Don't sink static alloca instructions.  CodeGen assumes allocas outside the
  // entry block are dynamically sized stack objects.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Inst)) {
    PassPrediction::PassPeeper(2198); // if
    if (AI->isStaticAlloca()) {
      PassPrediction::PassPeeper(2199); // if
      return false;
    }
  }

  // Check if it's safe to move the instruction.
  if (!isSafeToMove(Inst, AA, Stores)) {
    PassPrediction::PassPeeper(2200); // if
    return false;
  }

  // FIXME: This should include support for sinking instructions within the
  // block they are currently in to shorten the live ranges.  We often get
  // instructions sunk into the top of a large block, but it would be better to
  // also sink them down before their first use in the block.  This xform has to
  // be careful not to *increase* register pressure though, e.g. sinking
  // "x = y + z" down if it kills y and z would increase the live ranges of y
  // and z and only shrink the live range of x.

  // SuccToSinkTo - This is the successor to sink this instruction to, once we
  // decide.
  BasicBlock *SuccToSinkTo = nullptr;

  // Instructions can only be sunk if all their uses are in blocks
  // dominated by one of the successors.
  // Look at all the dominated blocks and see if we can sink it in one.
  DomTreeNode *DTN = DT.getNode(Inst->getParent());
  for (DomTreeNode::iterator I = DTN->begin(), E = DTN->end();
       I != E && SuccToSinkTo == nullptr; ++I) {
    PassPrediction::PassPeeper(2201); // for
    BasicBlock *Candidate = (*I)->getBlock();
    // A node always immediate-dominates its children on the dominator
    // tree.
    if (IsAcceptableTarget(Inst, Candidate, DT, LI)) {
      PassPrediction::PassPeeper(2202); // if
      SuccToSinkTo = Candidate;
    }
  }

  // If no suitable postdominator was found, look at all the successors and
  // decide which one we should sink to, if any.
  for (succ_iterator I = succ_begin(Inst->getParent()),
                     E = succ_end(Inst->getParent());
       I != E && !SuccToSinkTo; ++I) {
    PassPrediction::PassPeeper(2203); // for
    if (IsAcceptableTarget(Inst, *I, DT, LI)) {
      PassPrediction::PassPeeper(2204); // if
      SuccToSinkTo = *I;
    }
  }

  // If we couldn't find a block to sink to, ignore this instruction.
  if (!SuccToSinkTo) {
    PassPrediction::PassPeeper(2205); // if
    return false;
  }

  DEBUG(dbgs() << "Sink" << *Inst << " (";
        Inst->getParent()->printAsOperand(dbgs(), false); dbgs() << " -> ";
        SuccToSinkTo->printAsOperand(dbgs(), false); dbgs() << ")\n");

  // Move the instruction.
  Inst->moveBefore(&*SuccToSinkTo->getFirstInsertionPt());
  return true;
}

static bool ProcessBlock(BasicBlock &BB, DominatorTree &DT, LoopInfo &LI,
                         AAResults &AA) {
  // Can't sink anything out of a block that has less than two successors.
  if (BB.getTerminator()->getNumSuccessors() <= 1) {
    PassPrediction::PassPeeper(2206); // if
    return false;
  }

  // Don't bother sinking code out of unreachable blocks. In addition to being
  // unprofitable, it can also lead to infinite looping, because in an
  // unreachable loop there may be nowhere to stop.
  if (!DT.isReachableFromEntry(&BB)) {
    PassPrediction::PassPeeper(2207); // if
    return false;
  }

  bool MadeChange = false;

  // Walk the basic block bottom-up.  Remember if we saw a store.
  BasicBlock::iterator I = BB.end();
  --I;
  bool ProcessedBegin = false;
  SmallPtrSet<Instruction *, 8> Stores;
  do {
    PassPrediction::PassPeeper(2208); // do-while
    Instruction *Inst = &*I;          // The instruction to sink.

    // Predecrement I (if it's not begin) so that it isn't invalidated by
    // sinking.
    ProcessedBegin = I == BB.begin();
    if (!ProcessedBegin) {
      PassPrediction::PassPeeper(2209); // if
      --I;
    }

    if (isa<DbgInfoIntrinsic>(Inst)) {
      PassPrediction::PassPeeper(2210); // if
      continue;
    }

    if (SinkInstruction(Inst, Stores, DT, LI, AA)) {
      PassPrediction::PassPeeper(2211); // if
      ++NumSunk;
      MadeChange = true;
    }

    // If we just processed the first instruction in the block, we're done.
  } while (!ProcessedBegin);

  return MadeChange;
}

static bool iterativelySinkInstructions(Function &F, DominatorTree &DT,
                                        LoopInfo &LI, AAResults &AA) {
  bool MadeChange, EverMadeChange = false;

  do {
    PassPrediction::PassPeeper(2212); // do-while
    MadeChange = false;
    DEBUG(dbgs() << "Sinking iteration " << NumSinkIter << "\n");
    // Process all basic blocks.
    for (BasicBlock &I : F) {
      PassPrediction::PassPeeper(2213); // for-range
      MadeChange |= ProcessBlock(I, DT, LI, AA);
    }
    EverMadeChange |= MadeChange;
    NumSinkIter++;
  } while (MadeChange);

  return EverMadeChange;
}

PreservedAnalyses SinkingPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);

  if (!iterativelySinkInstructions(F, DT, LI, AA)) {
    PassPrediction::PassPeeper(2214); // if
    return PreservedAnalyses::all();
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {
class SinkingLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification
  SinkingLegacyPass() : FunctionPass(ID) {
    initializeSinkingLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();

    return iterativelySinkInstructions(F, DT, LI, AA);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
  }
};
} // end anonymous namespace

char SinkingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(SinkingLegacyPass, "sink", "Code sinking", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(SinkingLegacyPass, "sink", "Code sinking", false, false)

FunctionPass *llvm::createSinkingPass() { return new SinkingLegacyPass(); }
