#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===- TailRecursionElimination.cpp - Eliminate Tail Calls ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file transforms calls of the current function (self recursion) followed
// by a return instruction with a branch to the entry of the function, creating
// a loop.  This pass also implements the following extensions to the basic
// algorithm:
//
//  1. Trivial instructions between the call and return do not prevent the
//     transformation from taking place, though currently the analysis cannot
//     support moving any really useful instructions (only dead ones).
//  2. This pass transforms functions that are prevented from being tail
//     recursive by an associative and commutative expression to use an
//     accumulator variable, thus compiling the typical naive factorial or
//     'fib' implementation into efficient code.
//  3. TRE is performed if the function returns void, if the return
//     returns the result returned by the call, or if the function returns a
//     run-time constant on all exits from the function.  It is possible, though
//     unlikely, that the return returns something else (like constant 0), and
//     can still be TRE'd.  It can be TRE'd if ALL OTHER return instructions in
//     the function return the exact same value.
//  4. If it can prove that callees do not access their caller stack frame,
//     they are marked as eligible for tail call elimination (by the code
//     generator).
//
// There are several improvements that could be made:
//
//  1. If the function has any alloca instructions, these instructions will be
//     moved out of the entry block of the function, causing them to be
//     evaluated each time through the tail recursion.  Safely keeping allocas
//     in the entry block requires analysis to proves that the tail-called
//     function does not read or write the stack object.
//  2. Tail recursion is only performed if the call immediately precedes the
//     return instruction.  It's possible that there could be a jump between
//     the call and the return.
//  3. There can be intervening operations between the call and the return that
//     prevent the TRE from occurring.  For example, there could be GEP's and
//     stores to memory that will not be read or written by the call.  This
//     requires some substantial analysis (such as with DSA) to prove safe to
//     move ahead of the call, but doing so could allow many more TREs to be
//     performed, for example in TreeAdd/TreeAlloc from the treeadd benchmark.
//  4. The algorithm we use to detect if callees access their caller stack
//     frames is very primitive.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "tailcallelim"

STATISTIC(NumEliminated, "Number of tail calls removed");
STATISTIC(NumRetDuped, "Number of return duplicated");
STATISTIC(NumAccumAdded, "Number of accumulators introduced");

/// \brief Scan the specified function for alloca instructions.
/// If it contains any dynamic allocas, returns false.
static bool canTRE(Function &F) {
  // Because of PR962, we don't TRE dynamic allocas.
  return llvm::all_of(instructions(F), [](Instruction &I) {
    auto *AI = dyn_cast<AllocaInst>(&I);
    return !AI || AI->isStaticAlloca();
  });
}

namespace {
struct AllocaDerivedValueTracker {
  // Start at a root value and walk its use-def chain to mark calls that use the
  // value or a derived value in AllocaUsers, and places where it may escape in
  // EscapePoints.
  void walk(Value *Root) {
    SmallVector<Use *, 32> Worklist;
    SmallPtrSet<Use *, 32> Visited;

    auto AddUsesToWorklist = [&](Value *V) {
      for (auto &U : V->uses()) {
        PassPrediction::PassPeeper(2903); // for-range
        if (!Visited.insert(&U).second) {
          PassPrediction::PassPeeper(2904); // if
          continue;
        }
        Worklist.push_back(&U);
      }
    };

    AddUsesToWorklist(Root);

    while (!Worklist.empty()) {
      PassPrediction::PassPeeper(2905); // while
      Use *U = Worklist.pop_back_val();
      Instruction *I = cast<Instruction>(U->getUser());

      switch (I->getOpcode()) {
      case Instruction::Call:
        PassPrediction::PassPeeper(2906); // case

      case Instruction::Invoke:
        PassPrediction::PassPeeper(2907); // case
        {
          CallSite CS(I);
          bool IsNocapture =
              CS.isDataOperand(U) && CS.doesNotCapture(CS.getDataOperandNo(U));
          callUsesLocalStack(CS, IsNocapture);
          if (IsNocapture) {
            // If the alloca-derived argument is passed in as nocapture, then it
            // can't propagate to the call's return. That would be capturing.
            PassPrediction::PassPeeper(2908); // if
            continue;
          }
          PassPrediction::PassPeeper(2909); // break
          break;
        }
      case Instruction::Load:
        PassPrediction::PassPeeper(2910); // case
        {
          // The result of a load is not alloca-derived (unless an alloca has
          // otherwise escaped, but this is a local analysis).
          continue;
        }
      case Instruction::Store:
        PassPrediction::PassPeeper(2911); // case
        {
          if (U->getOperandNo() == 0) {
            PassPrediction::PassPeeper(2912); // if
            EscapePoints.insert(I);
          }
          continue; // Stores have no users to analyze.
        }
      case Instruction::BitCast:
        PassPrediction::PassPeeper(2913); // case

      case Instruction::GetElementPtr:
        PassPrediction::PassPeeper(2914); // case

      case Instruction::PHI:
        PassPrediction::PassPeeper(2915); // case

      case Instruction::Select:
        PassPrediction::PassPeeper(2916); // case

      case Instruction::AddrSpaceCast:
        PassPrediction::PassPeeper(2917); // case

        PassPrediction::PassPeeper(2918); // break
        break;
      default:
        EscapePoints.insert(I);
        PassPrediction::PassPeeper(2919); // break
        break;
      }

      AddUsesToWorklist(I);
    }
  }

  void callUsesLocalStack(CallSite CS, bool IsNocapture) {
    // Add it to the list of alloca users.
    AllocaUsers.insert(CS.getInstruction());

    // If it's nocapture then it can't capture this alloca.
    if (IsNocapture) {
      PassPrediction::PassPeeper(2920); // if
      return;
    }

    // If it can write to memory, it can leak the alloca value.
    if (!CS.onlyReadsMemory()) {
      PassPrediction::PassPeeper(2921); // if
      EscapePoints.insert(CS.getInstruction());
    }
  }

  SmallPtrSet<Instruction *, 32> AllocaUsers;
  SmallPtrSet<Instruction *, 32> EscapePoints;
};
} // namespace

static bool markTails(Function &F, bool &AllCallsAreTailCalls) {
  if (F.callsFunctionThatReturnsTwice()) {
    PassPrediction::PassPeeper(2922); // if
    return false;
  }
  AllCallsAreTailCalls = true;

  // The local stack holds all alloca instructions and all byval arguments.
  AllocaDerivedValueTracker Tracker;
  for (Argument &Arg : F.args()) {
    PassPrediction::PassPeeper(2923); // for-range
    if (Arg.hasByValAttr()) {
      PassPrediction::PassPeeper(2924); // if
      Tracker.walk(&Arg);
    }
  }
  for (auto &BB : F) {
    PassPrediction::PassPeeper(2925); // for-range
    for (auto &I : BB) {
      PassPrediction::PassPeeper(2926); // for-range
      if (AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
        PassPrediction::PassPeeper(2927); // if
        Tracker.walk(AI);
      }
    }
  }

  bool Modified = false;

  // Track whether a block is reachable after an alloca has escaped. Blocks that
  // contain the escaping instruction will be marked as being visited without an
  // escaped alloca, since that is how the block began.
  enum VisitType { UNVISITED, UNESCAPED, ESCAPED };
  DenseMap<BasicBlock *, VisitType> Visited;

  // We propagate the fact that an alloca has escaped from block to successor.
  // Visit the blocks that are propagating the escapedness first. To do this, we
  // maintain two worklists.
  SmallVector<BasicBlock *, 32> WorklistUnescaped, WorklistEscaped;

  // We may enter a block and visit it thinking that no alloca has escaped yet,
  // then see an escape point and go back around a loop edge and come back to
  // the same block twice. Because of this, we defer setting tail on calls when
  // we first encounter them in a block. Every entry in this list does not
  // statically use an alloca via use-def chain analysis, but may find an alloca
  // through other means if the block turns out to be reachable after an escape
  // point.
  SmallVector<CallInst *, 32> DeferredTails;

  BasicBlock *BB = &F.getEntryBlock();
  VisitType Escaped = UNESCAPED;
  do {
    PassPrediction::PassPeeper(2928); // do-while
    for (auto &I : *BB) {
      PassPrediction::PassPeeper(2929); // for-range
      if (Tracker.EscapePoints.count(&I)) {
        PassPrediction::PassPeeper(2930); // if
        Escaped = ESCAPED;
      }

      CallInst *CI = dyn_cast<CallInst>(&I);
      if (!CI || CI->isTailCall()) {
        PassPrediction::PassPeeper(2931); // if
        continue;
      }

      bool IsNoTail = CI->isNoTailCall() || CI->hasOperandBundles();

      if (!IsNoTail && CI->doesNotAccessMemory()) {
        // A call to a readnone function whose arguments are all things computed
        // outside this function can be marked tail. Even if you stored the
        // alloca address into a global, a readnone function can't load the
        // global anyhow.
        //
        // Note that this runs whether we know an alloca has escaped or not. If
        // it has, then we can't trust Tracker.AllocaUsers to be accurate.
        PassPrediction::PassPeeper(2932); // if
        bool SafeToTail = true;
        for (auto &Arg : CI->arg_operands()) {
          PassPrediction::PassPeeper(2933); // for-range
          if (isa<Constant>(Arg.getUser())) {
            PassPrediction::PassPeeper(2934); // if
            continue;
          }
          if (Argument *A = dyn_cast<Argument>(Arg.getUser())) {
            PassPrediction::PassPeeper(2935); // if
            if (!A->hasByValAttr()) {
              PassPrediction::PassPeeper(2936); // if
              continue;
            }
          }
          SafeToTail = false;
          PassPrediction::PassPeeper(2937); // break
          break;
        }
        if (SafeToTail) {
          PassPrediction::PassPeeper(2938); // if
          emitOptimizationRemark(
              F.getContext(), "tailcallelim", F, CI->getDebugLoc(),
              "marked this readnone call a tail call candidate");
          CI->setTailCall();
          Modified = true;
          continue;
        }
      }

      if (!IsNoTail && Escaped == UNESCAPED && !Tracker.AllocaUsers.count(CI)) {
        PassPrediction::PassPeeper(2939); // if
        DeferredTails.push_back(CI);
      } else {
        PassPrediction::PassPeeper(2940); // else
        AllCallsAreTailCalls = false;
      }
    }

    for (auto *SuccBB : make_range(succ_begin(BB), succ_end(BB))) {
      PassPrediction::PassPeeper(2941); // for-range
      auto &State = Visited[SuccBB];
      if (State < Escaped) {
        PassPrediction::PassPeeper(2942); // if
        State = Escaped;
        if (State == ESCAPED) {
          PassPrediction::PassPeeper(2943); // if
          WorklistEscaped.push_back(SuccBB);
        } else {
          PassPrediction::PassPeeper(2944); // else
          WorklistUnescaped.push_back(SuccBB);
        }
      }
    }

    if (!WorklistEscaped.empty()) {
      PassPrediction::PassPeeper(2945); // if
      BB = WorklistEscaped.pop_back_val();
      Escaped = ESCAPED;
    } else {
      PassPrediction::PassPeeper(2946); // else
      BB = nullptr;
      while (!WorklistUnescaped.empty()) {
        PassPrediction::PassPeeper(2947); // while
        auto *NextBB = WorklistUnescaped.pop_back_val();
        if (Visited[NextBB] == UNESCAPED) {
          PassPrediction::PassPeeper(2948); // if
          BB = NextBB;
          Escaped = UNESCAPED;
          PassPrediction::PassPeeper(2949); // break
          break;
        }
      }
    }
  } while (BB);

  for (CallInst *CI : DeferredTails) {
    PassPrediction::PassPeeper(2950); // for-range
    if (Visited[CI->getParent()] != ESCAPED) {
      // If the escape point was part way through the block, calls after the
      // escape point wouldn't have been put into DeferredTails.
      PassPrediction::PassPeeper(2951); // if
      emitOptimizationRemark(F.getContext(), "tailcallelim", F,
                             CI->getDebugLoc(),
                             "marked this call a tail call candidate");
      CI->setTailCall();
      Modified = true;
    } else {
      PassPrediction::PassPeeper(2952); // else
      AllCallsAreTailCalls = false;
    }
  }

  return Modified;
}

/// Return true if it is safe to move the specified
/// instruction from after the call to before the call, assuming that all
/// instructions between the call and this instruction are movable.
///
static bool canMoveAboveCall(Instruction *I, CallInst *CI, AliasAnalysis *AA) {
  // FIXME: We can move load/store/call/free instructions above the call if the
  // call does not mod/ref the memory location being processed.
  if (I->mayHaveSideEffects()) {      // This also handles volatile loads.
    PassPrediction::PassPeeper(2953); // if
    return false;
  }

  if (LoadInst *L = dyn_cast<LoadInst>(I)) {
    // Loads may always be moved above calls without side effects.
    PassPrediction::PassPeeper(2954); // if
    if (CI->mayHaveSideEffects()) {
      // Non-volatile loads may be moved above a call with side effects if it
      // does not write to memory and the load provably won't trap.
      // Writes to memory only matter if they may alias the pointer
      // being loaded from.
      PassPrediction::PassPeeper(2955); // if
      const DataLayout &DL = L->getModule()->getDataLayout();
      if ((AA->getModRefInfo(CI, MemoryLocation::get(L)) & MRI_Mod) ||
          !isSafeToLoadUnconditionally(L->getPointerOperand(),
                                       L->getAlignment(), DL, L)) {
        PassPrediction::PassPeeper(2956); // if
        return false;
      }
    }
  }

  // Otherwise, if this is a side-effect free instruction, check to make sure
  // that it does not use the return value of the call.  If it doesn't use the
  // return value of the call, it must only use things that are defined before
  // the call, or movable instructions between the call and the instruction
  // itself.
  return !is_contained(I->operands(), CI);
}

/// Return true if the specified value is the same when the return would exit
/// as it was when the initial iteration of the recursive function was executed.
///
/// We currently handle static constants and arguments that are not modified as
/// part of the recursion.
static bool isDynamicConstant(Value *V, CallInst *CI, ReturnInst *RI) {
  if (isa<Constant>(V)) {
    PassPrediction::PassPeeper(2957); // if
    return true;                      // Static constants are always dyn consts
  }

  // Check to see if this is an immutable argument, if so, the value
  // will be available to initialize the accumulator.
  if (Argument *Arg = dyn_cast<Argument>(V)) {
    // Figure out which argument number this is...
    PassPrediction::PassPeeper(2958); // if
    unsigned ArgNo = 0;
    Function *F = CI->getParent()->getParent();
    for (Function::arg_iterator AI = F->arg_begin(); &*AI != Arg; ++AI) {
      PassPrediction::PassPeeper(2959); // for
      ++ArgNo;
    }

    // If we are passing this argument into call as the corresponding
    // argument operand, then the argument is dynamically constant.
    // Otherwise, we cannot transform this function safely.
    if (CI->getArgOperand(ArgNo) == Arg) {
      PassPrediction::PassPeeper(2960); // if
      return true;
    }
  }

  // Switch cases are always constant integers. If the value is being switched
  // on and the return is only reachable from one of its cases, it's
  // effectively constant.
  if (BasicBlock *UniquePred = RI->getParent()->getUniquePredecessor()) {
    PassPrediction::PassPeeper(2961); // if
    if (SwitchInst *SI = dyn_cast<SwitchInst>(UniquePred->getTerminator())) {
      PassPrediction::PassPeeper(2962); // if
      if (SI->getCondition() == V) {
        PassPrediction::PassPeeper(2963); // if
        return SI->getDefaultDest() != RI->getParent();
      }
    }
  }

  // Not a constant or immutable argument, we can't safely transform.
  return false;
}

/// Check to see if the function containing the specified tail call consistently
/// returns the same runtime-constant value at all exit points except for
/// IgnoreRI. If so, return the returned value.
static Value *getCommonReturnValue(ReturnInst *IgnoreRI, CallInst *CI) {
  Function *F = CI->getParent()->getParent();
  Value *ReturnedValue = nullptr;

  for (BasicBlock &BBI : *F) {
    PassPrediction::PassPeeper(2964); // for-range
    ReturnInst *RI = dyn_cast<ReturnInst>(BBI.getTerminator());
    if (RI == nullptr || RI == IgnoreRI) {
      PassPrediction::PassPeeper(2965); // if
      continue;
    }

    // We can only perform this transformation if the value returned is
    // evaluatable at the start of the initial invocation of the function,
    // instead of at the end of the evaluation.
    //
    Value *RetOp = RI->getOperand(0);
    if (!isDynamicConstant(RetOp, CI, RI)) {
      PassPrediction::PassPeeper(2966); // if
      return nullptr;
    }

    if (ReturnedValue && RetOp != ReturnedValue) {
      PassPrediction::PassPeeper(2967); // if
      return nullptr; // Cannot transform if differing values are returned.
    }
    ReturnedValue = RetOp;
  }
  return ReturnedValue;
}

/// If the specified instruction can be transformed using accumulator recursion
/// elimination, return the constant which is the start of the accumulator
/// value.  Otherwise return null.
static Value *canTransformAccumulatorRecursion(Instruction *I, CallInst *CI) {
  if (!I->isAssociative() || !I->isCommutative()) {
    PassPrediction::PassPeeper(2968); // if
    return nullptr;
  }
  assert(I->getNumOperands() == 2 &&
         "Associative/commutative operations should have 2 args!");

  // Exactly one operand should be the result of the call instruction.
  if ((I->getOperand(0) == CI && I->getOperand(1) == CI) ||
      (I->getOperand(0) != CI && I->getOperand(1) != CI)) {
    PassPrediction::PassPeeper(2969); // if
    return nullptr;
  }

  // The only user of this instruction we allow is a single return instruction.
  if (!I->hasOneUse() || !isa<ReturnInst>(I->user_back())) {
    PassPrediction::PassPeeper(2970); // if
    return nullptr;
  }

  // Ok, now we have to check all of the other return instructions in this
  // function.  If they return non-constants or differing values, then we cannot
  // transform the function safely.
  return getCommonReturnValue(cast<ReturnInst>(I->user_back()), CI);
}

static Instruction *firstNonDbg(BasicBlock::iterator I) {
  while (isa<DbgInfoIntrinsic>(I)) {
    PassPrediction::PassPeeper(2971); // while
    ++I;
  }
  return &*I;
}

static CallInst *findTRECandidate(Instruction *TI,
                                  bool CannotTailCallElimCallsMarkedTail,
                                  const TargetTransformInfo *TTI) {
  BasicBlock *BB = TI->getParent();
  Function *F = BB->getParent();

  if (&BB->front() ==
      TI) { // Make sure there is something before the terminator.
    PassPrediction::PassPeeper(2972); // if
    return nullptr;
  }

  // Scan backwards from the return, checking to see if there is a tail call in
  // this block.  If so, set CI to it.
  CallInst *CI = nullptr;
  BasicBlock::iterator BBI(TI);
  while (true) {
    PassPrediction::PassPeeper(2973); // while
    CI = dyn_cast<CallInst>(BBI);
    if (CI && CI->getCalledFunction() == F) {
      PassPrediction::PassPeeper(2974); // if
      break;
    }

    if (BBI == BB->begin()) {
      PassPrediction::PassPeeper(2975); // if
      return nullptr;                   // Didn't find a potential tail call.
    }
    --BBI;
  }

  // If this call is marked as a tail call, and if there are dynamic allocas in
  // the function, we cannot perform this optimization.
  if (CI->isTailCall() && CannotTailCallElimCallsMarkedTail) {
    PassPrediction::PassPeeper(2976); // if
    return nullptr;
  }

  // As a special case, detect code like this:
  //   double fabs(double f) { return __builtin_fabs(f); } // a 'fabs' call
  // and disable this xform in this case, because the code generator will
  // lower the call to fabs into inline code.
  if (BB == &F->getEntryBlock() &&
      firstNonDbg(BB->front().getIterator()) == CI &&
      firstNonDbg(std::next(BB->begin())) == TI && CI->getCalledFunction() &&
      !TTI->isLoweredToCall(CI->getCalledFunction())) {
    // A single-block function with just a call and a return. Check that
    // the arguments match.
    PassPrediction::PassPeeper(2977); // if
    CallSite::arg_iterator I = CallSite(CI).arg_begin(),
                           E = CallSite(CI).arg_end();
    Function::arg_iterator FI = F->arg_begin(), FE = F->arg_end();
    for (; I != E && FI != FE; ++I, ++FI) {
      PassPrediction::PassPeeper(2978); // for
      if (*I != &*FI) {
        break;
      }
    }
    if (I == E && FI == FE) {
      PassPrediction::PassPeeper(2979); // if
      return nullptr;
    }
  }

  return CI;
}

static bool eliminateRecursiveTailCall(CallInst *CI, ReturnInst *Ret,
                                       BasicBlock *&OldEntry,
                                       bool &TailCallsAreMarkedTail,
                                       SmallVectorImpl<PHINode *> &ArgumentPHIs,
                                       AliasAnalysis *AA) {
  // If we are introducing accumulator recursion to eliminate operations after
  // the call instruction that are both associative and commutative, the initial
  // value for the accumulator is placed in this variable.  If this value is set
  // then we actually perform accumulator recursion elimination instead of
  // simple tail recursion elimination.  If the operation is an LLVM instruction
  // (eg: "add") then it is recorded in AccumulatorRecursionInstr.  If not, then
  // we are handling the case when the return instruction returns a constant C
  // which is different to the constant returned by other return instructions
  // (which is recorded in AccumulatorRecursionEliminationInitVal).  This is a
  // special case of accumulator recursion, the operation being "return C".
  Value *AccumulatorRecursionEliminationInitVal = nullptr;
  Instruction *AccumulatorRecursionInstr = nullptr;

  // Ok, we found a potential tail call.  We can currently only transform the
  // tail call if all of the instructions between the call and the return are
  // movable to above the call itself, leaving the call next to the return.
  // Check that this is the case now.
  BasicBlock::iterator BBI(CI);
  for (++BBI; &*BBI != Ret; ++BBI) {
    PassPrediction::PassPeeper(2980); // for
    if (canMoveAboveCall(&*BBI, CI, AA)) {
      PassPrediction::PassPeeper(2981); // if
      continue;
    }

    // If we can't move the instruction above the call, it might be because it
    // is an associative and commutative operation that could be transformed
    // using accumulator recursion elimination.  Check to see if this is the
    // case, and if so, remember the initial accumulator value for later.
    if ((AccumulatorRecursionEliminationInitVal =
             canTransformAccumulatorRecursion(&*BBI, CI))) {
      // Yes, this is accumulator recursion.  Remember which instruction
      // accumulates.
      PassPrediction::PassPeeper(2982); // if
      AccumulatorRecursionInstr = &*BBI;
    } else {
      PassPrediction::PassPeeper(2983); // else
      return false; // Otherwise, we cannot eliminate the tail recursion!
    }
  }

  // We can only transform call/return pairs that either ignore the return value
  // of the call and return void, ignore the value of the call and return a
  // constant, return the value returned by the tail call, or that are being
  // accumulator recursion variable eliminated.
  if (Ret->getNumOperands() == 1 && Ret->getReturnValue() != CI &&
      !isa<UndefValue>(Ret->getReturnValue()) &&
      AccumulatorRecursionEliminationInitVal == nullptr &&
      !getCommonReturnValue(nullptr, CI)) {
    // One case remains that we are able to handle: the current return
    // instruction returns a constant, and all other return instructions
    // return a different constant.
    PassPrediction::PassPeeper(2984); // if
    if (!isDynamicConstant(Ret->getReturnValue(), CI, Ret)) {
      PassPrediction::PassPeeper(2985); // if
      return false; // Current return instruction does not return a constant.
    }
    // Check that all other return instructions return a common constant.  If
    // so, record it in AccumulatorRecursionEliminationInitVal.
    AccumulatorRecursionEliminationInitVal = getCommonReturnValue(Ret, CI);
    if (!AccumulatorRecursionEliminationInitVal) {
      PassPrediction::PassPeeper(2986); // if
      return false;
    }
  }

  BasicBlock *BB = Ret->getParent();
  Function *F = BB->getParent();

  emitOptimizationRemark(F->getContext(), "tailcallelim", *F, CI->getDebugLoc(),
                         "transforming tail recursion to loop");

  // OK! We can transform this tail call.  If this is the first one found,
  // create the new entry block, allowing us to branch back to the old entry.
  if (!OldEntry) {
    PassPrediction::PassPeeper(2987); // if
    OldEntry = &F->getEntryBlock();
    BasicBlock *NewEntry = BasicBlock::Create(F->getContext(), "", F, OldEntry);
    NewEntry->takeName(OldEntry);
    OldEntry->setName("tailrecurse");
    BranchInst::Create(OldEntry, NewEntry);

    // If this tail call is marked 'tail' and if there are any allocas in the
    // entry block, move them up to the new entry block.
    TailCallsAreMarkedTail = CI->isTailCall();
    if (TailCallsAreMarkedTail) {
      // Move all fixed sized allocas from OldEntry to NewEntry.
      PassPrediction::PassPeeper(2988); // if
      for (BasicBlock::iterator OEBI = OldEntry->begin(), E = OldEntry->end(),
                                NEBI = NewEntry->begin();
           OEBI != E;) {
        PassPrediction::PassPeeper(2989); // for
        if (AllocaInst *AI = dyn_cast<AllocaInst>(OEBI++)) {
          PassPrediction::PassPeeper(2990); // if
          if (isa<ConstantInt>(AI->getArraySize())) {
            PassPrediction::PassPeeper(2991); // if
            AI->moveBefore(&*NEBI);
          }
        }
      }
    }

    // Now that we have created a new block, which jumps to the entry
    // block, insert a PHI node for each argument of the function.
    // For now, we initialize each PHI to only have the real arguments
    // which are passed in.
    Instruction *InsertPos = &OldEntry->front();
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
         ++I) {
      PassPrediction::PassPeeper(2992); // for
      PHINode *PN =
          PHINode::Create(I->getType(), 2, I->getName() + ".tr", InsertPos);
      I->replaceAllUsesWith(PN); // Everyone use the PHI node now!
      PN->addIncoming(&*I, NewEntry);
      ArgumentPHIs.push_back(PN);
    }
  }

  // If this function has self recursive calls in the tail position where some
  // are marked tail and some are not, only transform one flavor or another.  We
  // have to choose whether we move allocas in the entry block to the new entry
  // block or not, so we can't make a good choice for both.  NOTE: We could do
  // slightly better here in the case that the function has no entry block
  // allocas.
  if (TailCallsAreMarkedTail && !CI->isTailCall()) {
    PassPrediction::PassPeeper(2993); // if
    return false;
  }

  // Ok, now that we know we have a pseudo-entry block WITH all of the
  // required PHI nodes, add entries into the PHI node for the actual
  // parameters passed into the tail-recursive call.
  for (unsigned i = 0, e = CI->getNumArgOperands(); i != e; ++i) {
    PassPrediction::PassPeeper(2994); // for
    ArgumentPHIs[i]->addIncoming(CI->getArgOperand(i), BB);
  }

  // If we are introducing an accumulator variable to eliminate the recursion,
  // do so now.  Note that we _know_ that no subsequent tail recursion
  // eliminations will happen on this function because of the way the
  // accumulator recursion predicate is set up.
  //
  if (AccumulatorRecursionEliminationInitVal) {
    PassPrediction::PassPeeper(2995); // if
    Instruction *AccRecInstr = AccumulatorRecursionInstr;
    // Start by inserting a new PHI node for the accumulator.
    pred_iterator PB = pred_begin(OldEntry), PE = pred_end(OldEntry);
    PHINode *AccPN = PHINode::Create(
        AccumulatorRecursionEliminationInitVal->getType(),
        std::distance(PB, PE) + 1, "accumulator.tr", &OldEntry->front());

    // Loop over all of the predecessors of the tail recursion block.  For the
    // real entry into the function we seed the PHI with the initial value,
    // computed earlier.  For any other existing branches to this block (due to
    // other tail recursions eliminated) the accumulator is not modified.
    // Because we haven't added the branch in the current block to OldEntry yet,
    // it will not show up as a predecessor.
    for (pred_iterator PI = PB; PI != PE; ++PI) {
      PassPrediction::PassPeeper(2996); // for
      BasicBlock *P = *PI;
      if (P == &F->getEntryBlock()) {
        PassPrediction::PassPeeper(2997); // if
        AccPN->addIncoming(AccumulatorRecursionEliminationInitVal, P);
      } else {
        PassPrediction::PassPeeper(2998); // else
        AccPN->addIncoming(AccPN, P);
      }
    }

    if (AccRecInstr) {
      // Add an incoming argument for the current block, which is computed by
      // our associative and commutative accumulator instruction.
      PassPrediction::PassPeeper(2999); // if
      AccPN->addIncoming(AccRecInstr, BB);

      // Next, rewrite the accumulator recursion instruction so that it does not
      // use the result of the call anymore, instead, use the PHI node we just
      // inserted.
      AccRecInstr->setOperand(AccRecInstr->getOperand(0) != CI, AccPN);
    } else {
      // Add an incoming argument for the current block, which is just the
      // constant returned by the current return instruction.
      PassPrediction::PassPeeper(3000); // else
      AccPN->addIncoming(Ret->getReturnValue(), BB);
    }

    // Finally, rewrite any return instructions in the program to return the PHI
    // node instead of the "initval" that they do currently.  This loop will
    // actually rewrite the return value we are destroying, but that's ok.
    for (BasicBlock &BBI : *F) {
      PassPrediction::PassPeeper(3001); // for-range
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BBI.getTerminator())) {
        PassPrediction::PassPeeper(3002); // if
        RI->setOperand(0, AccPN);
      }
    }
    ++NumAccumAdded;
  }

  // Now that all of the PHI nodes are in place, remove the call and
  // ret instructions, replacing them with an unconditional branch.
  BranchInst *NewBI = BranchInst::Create(OldEntry, Ret);
  NewBI->setDebugLoc(CI->getDebugLoc());

  BB->getInstList().erase(Ret); // Remove return.
  BB->getInstList().erase(CI);  // Remove call.
  ++NumEliminated;
  return true;
}

static bool foldReturnAndProcessPred(BasicBlock *BB, ReturnInst *Ret,
                                     BasicBlock *&OldEntry,
                                     bool &TailCallsAreMarkedTail,
                                     SmallVectorImpl<PHINode *> &ArgumentPHIs,
                                     bool CannotTailCallElimCallsMarkedTail,
                                     const TargetTransformInfo *TTI,
                                     AliasAnalysis *AA) {
  bool Change = false;

  // Make sure this block is a trivial return block.
  assert(BB->getFirstNonPHIOrDbg() == Ret &&
         "Trying to fold non-trivial return block");

  // If the return block contains nothing but the return and PHI's,
  // there might be an opportunity to duplicate the return in its
  // predecessors and perform TRE there. Look for predecessors that end
  // in unconditional branch and recursive call(s).
  SmallVector<BranchInst *, 8> UncondBranchPreds;
  for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
    PassPrediction::PassPeeper(3003); // for
    BasicBlock *Pred = *PI;
    TerminatorInst *PTI = Pred->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(PTI)) {
      PassPrediction::PassPeeper(3004); // if
      if (BI->isUnconditional()) {
        PassPrediction::PassPeeper(3005); // if
        UncondBranchPreds.push_back(BI);
      }
    }
  }

  while (!UncondBranchPreds.empty()) {
    PassPrediction::PassPeeper(3006); // while
    BranchInst *BI = UncondBranchPreds.pop_back_val();
    BasicBlock *Pred = BI->getParent();
    if (CallInst *CI =
            findTRECandidate(BI, CannotTailCallElimCallsMarkedTail, TTI)) {
      DEBUG(dbgs() << "FOLDING: " << *BB
                   << "INTO UNCOND BRANCH PRED: " << *Pred);
      ReturnInst *RI = FoldReturnIntoUncondBranch(Ret, BB, Pred);

      // Cleanup: if all predecessors of BB have been eliminated by
      // FoldReturnIntoUncondBranch, delete it.  It is important to empty it,
      // because the ret instruction in there is still using a value which
      // eliminateRecursiveTailCall will attempt to remove.
      if (!BB->hasAddressTaken() && pred_begin(BB) == pred_end(BB)) {
        PassPrediction::PassPeeper(3007); // if
        BB->eraseFromParent();
      }

      eliminateRecursiveTailCall(CI, RI, OldEntry, TailCallsAreMarkedTail,
                                 ArgumentPHIs, AA);
      ++NumRetDuped;
      Change = true;
    }
  }

  return Change;
}

static bool processReturningBlock(ReturnInst *Ret, BasicBlock *&OldEntry,
                                  bool &TailCallsAreMarkedTail,
                                  SmallVectorImpl<PHINode *> &ArgumentPHIs,
                                  bool CannotTailCallElimCallsMarkedTail,
                                  const TargetTransformInfo *TTI,
                                  AliasAnalysis *AA) {
  CallInst *CI = findTRECandidate(Ret, CannotTailCallElimCallsMarkedTail, TTI);
  if (!CI) {
    PassPrediction::PassPeeper(3008); // if
    return false;
  }

  return eliminateRecursiveTailCall(CI, Ret, OldEntry, TailCallsAreMarkedTail,
                                    ArgumentPHIs, AA);
}

static bool eliminateTailRecursion(Function &F, const TargetTransformInfo *TTI,
                                   AliasAnalysis *AA) {
  if (F.getFnAttribute("disable-tail-calls").getValueAsString() == "true") {
    PassPrediction::PassPeeper(3009); // if
    return false;
  }

  bool MadeChange = false;
  bool AllCallsAreTailCalls = false;
  MadeChange |= markTails(F, AllCallsAreTailCalls);
  if (!AllCallsAreTailCalls) {
    PassPrediction::PassPeeper(3010); // if
    return MadeChange;
  }

  // If this function is a varargs function, we won't be able to PHI the args
  // right, so don't even try to convert it...
  if (F.getFunctionType()->isVarArg()) {
    PassPrediction::PassPeeper(3011); // if
    return false;
  }

  BasicBlock *OldEntry = nullptr;
  bool TailCallsAreMarkedTail = false;
  SmallVector<PHINode *, 8> ArgumentPHIs;

  // If false, we cannot perform TRE on tail calls marked with the 'tail'
  // attribute, because doing so would cause the stack size to increase (real
  // TRE would deallocate variable sized allocas, TRE doesn't).
  bool CanTRETailMarkedCall = canTRE(F);

  // Change any tail recursive calls to loops.
  //
  // FIXME: The code generator produces really bad code when an 'escaping
  // alloca' is changed from being a static alloca to being a dynamic alloca.
  // Until this is resolved, disable this transformation if that would ever
  // happen.  This bug is PR962.
  for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E; /*in loop*/) {
    PassPrediction::PassPeeper(3012); // for
    BasicBlock *BB = &*BBI++;         // foldReturnAndProcessPred may delete BB.
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
      PassPrediction::PassPeeper(3013); // if
      bool Change =
          processReturningBlock(Ret, OldEntry, TailCallsAreMarkedTail,
                                ArgumentPHIs, !CanTRETailMarkedCall, TTI, AA);
      if (!Change && BB->getFirstNonPHIOrDbg() == Ret) {
        PassPrediction::PassPeeper(3014); // if
        Change = foldReturnAndProcessPred(BB, Ret, OldEntry,
                                          TailCallsAreMarkedTail, ArgumentPHIs,
                                          !CanTRETailMarkedCall, TTI, AA);
      }
      MadeChange |= Change;
    }
  }

  // If we eliminated any tail recursions, it's possible that we inserted some
  // silly PHI nodes which just merge an initial value (the incoming operand)
  // with themselves.  Check to see if we did and clean up our mess if so.  This
  // occurs when a function passes an argument straight through to its tail
  // call.
  for (PHINode *PN : ArgumentPHIs) {
    // If the PHI Node is a dynamic constant, replace it with the value it is.
    PassPrediction::PassPeeper(3015); // for-range
    if (Value *PNV = SimplifyInstruction(PN, F.getParent()->getDataLayout())) {
      PassPrediction::PassPeeper(3016); // if
      PN->replaceAllUsesWith(PNV);
      PN->eraseFromParent();
    }
  }

  return MadeChange;
}

namespace {
struct TailCallElim : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  TailCallElim() : FunctionPass(ID) {
    initializeTailCallElimPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F)) {
      PassPrediction::PassPeeper(3017); // if
      return false;
    }

    return eliminateTailRecursion(
        F, &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F),
        &getAnalysis<AAResultsWrapperPass>().getAAResults());
  }
};
} // namespace

char TailCallElim::ID = 0;
INITIALIZE_PASS_BEGIN(TailCallElim, "tailcallelim", "Tail Call Elimination",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(TailCallElim, "tailcallelim", "Tail Call Elimination",
                    false, false)

// Public interface to the TailCallElimination pass
FunctionPass *llvm::createTailCallEliminationPass() {
  return new TailCallElim();
}

PreservedAnalyses TailCallElimPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {

  TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  AliasAnalysis &AA = AM.getResult<AAManager>(F);

  bool Changed = eliminateTailRecursion(F, &TTI, &AA);

  if (!Changed) {
    PassPrediction::PassPeeper(3018); // if
    return PreservedAnalyses::all();
  }
  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  return PA;
}
