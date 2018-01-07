#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===- Reassociate.cpp - Reassociate binary expressions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass reassociates commutative expressions in an order that is designed
// to promote better constant propagation, GCSE, LICM, PRE, etc.
//
// For example: 4 + (x + 5) -> x + (4 + 5)
//
// In the implementation of this algorithm, constants are assigned rank = 0,
// function arguments are rank = 1, and other values are assigned ranks
// corresponding to the reverse post order traversal of current function
// (starting at 2), which effectively gives values in deep loops higher rank
// than values not in loops.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
using namespace llvm;
using namespace reassociate;

#define DEBUG_TYPE "reassociate"

STATISTIC(NumChanged, "Number of insts reassociated");
STATISTIC(NumAnnihil, "Number of expr tree annihilated");
STATISTIC(NumFactor, "Number of multiplies factored");

#ifndef NDEBUG
/// Print out the expression identified in the Ops list.
///
static void PrintOps(Instruction *I, const SmallVectorImpl<ValueEntry> &Ops) {
  Module *M = I->getModule();
  dbgs() << Instruction::getOpcodeName(I->getOpcode()) << " "
         << *Ops[0].Op->getType() << '\t';
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3629); // for
    dbgs() << "[ ";
    Ops[i].Op->printAsOperand(dbgs(), false, M);
    dbgs() << ", #" << Ops[i].Rank << "] ";
  }
}
#endif

/// Utility class representing a non-constant Xor-operand. We classify
/// non-constant Xor-Operands into two categories:
///  C1) The operand is in the form "X & C", where C is a constant and C != ~0
///  C2)
///    C2.1) The operand is in the form of "X | C", where C is a non-zero
///          constant.
///    C2.2) Any operand E which doesn't fall into C1 and C2.1, we view this
///          operand as "E | 0"
class llvm::reassociate::XorOpnd {
public:
  XorOpnd(Value *V);

  bool isInvalid() const { return SymbolicPart == nullptr; }
  bool isOrExpr() const { return isOr; }
  Value *getValue() const { return OrigVal; }
  Value *getSymbolicPart() const { return SymbolicPart; }
  unsigned getSymbolicRank() const { return SymbolicRank; }
  const APInt &getConstPart() const { return ConstPart; }

  void Invalidate() { SymbolicPart = OrigVal = nullptr; }
  void setSymbolicRank(unsigned R) { SymbolicRank = R; }

private:
  Value *OrigVal;
  Value *SymbolicPart;
  APInt ConstPart;
  unsigned SymbolicRank;
  bool isOr;
};

XorOpnd::XorOpnd(Value *V) {
  assert(!isa<ConstantInt>(V) && "No ConstantInt");
  OrigVal = V;
  Instruction *I = dyn_cast<Instruction>(V);
  SymbolicRank = 0;

  if (I && (I->getOpcode() == Instruction::Or ||
            I->getOpcode() == Instruction::And)) {
    PassPrediction::PassPeeper(__FILE__, 3630); // if
    Value *V0 = I->getOperand(0);
    Value *V1 = I->getOperand(1);
    const APInt *C;
    if (match(V0, PatternMatch::m_APInt(C))) {
      PassPrediction::PassPeeper(__FILE__, 3631); // if
      std::swap(V0, V1);
    }

    if (match(V1, PatternMatch::m_APInt(C))) {
      PassPrediction::PassPeeper(__FILE__, 3632); // if
      ConstPart = *C;
      SymbolicPart = V0;
      isOr = (I->getOpcode() == Instruction::Or);
      return;
    }
  }

  // view the operand as "V | 0"
  SymbolicPart = V;
  ConstPart = APInt::getNullValue(V->getType()->getScalarSizeInBits());
  isOr = true;
}

/// Return true if V is an instruction of the specified opcode and if it
/// only has one use.
static BinaryOperator *isReassociableOp(Value *V, unsigned Opcode) {
  if (V->hasOneUse() && isa<Instruction>(V) &&
      cast<Instruction>(V)->getOpcode() == Opcode &&
      (!isa<FPMathOperator>(V) || cast<Instruction>(V)->hasUnsafeAlgebra())) {
    PassPrediction::PassPeeper(__FILE__, 3633); // if
    return cast<BinaryOperator>(V);
  }
  return nullptr;
}

static BinaryOperator *isReassociableOp(Value *V, unsigned Opcode1,
                                        unsigned Opcode2) {
  if (V->hasOneUse() && isa<Instruction>(V) &&
      (cast<Instruction>(V)->getOpcode() == Opcode1 ||
       cast<Instruction>(V)->getOpcode() == Opcode2) &&
      (!isa<FPMathOperator>(V) || cast<Instruction>(V)->hasUnsafeAlgebra())) {
    PassPrediction::PassPeeper(__FILE__, 3634); // if
    return cast<BinaryOperator>(V);
  }
  return nullptr;
}

void ReassociatePass::BuildRankMap(
    Function &F, ReversePostOrderTraversal<Function *> &RPOT) {
  unsigned i = 2;

  // Assign distinct ranks to function arguments.
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    PassPrediction::PassPeeper(__FILE__, 3635); // for
    ValueRankMap[&*I] = ++i;
    DEBUG(dbgs() << "Calculated Rank[" << I->getName() << "] = " << i << "\n");
  }

  // Traverse basic blocks in ReversePostOrder
  for (BasicBlock *BB : RPOT) {
    PassPrediction::PassPeeper(__FILE__, 3636); // for-range
    unsigned BBRank = RankMap[BB] = ++i << 16;

    // Walk the basic block, adding precomputed ranks for any instructions that
    // we cannot move.  This ensures that the ranks for these instructions are
    // all different in the block.
    for (Instruction &I : *BB) {
      PassPrediction::PassPeeper(__FILE__, 3637); // for-range
      if (mayBeMemoryDependent(I)) {
        PassPrediction::PassPeeper(__FILE__, 3638); // if
        ValueRankMap[&I] = ++BBRank;
      }
    }
  }
}

unsigned ReassociatePass::getRank(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) {
    PassPrediction::PassPeeper(__FILE__, 3639); // if
    if (isa<Argument>(V)) {
      return ValueRankMap[V]; // Function argument.
    }
    return 0; // Otherwise it's a global or constant, rank 0.
  }

  if (unsigned Rank = ValueRankMap[I]) {
    PassPrediction::PassPeeper(__FILE__, 3640); // if
    return Rank;                                // Rank already known?
  }

  // If this is an expression, return the 1+MAX(rank(LHS), rank(RHS)) so that
  // we can reassociate expressions for code motion!  Since we do not recurse
  // for PHI nodes, we cannot have infinite recursion here, because there
  // cannot be loops in the value graph that do not go through PHI nodes.
  unsigned Rank = 0, MaxRank = RankMap[I->getParent()];
  for (unsigned i = 0, e = I->getNumOperands(); i != e && Rank != MaxRank;
       ++i) {
    PassPrediction::PassPeeper(__FILE__, 3641); // for
    Rank = std::max(Rank, getRank(I->getOperand(i)));
  }

  // If this is a not or neg instruction, do not count it for rank.  This
  // assures us that X and ~X will have the same rank.
  if (!BinaryOperator::isNot(I) && !BinaryOperator::isNeg(I) &&
      !BinaryOperator::isFNeg(I)) {
    PassPrediction::PassPeeper(__FILE__, 3642); // if
    ++Rank;
  }

  DEBUG(dbgs() << "Calculated Rank[" << V->getName() << "] = " << Rank << "\n");

  return ValueRankMap[I] = Rank;
}

// Canonicalize constants to RHS.  Otherwise, sort the operands by rank.
void ReassociatePass::canonicalizeOperands(Instruction *I) {
  assert(isa<BinaryOperator>(I) && "Expected binary operator.");
  assert(I->isCommutative() && "Expected commutative operator.");

  Value *LHS = I->getOperand(0);
  Value *RHS = I->getOperand(1);
  unsigned LHSRank = getRank(LHS);
  unsigned RHSRank = getRank(RHS);

  if (isa<Constant>(RHS)) {
    PassPrediction::PassPeeper(__FILE__, 3643); // if
    return;
  }

  if (isa<Constant>(LHS) || RHSRank < LHSRank) {
    PassPrediction::PassPeeper(__FILE__, 3644); // if
    cast<BinaryOperator>(I)->swapOperands();
  }
}

static BinaryOperator *CreateAdd(Value *S1, Value *S2, const Twine &Name,
                                 Instruction *InsertBefore, Value *FlagsOp) {
  if (S1->getType()->isIntOrIntVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 3645); // if
    return BinaryOperator::CreateAdd(S1, S2, Name, InsertBefore);
  } else {
    PassPrediction::PassPeeper(__FILE__, 3646); // else
    BinaryOperator *Res =
        BinaryOperator::CreateFAdd(S1, S2, Name, InsertBefore);
    Res->setFastMathFlags(cast<FPMathOperator>(FlagsOp)->getFastMathFlags());
    return Res;
  }
}

static BinaryOperator *CreateMul(Value *S1, Value *S2, const Twine &Name,
                                 Instruction *InsertBefore, Value *FlagsOp) {
  if (S1->getType()->isIntOrIntVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 3647); // if
    return BinaryOperator::CreateMul(S1, S2, Name, InsertBefore);
  } else {
    PassPrediction::PassPeeper(__FILE__, 3648); // else
    BinaryOperator *Res =
        BinaryOperator::CreateFMul(S1, S2, Name, InsertBefore);
    Res->setFastMathFlags(cast<FPMathOperator>(FlagsOp)->getFastMathFlags());
    return Res;
  }
}

static BinaryOperator *CreateNeg(Value *S1, const Twine &Name,
                                 Instruction *InsertBefore, Value *FlagsOp) {
  if (S1->getType()->isIntOrIntVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 3649); // if
    return BinaryOperator::CreateNeg(S1, Name, InsertBefore);
  } else {
    PassPrediction::PassPeeper(__FILE__, 3650); // else
    BinaryOperator *Res = BinaryOperator::CreateFNeg(S1, Name, InsertBefore);
    Res->setFastMathFlags(cast<FPMathOperator>(FlagsOp)->getFastMathFlags());
    return Res;
  }
}

/// Replace 0-X with X*-1.
static BinaryOperator *LowerNegateToMultiply(Instruction *Neg) {
  Type *Ty = Neg->getType();
  Constant *NegOne = Ty->isIntOrIntVectorTy() ? ConstantInt::getAllOnesValue(Ty)
                                              : ConstantFP::get(Ty, -1.0);

  BinaryOperator *Res = CreateMul(Neg->getOperand(1), NegOne, "", Neg, Neg);
  Neg->setOperand(1, Constant::getNullValue(Ty)); // Drop use of op.
  Res->takeName(Neg);
  Neg->replaceAllUsesWith(Res);
  Res->setDebugLoc(Neg->getDebugLoc());
  return Res;
}

/// Returns k such that lambda(2^Bitwidth) = 2^k, where lambda is the Carmichael
/// function. This means that x^(2^k) === 1 mod 2^Bitwidth for
/// every odd x, i.e. x^(2^k) = 1 for every odd x in Bitwidth-bit arithmetic.
/// Note that 0 <= k < Bitwidth, and if Bitwidth > 3 then x^(2^k) = 0 for every
/// even x in Bitwidth-bit arithmetic.
static unsigned CarmichaelShift(unsigned Bitwidth) {
  if (Bitwidth < 3) {
    PassPrediction::PassPeeper(__FILE__, 3651); // if
    return Bitwidth - 1;
  }
  return Bitwidth - 2;
}

/// Add the extra weight 'RHS' to the existing weight 'LHS',
/// reducing the combined weight using any special properties of the operation.
/// The existing weight LHS represents the computation X op X op ... op X where
/// X occurs LHS times.  The combined weight represents  X op X op ... op X with
/// X occurring LHS + RHS times.  If op is "Xor" for example then the combined
/// operation is equivalent to X if LHS + RHS is odd, or 0 if LHS + RHS is even;
/// the routine returns 1 in LHS in the first case, and 0 in LHS in the second.
static void IncorporateWeight(APInt &LHS, const APInt &RHS, unsigned Opcode) {
  // If we were working with infinite precision arithmetic then the combined
  // weight would be LHS + RHS.  But we are using finite precision arithmetic,
  // and the APInt sum LHS + RHS may not be correct if it wraps (it is correct
  // for nilpotent operations and addition, but not for idempotent operations
  // and multiplication), so it is important to correctly reduce the combined
  // weight back into range if wrapping would be wrong.

  // If RHS is zero then the weight didn't change.
  if (RHS.isMinValue()) {
    PassPrediction::PassPeeper(__FILE__, 3652); // if
    return;
  }
  // If LHS is zero then the combined weight is RHS.
  if (LHS.isMinValue()) {
    PassPrediction::PassPeeper(__FILE__, 3653); // if
    LHS = RHS;
    return;
  }
  // From this point on we know that neither LHS nor RHS is zero.

  if (Instruction::isIdempotent(Opcode)) {
    // Idempotent means X op X === X, so any non-zero weight is equivalent to a
    // weight of 1.  Keeping weights at zero or one also means that wrapping is
    // not a problem.
    assert(LHS == 1 && RHS == 1 && "Weights not reduced!");
    return; // Return a weight of 1.
  }
  if (Instruction::isNilpotent(Opcode)) {
    // Nilpotent means X op X === 0, so reduce weights modulo 2.
    assert(LHS == 1 && RHS == 1 && "Weights not reduced!");
    LHS = 0; // 1 + 1 === 0 modulo 2.
    return;
  }
  if (Opcode == Instruction::Add || Opcode == Instruction::FAdd) {
    // TODO: Reduce the weight by exploiting nsw/nuw?
    PassPrediction::PassPeeper(__FILE__, 3654); // if
    LHS += RHS;
    return;
  }

  assert((Opcode == Instruction::Mul || Opcode == Instruction::FMul) &&
         "Unknown associative operation!");
  unsigned Bitwidth = LHS.getBitWidth();
  // If CM is the Carmichael number then a weight W satisfying W >= CM+Bitwidth
  // can be replaced with W-CM.  That's because x^W=x^(W-CM) for every Bitwidth
  // bit number x, since either x is odd in which case x^CM = 1, or x is even in
  // which case both x^W and x^(W - CM) are zero.  By subtracting off multiples
  // of CM like this weights can always be reduced to the range [0, CM+Bitwidth)
  // which by a happy accident means that they can always be represented using
  // Bitwidth bits.
  // TODO: Reduce the weight by exploiting nsw/nuw?  (Could do much better than
  // the Carmichael number).
  if (Bitwidth > 3) {
    /// CM - The value of Carmichael's lambda function.
    PassPrediction::PassPeeper(__FILE__, 3655); // if
    APInt CM = APInt::getOneBitSet(Bitwidth, CarmichaelShift(Bitwidth));
    // Any weight W >= Threshold can be replaced with W - CM.
    APInt Threshold = CM + Bitwidth;
    assert(LHS.ult(Threshold) && RHS.ult(Threshold) && "Weights not reduced!");
    // For Bitwidth 4 or more the following sum does not overflow.
    LHS += RHS;
    while (LHS.uge(Threshold)) {
      PassPrediction::PassPeeper(__FILE__, 3657); // while
      LHS -= CM;
    }
  } else {
    // To avoid problems with overflow do everything the same as above but using
    // a larger type.
    PassPrediction::PassPeeper(__FILE__, 3656); // else
    unsigned CM = 1U << CarmichaelShift(Bitwidth);
    unsigned Threshold = CM + Bitwidth;
    assert(LHS.getZExtValue() < Threshold && RHS.getZExtValue() < Threshold &&
           "Weights not reduced!");
    unsigned Total = LHS.getZExtValue() + RHS.getZExtValue();
    while (Total >= Threshold) {
      PassPrediction::PassPeeper(__FILE__, 3658); // while
      Total -= CM;
    }
    LHS = Total;
  }
}

typedef std::pair<Value *, APInt> RepeatedValue;

/// Given an associative binary expression, return the leaf
/// nodes in Ops along with their weights (how many times the leaf occurs).  The
/// original expression is the same as
///   (Ops[0].first op Ops[0].first op ... Ops[0].first)  <- Ops[0].second times
/// op
///   (Ops[1].first op Ops[1].first op ... Ops[1].first)  <- Ops[1].second times
/// op
///   ...
/// op
///   (Ops[N].first op Ops[N].first op ... Ops[N].first)  <- Ops[N].second times
///
/// Note that the values Ops[0].first, ..., Ops[N].first are all distinct.
///
/// This routine may modify the function, in which case it returns 'true'.  The
/// changes it makes may well be destructive, changing the value computed by 'I'
/// to something completely different.  Thus if the routine returns 'true' then
/// you MUST either replace I with a new expression computed from the Ops array,
/// or use RewriteExprTree to put the values back in.
///
/// A leaf node is either not a binary operation of the same kind as the root
/// node 'I' (i.e. is not a binary operator at all, or is, but with a different
/// opcode), or is the same kind of binary operator but has a use which either
/// does not belong to the expression, or does belong to the expression but is
/// a leaf node.  Every leaf node has at least one use that is a non-leaf node
/// of the expression, while for non-leaf nodes (except for the root 'I') every
/// use is a non-leaf node of the expression.
///
/// For example:
///           expression graph        node names
///
///                     +        |        I
///                    / \       |
///                   +   +      |      A,  B
///                  / \ / \     |
///                 *   +   *    |    C,  D,  E
///                / \ / \ / \   |
///                   +   *      |      F,  G
///
/// The leaf nodes are C, E, F and G.  The Ops array will contain (maybe not in
/// that order) (C, 1), (E, 1), (F, 2), (G, 2).
///
/// The expression is maximal: if some instruction is a binary operator of the
/// same kind as 'I', and all of its uses are non-leaf nodes of the expression,
/// then the instruction also belongs to the expression, is not a leaf node of
/// it, and its operands also belong to the expression (but may be leaf nodes).
///
/// NOTE: This routine will set operands of non-leaf non-root nodes to undef in
/// order to ensure that every non-root node in the expression has *exactly one*
/// use by a non-leaf node of the expression.  This destruction means that the
/// caller MUST either replace 'I' with a new expression or use something like
/// RewriteExprTree to put the values back in if the routine indicates that it
/// made a change by returning 'true'.
///
/// In the above example either the right operand of A or the left operand of B
/// will be replaced by undef.  If it is B's operand then this gives:
///
///                     +        |        I
///                    / \       |
///                   +   +      |      A,  B - operand of B replaced with undef
///                  / \   \     |
///                 *   +   *    |    C,  D,  E
///                / \ / \ / \   |
///                   +   *      |      F,  G
///
/// Note that such undef operands can only be reached by passing through 'I'.
/// For example, if you visit operands recursively starting from a leaf node
/// then you will never see such an undef operand unless you get back to 'I',
/// which requires passing through a phi node.
///
/// Note that this routine may also mutate binary operators of the wrong type
/// that have all uses inside the expression (i.e. only used by non-leaf nodes
/// of the expression) if it can turn them into binary operators of the right
/// type and thus make the expression bigger.

static bool LinearizeExprTree(BinaryOperator *I,
                              SmallVectorImpl<RepeatedValue> &Ops) {
  DEBUG(dbgs() << "LINEARIZE: " << *I << '\n');
  unsigned Bitwidth = I->getType()->getScalarType()->getPrimitiveSizeInBits();
  unsigned Opcode = I->getOpcode();
  assert(I->isAssociative() && I->isCommutative() &&
         "Expected an associative and commutative operation!");

  // Visit all operands of the expression, keeping track of their weight (the
  // number of paths from the expression root to the operand, or if you like
  // the number of times that operand occurs in the linearized expression).
  // For example, if I = X + A, where X = A + B, then I, X and B have weight 1
  // while A has weight two.

  // Worklist of non-leaf nodes (their operands are in the expression too) along
  // with their weights, representing a certain number of paths to the operator.
  // If an operator occurs in the worklist multiple times then we found multiple
  // ways to get to it.
  SmallVector<std::pair<BinaryOperator *, APInt>, 8> Worklist; // (Op, Weight)
  Worklist.push_back(std::make_pair(I, APInt(Bitwidth, 1)));
  bool Changed = false;

  // Leaves of the expression are values that either aren't the right kind of
  // operation (eg: a constant, or a multiply in an add tree), or are, but have
  // some uses that are not inside the expression.  For example, in I = X + X,
  // X = A + B, the value X has two uses (by I) that are in the expression.  If
  // X has any other uses, for example in a return instruction, then we consider
  // X to be a leaf, and won't analyze it further.  When we first visit a value,
  // if it has more than one use then at first we conservatively consider it to
  // be a leaf.  Later, as the expression is explored, we may discover some more
  // uses of the value from inside the expression.  If all uses turn out to be
  // from within the expression (and the value is a binary operator of the right
  // kind) then the value is no longer considered to be a leaf, and its operands
  // are explored.

  // Leaves - Keeps track of the set of putative leaves as well as the number of
  // paths to each leaf seen so far.
  typedef DenseMap<Value *, APInt> LeafMap;
  LeafMap Leaves;                    // Leaf -> Total weight so far.
  SmallVector<Value *, 8> LeafOrder; // Ensure deterministic leaf output order.

#ifndef NDEBUG
  SmallPtrSet<Value *, 8> Visited; // For sanity checking the iteration scheme.
#endif
  while (!Worklist.empty()) {
    PassPrediction::PassPeeper(__FILE__, 3659); // while
    std::pair<BinaryOperator *, APInt> P = Worklist.pop_back_val();
    I = P.first; // We examine the operands of this binary operator.

    for (unsigned OpIdx = 0; OpIdx < 2; ++OpIdx) { // Visit operands.
      PassPrediction::PassPeeper(__FILE__, 3660);  // for
      Value *Op = I->getOperand(OpIdx);
      APInt Weight = P.second; // Number of paths to this operand.
      DEBUG(dbgs() << "OPERAND: " << *Op << " (" << Weight << ")\n");
      assert(!Op->use_empty() && "No uses, so how did we get to it?!");

      // If this is a binary operation of the right kind with only one use then
      // add its operands to the expression.
      if (BinaryOperator *BO = isReassociableOp(Op, Opcode)) {
        assert(Visited.insert(Op).second && "Not first visit!");
        DEBUG(dbgs() << "DIRECT ADD: " << *Op << " (" << Weight << ")\n");
        Worklist.push_back(std::make_pair(BO, Weight));
        continue;
      }

      // Appears to be a leaf.  Is the operand already in the set of leaves?
      LeafMap::iterator It = Leaves.find(Op);
      if (It == Leaves.end()) {
        // Not in the leaf map.  Must be the first time we saw this operand.
        assert(Visited.insert(Op).second && "Not first visit!");
        if (!Op->hasOneUse()) {
          // This value has uses not accounted for by the expression, so it is
          // not safe to modify.  Mark it as being a leaf.
          DEBUG(dbgs() << "ADD USES LEAF: " << *Op << " (" << Weight << ")\n");
          LeafOrder.push_back(Op);
          Leaves[Op] = Weight;
          continue;
        }
        // No uses outside the expression, try morphing it.
      } else {
        // Already in the leaf map.
        assert(It != Leaves.end() && Visited.count(Op) &&
               "In leaf map but not visited!");

        // Update the number of paths to the leaf.
        IncorporateWeight(It->second, Weight, Opcode);

#if 0 // TODO: Re-enable once PR13021 is fixed.
      // The leaf already has one use from inside the expression.  As we want
      // exactly one such use, drop this new use of the leaf.
        assert(!Op->hasOneUse() && "Only one use, but we got here twice!");
        I->setOperand(OpIdx, UndefValue::get(I->getType()));
        Changed = true;

        // If the leaf is a binary operation of the right kind and we now see
        // that its multiple original uses were in fact all by nodes belonging
        // to the expression, then no longer consider it to be a leaf and add
        // its operands to the expression.
        if (BinaryOperator *BO = isReassociableOp(Op, Opcode)) {
          DEBUG(dbgs() << "UNLEAF: " << *Op << " (" << It->second << ")\n");
          Worklist.push_back(std::make_pair(BO, It->second));
          Leaves.erase(It);
          continue;
        }
#endif

        // If we still have uses that are not accounted for by the expression
        // then it is not safe to modify the value.
        if (!Op->hasOneUse()) {
          PassPrediction::PassPeeper(__FILE__, 3661); // if
          continue;
        }

        // No uses outside the expression, try morphing it.
        Weight = It->second;
        Leaves.erase(It); // Since the value may be morphed below.
      }

      // At this point we have a value which, first of all, is not a binary
      // expression of the right kind, and secondly, is only used inside the
      // expression.  This means that it can safely be modified.  See if we
      // can usefully morph it into an expression of the right kind.
      assert((!isa<Instruction>(Op) ||
              cast<Instruction>(Op)->getOpcode() != Opcode ||
              (isa<FPMathOperator>(Op) &&
               !cast<Instruction>(Op)->hasUnsafeAlgebra())) &&
             "Should have been handled above!");
      assert(Op->hasOneUse() && "Has uses outside the expression tree!");

      // If this is a multiply expression, turn any internal negations into
      // multiplies by -1 so they can be reassociated.
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op)) {
        PassPrediction::PassPeeper(__FILE__, 3662); // if
        if ((Opcode == Instruction::Mul && BinaryOperator::isNeg(BO)) ||
            (Opcode == Instruction::FMul && BinaryOperator::isFNeg(BO))) {
          DEBUG(dbgs() << "MORPH LEAF: " << *Op << " (" << Weight << ") TO ");
          BO = LowerNegateToMultiply(BO);
          DEBUG(dbgs() << *BO << '\n');
          Worklist.push_back(std::make_pair(BO, Weight));
          Changed = true;
          continue;
        }
      }

      // Failed to morph into an expression of the right type.  This really is
      // a leaf.
      DEBUG(dbgs() << "ADD LEAF: " << *Op << " (" << Weight << ")\n");
      assert(!isReassociableOp(Op, Opcode) && "Value was morphed?");
      LeafOrder.push_back(Op);
      Leaves[Op] = Weight;
    }
  }

  // The leaves, repeated according to their weights, represent the linearized
  // form of the expression.
  for (unsigned i = 0, e = LeafOrder.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3663); // for
    Value *V = LeafOrder[i];
    LeafMap::iterator It = Leaves.find(V);
    if (It == Leaves.end()) {
      // Node initially thought to be a leaf wasn't.
      PassPrediction::PassPeeper(__FILE__, 3664); // if
      continue;
    }
    assert(!isReassociableOp(V, Opcode) && "Shouldn't be a leaf!");
    APInt Weight = It->second;
    if (Weight.isMinValue()) {
      // Leaf already output or weight reduction eliminated it.
      PassPrediction::PassPeeper(__FILE__, 3665); // if
      continue;
    }
    // Ensure the leaf is only output once.
    It->second = 0;
    Ops.push_back(std::make_pair(V, Weight));
  }

  // For nilpotent operations or addition there may be no operands, for example
  // because the expression was "X xor X" or consisted of 2^Bitwidth additions:
  // in both cases the weight reduces to 0 causing the value to be skipped.
  if (Ops.empty()) {
    PassPrediction::PassPeeper(__FILE__, 3666); // if
    Constant *Identity = ConstantExpr::getBinOpIdentity(Opcode, I->getType());
    assert(Identity && "Associative operation without identity!");
    Ops.emplace_back(Identity, APInt(Bitwidth, 1));
  }

  return Changed;
}

/// Now that the operands for this expression tree are
/// linearized and optimized, emit them in-order.
void ReassociatePass::RewriteExprTree(BinaryOperator *I,
                                      SmallVectorImpl<ValueEntry> &Ops) {
  assert(Ops.size() > 1 && "Single values should be used directly!");

  // Since our optimizations should never increase the number of operations, the
  // new expression can usually be written reusing the existing binary operators
  // from the original expression tree, without creating any new instructions,
  // though the rewritten expression may have a completely different topology.
  // We take care to not change anything if the new expression will be the same
  // as the original.  If more than trivial changes (like commuting operands)
  // were made then we are obliged to clear out any optional subclass data like
  // nsw flags.

  /// NodesToRewrite - Nodes from the original expression available for writing
  /// the new expression into.
  SmallVector<BinaryOperator *, 8> NodesToRewrite;
  unsigned Opcode = I->getOpcode();
  BinaryOperator *Op = I;

  /// NotRewritable - The operands being written will be the leaves of the new
  /// expression and must not be used as inner nodes (via NodesToRewrite) by
  /// mistake.  Inner nodes are always reassociable, and usually leaves are not
  /// (if they were they would have been incorporated into the expression and so
  /// would not be leaves), so most of the time there is no danger of this.  But
  /// in rare cases a leaf may become reassociable if an optimization kills uses
  /// of it, or it may momentarily become reassociable during rewriting (below)
  /// due it being removed as an operand of one of its uses.  Ensure that misuse
  /// of leaf nodes as inner nodes cannot occur by remembering all of the future
  /// leaves and refusing to reuse any of them as inner nodes.
  SmallPtrSet<Value *, 8> NotRewritable;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3667); // for
    NotRewritable.insert(Ops[i].Op);
  }

  // ExpressionChanged - Non-null if the rewritten expression differs from the
  // original in some non-trivial way, requiring the clearing of optional flags.
  // Flags are cleared from the operator in ExpressionChanged up to I inclusive.
  BinaryOperator *ExpressionChanged = nullptr;
  for (unsigned i = 0;; ++i) {
    // The last operation (which comes earliest in the IR) is special as both
    // operands will come from Ops, rather than just one with the other being
    // a subexpression.
    PassPrediction::PassPeeper(__FILE__, 3668); // for
    if (i + 2 == Ops.size()) {
      PassPrediction::PassPeeper(__FILE__, 3669); // if
      Value *NewLHS = Ops[i].Op;
      Value *NewRHS = Ops[i + 1].Op;
      Value *OldLHS = Op->getOperand(0);
      Value *OldRHS = Op->getOperand(1);

      if (NewLHS == OldLHS && NewRHS == OldRHS) {
        // Nothing changed, leave it alone.
        PassPrediction::PassPeeper(__FILE__, 3670); // if
        break;
      }

      if (NewLHS == OldRHS && NewRHS == OldLHS) {
        // The order of the operands was reversed.  Swap them.
        DEBUG(dbgs() << "RA: " << *Op << '\n');
        Op->swapOperands();
        DEBUG(dbgs() << "TO: " << *Op << '\n');
        MadeChange = true;
        ++NumChanged;
        PassPrediction::PassPeeper(__FILE__, 3671); // break
        break;
      }

      // The new operation differs non-trivially from the original. Overwrite
      // the old operands with the new ones.
      DEBUG(dbgs() << "RA: " << *Op << '\n');
      if (NewLHS != OldLHS) {
        PassPrediction::PassPeeper(__FILE__, 3672); // if
        BinaryOperator *BO = isReassociableOp(OldLHS, Opcode);
        if (BO && !NotRewritable.count(BO)) {
          PassPrediction::PassPeeper(__FILE__, 3673); // if
          NodesToRewrite.push_back(BO);
        }
        Op->setOperand(0, NewLHS);
      }
      if (NewRHS != OldRHS) {
        PassPrediction::PassPeeper(__FILE__, 3674); // if
        BinaryOperator *BO = isReassociableOp(OldRHS, Opcode);
        if (BO && !NotRewritable.count(BO)) {
          PassPrediction::PassPeeper(__FILE__, 3675); // if
          NodesToRewrite.push_back(BO);
        }
        Op->setOperand(1, NewRHS);
      }
      DEBUG(dbgs() << "TO: " << *Op << '\n');

      ExpressionChanged = Op;
      MadeChange = true;
      ++NumChanged;

      PassPrediction::PassPeeper(__FILE__, 3676); // break
      break;
    }

    // Not the last operation.  The left-hand side will be a sub-expression
    // while the right-hand side will be the current element of Ops.
    Value *NewRHS = Ops[i].Op;
    if (NewRHS != Op->getOperand(1)) {
      DEBUG(dbgs() << "RA: " << *Op << '\n');
      if (NewRHS == Op->getOperand(0)) {
        // The new right-hand side was already present as the left operand.  If
        // we are lucky then swapping the operands will sort out both of them.
        PassPrediction::PassPeeper(__FILE__, 3677); // if
        Op->swapOperands();
      } else {
        // Overwrite with the new right-hand side.
        PassPrediction::PassPeeper(__FILE__, 3678); // else
        BinaryOperator *BO = isReassociableOp(Op->getOperand(1), Opcode);
        if (BO && !NotRewritable.count(BO)) {
          PassPrediction::PassPeeper(__FILE__, 3679); // if
          NodesToRewrite.push_back(BO);
        }
        Op->setOperand(1, NewRHS);
        ExpressionChanged = Op;
      }
      DEBUG(dbgs() << "TO: " << *Op << '\n');
      MadeChange = true;
      ++NumChanged;
    }

    // Now deal with the left-hand side.  If this is already an operation node
    // from the original expression then just rewrite the rest of the expression
    // into it.
    BinaryOperator *BO = isReassociableOp(Op->getOperand(0), Opcode);
    if (BO && !NotRewritable.count(BO)) {
      PassPrediction::PassPeeper(__FILE__, 3680); // if
      Op = BO;
      continue;
    }

    // Otherwise, grab a spare node from the original expression and use that as
    // the left-hand side.  If there are no nodes left then the optimizers made
    // an expression with more nodes than the original!  This usually means that
    // they did something stupid but it might mean that the problem was just too
    // hard (finding the mimimal number of multiplications needed to realize a
    // multiplication expression is NP-complete).  Whatever the reason, smart or
    // stupid, create a new node if there are none left.
    BinaryOperator *NewOp;
    if (NodesToRewrite.empty()) {
      PassPrediction::PassPeeper(__FILE__, 3681); // if
      Constant *Undef = UndefValue::get(I->getType());
      NewOp = BinaryOperator::Create(Instruction::BinaryOps(Opcode), Undef,
                                     Undef, "", I);
      if (NewOp->getType()->isFPOrFPVectorTy()) {
        PassPrediction::PassPeeper(__FILE__, 3683); // if
        NewOp->setFastMathFlags(I->getFastMathFlags());
      }
    } else {
      PassPrediction::PassPeeper(__FILE__, 3682); // else
      NewOp = NodesToRewrite.pop_back_val();
    }

    DEBUG(dbgs() << "RA: " << *Op << '\n');
    Op->setOperand(0, NewOp);
    DEBUG(dbgs() << "TO: " << *Op << '\n');
    ExpressionChanged = Op;
    MadeChange = true;
    ++NumChanged;
    Op = NewOp;
  }

  // If the expression changed non-trivially then clear out all subclass data
  // starting from the operator specified in ExpressionChanged, and compactify
  // the operators to just before the expression root to guarantee that the
  // expression tree is dominated by all of Ops.
  if (ExpressionChanged) {
    PassPrediction::PassPeeper(__FILE__, 3684); // if
    do {
      // Preserve FastMathFlags.
      PassPrediction::PassPeeper(__FILE__, 3685); // do-while
      if (isa<FPMathOperator>(I)) {
        PassPrediction::PassPeeper(__FILE__, 3686); // if
        FastMathFlags Flags = I->getFastMathFlags();
        ExpressionChanged->clearSubclassOptionalData();
        ExpressionChanged->setFastMathFlags(Flags);
      } else {
        PassPrediction::PassPeeper(__FILE__, 3687); // else
        ExpressionChanged->clearSubclassOptionalData();
      }

      if (ExpressionChanged == I) {
        PassPrediction::PassPeeper(__FILE__, 3688); // if
        break;
      }
      ExpressionChanged->moveBefore(I);
      ExpressionChanged =
          cast<BinaryOperator>(*ExpressionChanged->user_begin());
    } while (1);
  }

  // Throw away any left over nodes from the original expression.
  for (unsigned i = 0, e = NodesToRewrite.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3689); // for
    RedoInsts.insert(NodesToRewrite[i]);
  }
}

/// Insert instructions before the instruction pointed to by BI,
/// that computes the negative version of the value specified.  The negative
/// version of the value is returned, and BI is left pointing at the instruction
/// that should be processed next by the reassociation pass.
/// Also add intermediate instructions to the redo list that are modified while
/// pushing the negates through adds.  These will be revisited to see if
/// additional opportunities have been exposed.
static Value *NegateValue(Value *V, Instruction *BI,
                          SetVector<AssertingVH<Instruction>> &ToRedo) {
  if (Constant *C = dyn_cast<Constant>(V)) {
    PassPrediction::PassPeeper(__FILE__, 3690); // if
    if (C->getType()->isFPOrFPVectorTy()) {
      PassPrediction::PassPeeper(__FILE__, 3691); // if
      return ConstantExpr::getFNeg(C);
    }
    return ConstantExpr::getNeg(C);
  }

  // We are trying to expose opportunity for reassociation.  One of the things
  // that we want to do to achieve this is to push a negation as deep into an
  // expression chain as possible, to expose the add instructions.  In practice,
  // this means that we turn this:
  //   X = -(A+12+C+D)   into    X = -A + -12 + -C + -D = -12 + -A + -C + -D
  // so that later, a: Y = 12+X could get reassociated with the -12 to eliminate
  // the constants.  We assume that instcombine will clean up the mess later if
  // we introduce tons of unnecessary negation instructions.
  //
  if (BinaryOperator *I =
          isReassociableOp(V, Instruction::Add, Instruction::FAdd)) {
    // Push the negates through the add.
    PassPrediction::PassPeeper(__FILE__, 3692); // if
    I->setOperand(0, NegateValue(I->getOperand(0), BI, ToRedo));
    I->setOperand(1, NegateValue(I->getOperand(1), BI, ToRedo));
    if (I->getOpcode() == Instruction::Add) {
      PassPrediction::PassPeeper(__FILE__, 3693); // if
      I->setHasNoUnsignedWrap(false);
      I->setHasNoSignedWrap(false);
    }

    // We must move the add instruction here, because the neg instructions do
    // not dominate the old add instruction in general.  By moving it, we are
    // assured that the neg instructions we just inserted dominate the
    // instruction we are about to insert after them.
    //
    I->moveBefore(BI);
    I->setName(I->getName() + ".neg");

    // Add the intermediate negates to the redo list as processing them later
    // could expose more reassociating opportunities.
    ToRedo.insert(I);
    return I;
  }

  // Okay, we need to materialize a negated version of V with an instruction.
  // Scan the use lists of V to see if we have one already.
  for (User *U : V->users()) {
    PassPrediction::PassPeeper(__FILE__, 3694); // for-range
    if (!BinaryOperator::isNeg(U) && !BinaryOperator::isFNeg(U)) {
      PassPrediction::PassPeeper(__FILE__, 3695); // if
      continue;
    }

    // We found one!  Now we have to make sure that the definition dominates
    // this use.  We do this by moving it to the entry block (if it is a
    // non-instruction value) or right after the definition.  These negates will
    // be zapped by reassociate later, so we don't need much finesse here.
    BinaryOperator *TheNeg = cast<BinaryOperator>(U);

    // Verify that the negate is in this function, V might be a constant expr.
    if (TheNeg->getParent()->getParent() != BI->getParent()->getParent()) {
      PassPrediction::PassPeeper(__FILE__, 3696); // if
      continue;
    }

    BasicBlock::iterator InsertPt;
    if (Instruction *InstInput = dyn_cast<Instruction>(V)) {
      PassPrediction::PassPeeper(__FILE__, 3697); // if
      if (InvokeInst *II = dyn_cast<InvokeInst>(InstInput)) {
        PassPrediction::PassPeeper(__FILE__, 3699); // if
        InsertPt = II->getNormalDest()->begin();
      } else {
        PassPrediction::PassPeeper(__FILE__, 3700); // else
        InsertPt = ++InstInput->getIterator();
      }
      while (isa<PHINode>(InsertPt)) {
        PassPrediction::PassPeeper(__FILE__, 3701); // while
        ++InsertPt;
      }
    } else {
      PassPrediction::PassPeeper(__FILE__, 3698); // else
      InsertPt = TheNeg->getParent()->getParent()->getEntryBlock().begin();
    }
    TheNeg->moveBefore(&*InsertPt);
    if (TheNeg->getOpcode() == Instruction::Sub) {
      PassPrediction::PassPeeper(__FILE__, 3702); // if
      TheNeg->setHasNoUnsignedWrap(false);
      TheNeg->setHasNoSignedWrap(false);
    } else {
      PassPrediction::PassPeeper(__FILE__, 3703); // else
      TheNeg->andIRFlags(BI);
    }
    ToRedo.insert(TheNeg);
    return TheNeg;
  }

  // Insert a 'neg' instruction that subtracts the value from zero to get the
  // negation.
  BinaryOperator *NewNeg = CreateNeg(V, V->getName() + ".neg", BI, BI);
  ToRedo.insert(NewNeg);
  return NewNeg;
}

/// Return true if we should break up this subtract of X-Y into (X + -Y).
static bool ShouldBreakUpSubtract(Instruction *Sub) {
  // If this is a negation, we can't split it up!
  if (BinaryOperator::isNeg(Sub) || BinaryOperator::isFNeg(Sub)) {
    PassPrediction::PassPeeper(__FILE__, 3704); // if
    return false;
  }

  // Don't breakup X - undef.
  if (isa<UndefValue>(Sub->getOperand(1))) {
    PassPrediction::PassPeeper(__FILE__, 3705); // if
    return false;
  }

  // Don't bother to break this up unless either the LHS is an associable add or
  // subtract or if this is only used by one.
  Value *V0 = Sub->getOperand(0);
  if (isReassociableOp(V0, Instruction::Add, Instruction::FAdd) ||
      isReassociableOp(V0, Instruction::Sub, Instruction::FSub)) {
    PassPrediction::PassPeeper(__FILE__, 3706); // if
    return true;
  }
  Value *V1 = Sub->getOperand(1);
  if (isReassociableOp(V1, Instruction::Add, Instruction::FAdd) ||
      isReassociableOp(V1, Instruction::Sub, Instruction::FSub)) {
    PassPrediction::PassPeeper(__FILE__, 3707); // if
    return true;
  }
  Value *VB = Sub->user_back();
  if (Sub->hasOneUse() &&
      (isReassociableOp(VB, Instruction::Add, Instruction::FAdd) ||
       isReassociableOp(VB, Instruction::Sub, Instruction::FSub))) {
    PassPrediction::PassPeeper(__FILE__, 3708); // if
    return true;
  }

  return false;
}

/// If we have (X-Y), and if either X is an add, or if this is only used by an
/// add, transform this into (X+(0-Y)) to promote better reassociation.
static BinaryOperator *
BreakUpSubtract(Instruction *Sub, SetVector<AssertingVH<Instruction>> &ToRedo) {
  // Convert a subtract into an add and a neg instruction. This allows sub
  // instructions to be commuted with other add instructions.
  //
  // Calculate the negative value of Operand 1 of the sub instruction,
  // and set it as the RHS of the add instruction we just made.
  //
  Value *NegVal = NegateValue(Sub->getOperand(1), Sub, ToRedo);
  BinaryOperator *New = CreateAdd(Sub->getOperand(0), NegVal, "", Sub, Sub);
  Sub->setOperand(0, Constant::getNullValue(Sub->getType())); // Drop use of op.
  Sub->setOperand(1, Constant::getNullValue(Sub->getType())); // Drop use of op.
  New->takeName(Sub);

  // Everyone now refers to the add instruction.
  Sub->replaceAllUsesWith(New);
  New->setDebugLoc(Sub->getDebugLoc());

  DEBUG(dbgs() << "Negated: " << *New << '\n');
  return New;
}

/// If this is a shift of a reassociable multiply or is used by one, change
/// this into a multiply by a constant to assist with further reassociation.
static BinaryOperator *ConvertShiftToMul(Instruction *Shl) {
  Constant *MulCst = ConstantInt::get(Shl->getType(), 1);
  MulCst = ConstantExpr::getShl(MulCst, cast<Constant>(Shl->getOperand(1)));

  BinaryOperator *Mul =
      BinaryOperator::CreateMul(Shl->getOperand(0), MulCst, "", Shl);
  Shl->setOperand(0, UndefValue::get(Shl->getType())); // Drop use of op.
  Mul->takeName(Shl);

  // Everyone now refers to the mul instruction.
  Shl->replaceAllUsesWith(Mul);
  Mul->setDebugLoc(Shl->getDebugLoc());

  // We can safely preserve the nuw flag in all cases.  It's also safe to turn a
  // nuw nsw shl into a nuw nsw mul.  However, nsw in isolation requires special
  // handling.
  bool NSW = cast<BinaryOperator>(Shl)->hasNoSignedWrap();
  bool NUW = cast<BinaryOperator>(Shl)->hasNoUnsignedWrap();
  if (NSW && NUW) {
    PassPrediction::PassPeeper(__FILE__, 3709); // if
    Mul->setHasNoSignedWrap(true);
  }
  Mul->setHasNoUnsignedWrap(NUW);
  return Mul;
}

/// Scan backwards and forwards among values with the same rank as element i
/// to see if X exists.  If X does not exist, return i.  This is useful when
/// scanning for 'x' when we see '-x' because they both get the same rank.
static unsigned FindInOperandList(const SmallVectorImpl<ValueEntry> &Ops,
                                  unsigned i, Value *X) {
  unsigned XRank = Ops[i].Rank;
  unsigned e = Ops.size();
  for (unsigned j = i + 1; j != e && Ops[j].Rank == XRank; ++j) {
    PassPrediction::PassPeeper(__FILE__, 3710); // for
    if (Ops[j].Op == X) {
      PassPrediction::PassPeeper(__FILE__, 3711); // if
      return j;
    }
    if (Instruction *I1 = dyn_cast<Instruction>(Ops[j].Op)) {
      PassPrediction::PassPeeper(__FILE__, 3712); // if
      if (Instruction *I2 = dyn_cast<Instruction>(X)) {
        PassPrediction::PassPeeper(__FILE__, 3713); // if
        if (I1->isIdenticalTo(I2)) {
          PassPrediction::PassPeeper(__FILE__, 3714); // if
          return j;
        }
      }
    }
  }
  // Scan backwards.
  for (unsigned j = i - 1; j != ~0U && Ops[j].Rank == XRank; --j) {
    PassPrediction::PassPeeper(__FILE__, 3715); // for
    if (Ops[j].Op == X) {
      PassPrediction::PassPeeper(__FILE__, 3716); // if
      return j;
    }
    if (Instruction *I1 = dyn_cast<Instruction>(Ops[j].Op)) {
      PassPrediction::PassPeeper(__FILE__, 3717); // if
      if (Instruction *I2 = dyn_cast<Instruction>(X)) {
        PassPrediction::PassPeeper(__FILE__, 3718); // if
        if (I1->isIdenticalTo(I2)) {
          PassPrediction::PassPeeper(__FILE__, 3719); // if
          return j;
        }
      }
    }
  }
  return i;
}

/// Emit a tree of add instructions, summing Ops together
/// and returning the result.  Insert the tree before I.
static Value *EmitAddTreeOfValues(Instruction *I,
                                  SmallVectorImpl<WeakTrackingVH> &Ops) {
  if (Ops.size() == 1) {
    PassPrediction::PassPeeper(__FILE__, 3720); // if
    return Ops.back();
  }

  Value *V1 = Ops.back();
  Ops.pop_back();
  Value *V2 = EmitAddTreeOfValues(I, Ops);
  return CreateAdd(V2, V1, "tmp", I, I);
}

/// If V is an expression tree that is a multiplication sequence,
/// and if this sequence contains a multiply by Factor,
/// remove Factor from the tree and return the new tree.
Value *ReassociatePass::RemoveFactorFromExpression(Value *V, Value *Factor) {
  BinaryOperator *BO = isReassociableOp(V, Instruction::Mul, Instruction::FMul);
  if (!BO) {
    PassPrediction::PassPeeper(__FILE__, 3721); // if
    return nullptr;
  }

  SmallVector<RepeatedValue, 8> Tree;
  MadeChange |= LinearizeExprTree(BO, Tree);
  SmallVector<ValueEntry, 8> Factors;
  Factors.reserve(Tree.size());
  for (unsigned i = 0, e = Tree.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3722); // for
    RepeatedValue E = Tree[i];
    Factors.append(E.second.getZExtValue(),
                   ValueEntry(getRank(E.first), E.first));
  }

  bool FoundFactor = false;
  bool NeedsNegate = false;
  for (unsigned i = 0, e = Factors.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3723); // for
    if (Factors[i].Op == Factor) {
      PassPrediction::PassPeeper(__FILE__, 3724); // if
      FoundFactor = true;
      Factors.erase(Factors.begin() + i);
      PassPrediction::PassPeeper(__FILE__, 3725); // break
      break;
    }

    // If this is a negative version of this factor, remove it.
    if (ConstantInt *FC1 = dyn_cast<ConstantInt>(Factor)) {
      PassPrediction::PassPeeper(__FILE__, 3726); // if
      if (ConstantInt *FC2 = dyn_cast<ConstantInt>(Factors[i].Op)) {
        PassPrediction::PassPeeper(__FILE__, 3727); // if
        if (FC1->getValue() == -FC2->getValue()) {
          PassPrediction::PassPeeper(__FILE__, 3728); // if
          FoundFactor = NeedsNegate = true;
          Factors.erase(Factors.begin() + i);
          PassPrediction::PassPeeper(__FILE__, 3729); // break
          break;
        }
      }
    } else if (ConstantFP *FC1 = dyn_cast<ConstantFP>(Factor)) {
      PassPrediction::PassPeeper(__FILE__, 3730); // if
      if (ConstantFP *FC2 = dyn_cast<ConstantFP>(Factors[i].Op)) {
        PassPrediction::PassPeeper(__FILE__, 3731); // if
        const APFloat &F1 = FC1->getValueAPF();
        APFloat F2(FC2->getValueAPF());
        F2.changeSign();
        if (F1.compare(F2) == APFloat::cmpEqual) {
          PassPrediction::PassPeeper(__FILE__, 3732); // if
          FoundFactor = NeedsNegate = true;
          Factors.erase(Factors.begin() + i);
          PassPrediction::PassPeeper(__FILE__, 3733); // break
          break;
        }
      }
    }
  }

  if (!FoundFactor) {
    // Make sure to restore the operands to the expression tree.
    PassPrediction::PassPeeper(__FILE__, 3734); // if
    RewriteExprTree(BO, Factors);
    return nullptr;
  }

  BasicBlock::iterator InsertPt = ++BO->getIterator();

  // If this was just a single multiply, remove the multiply and return the only
  // remaining operand.
  if (Factors.size() == 1) {
    PassPrediction::PassPeeper(__FILE__, 3735); // if
    RedoInsts.insert(BO);
    V = Factors[0].Op;
  } else {
    PassPrediction::PassPeeper(__FILE__, 3736); // else
    RewriteExprTree(BO, Factors);
    V = BO;
  }

  if (NeedsNegate) {
    PassPrediction::PassPeeper(__FILE__, 3737); // if
    V = CreateNeg(V, "neg", &*InsertPt, BO);
  }

  return V;
}

/// If V is a single-use multiply, recursively add its operands as factors,
/// otherwise add V to the list of factors.
///
/// Ops is the top-level list of add operands we're trying to factor.
static void FindSingleUseMultiplyFactors(Value *V,
                                         SmallVectorImpl<Value *> &Factors) {
  BinaryOperator *BO = isReassociableOp(V, Instruction::Mul, Instruction::FMul);
  if (!BO) {
    PassPrediction::PassPeeper(__FILE__, 3738); // if
    Factors.push_back(V);
    return;
  }

  // Otherwise, add the LHS and RHS to the list of factors.
  FindSingleUseMultiplyFactors(BO->getOperand(1), Factors);
  FindSingleUseMultiplyFactors(BO->getOperand(0), Factors);
}

/// Optimize a series of operands to an 'and', 'or', or 'xor' instruction.
/// This optimizes based on identities.  If it can be reduced to a single Value,
/// it is returned, otherwise the Ops list is mutated as necessary.
static Value *OptimizeAndOrXor(unsigned Opcode,
                               SmallVectorImpl<ValueEntry> &Ops) {
  // Scan the operand lists looking for X and ~X pairs, along with X,X pairs.
  // If we find any, we can simplify the expression. X&~X == 0, X|~X == -1.
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    // First, check for X and ~X in the operand list.
    assert(i < Ops.size());
    if (BinaryOperator::isNot(Ops[i].Op)) {       // Cannot occur for ^.
      PassPrediction::PassPeeper(__FILE__, 3739); // if
      Value *X = BinaryOperator::getNotArgument(Ops[i].Op);
      unsigned FoundX = FindInOperandList(Ops, i, X);
      if (FoundX != i) {
        PassPrediction::PassPeeper(__FILE__, 3740);   // if
        if (Opcode == Instruction::And) {             // ...&X&~X = 0
          PassPrediction::PassPeeper(__FILE__, 3741); // if
          return Constant::getNullValue(X->getType());
        }

        if (Opcode == Instruction::Or) {              // ...|X|~X = -1
          PassPrediction::PassPeeper(__FILE__, 3742); // if
          return Constant::getAllOnesValue(X->getType());
        }
      }
    }

    // Next, check for duplicate pairs of values, which we assume are next to
    // each other, due to our sorting criteria.
    assert(i < Ops.size());
    if (i + 1 != Ops.size() && Ops[i + 1].Op == Ops[i].Op) {
      PassPrediction::PassPeeper(__FILE__, 3743); // if
      if (Opcode == Instruction::And || Opcode == Instruction::Or) {
        // Drop duplicate values for And and Or.
        PassPrediction::PassPeeper(__FILE__, 3744); // if
        Ops.erase(Ops.begin() + i);
        --i;
        --e;
        ++NumAnnihil;
        continue;
      }

      // Drop pairs of values for Xor.
      assert(Opcode == Instruction::Xor);
      if (e == 2) {
        PassPrediction::PassPeeper(__FILE__, 3745); // if
        return Constant::getNullValue(Ops[0].Op->getType());
      }

      // Y ^ X^X -> Y
      Ops.erase(Ops.begin() + i, Ops.begin() + i + 2);
      i -= 1;
      e -= 2;
      ++NumAnnihil;
    }
  }
  return nullptr;
}

/// Helper function of CombineXorOpnd(). It creates a bitwise-and
/// instruction with the given two operands, and return the resulting
/// instruction. There are two special cases: 1) if the constant operand is 0,
/// it will return NULL. 2) if the constant is ~0, the symbolic operand will
/// be returned.
static Value *createAndInstr(Instruction *InsertBefore, Value *Opnd,
                             const APInt &ConstOpnd) {
  if (ConstOpnd.isNullValue()) {
    PassPrediction::PassPeeper(__FILE__, 3746); // if
    return nullptr;
  }

  if (ConstOpnd.isAllOnesValue()) {
    PassPrediction::PassPeeper(__FILE__, 3747); // if
    return Opnd;
  }

  Instruction *I = BinaryOperator::CreateAnd(
      Opnd, ConstantInt::get(Opnd->getType(), ConstOpnd), "and.ra",
      InsertBefore);
  I->setDebugLoc(InsertBefore->getDebugLoc());
  return I;
}

// Helper function of OptimizeXor(). It tries to simplify "Opnd1 ^ ConstOpnd"
// into "R ^ C", where C would be 0, and R is a symbolic value.
//
// If it was successful, true is returned, and the "R" and "C" is returned
// via "Res" and "ConstOpnd", respectively; otherwise, false is returned,
// and both "Res" and "ConstOpnd" remain unchanged.
//
bool ReassociatePass::CombineXorOpnd(Instruction *I, XorOpnd *Opnd1,
                                     APInt &ConstOpnd, Value *&Res) {
  // Xor-Rule 1: (x | c1) ^ c2 = (x | c1) ^ (c1 ^ c1) ^ c2
  //                       = ((x | c1) ^ c1) ^ (c1 ^ c2)
  //                       = (x & ~c1) ^ (c1 ^ c2)
  // It is useful only when c1 == c2.
  if (!Opnd1->isOrExpr() || Opnd1->getConstPart().isNullValue()) {
    PassPrediction::PassPeeper(__FILE__, 3748); // if
    return false;
  }

  if (!Opnd1->getValue()->hasOneUse()) {
    PassPrediction::PassPeeper(__FILE__, 3749); // if
    return false;
  }

  const APInt &C1 = Opnd1->getConstPart();
  if (C1 != ConstOpnd) {
    PassPrediction::PassPeeper(__FILE__, 3750); // if
    return false;
  }

  Value *X = Opnd1->getSymbolicPart();
  Res = createAndInstr(I, X, ~C1);
  // ConstOpnd was C2, now C1 ^ C2.
  ConstOpnd ^= C1;

  if (Instruction *T = dyn_cast<Instruction>(Opnd1->getValue())) {
    PassPrediction::PassPeeper(__FILE__, 3751); // if
    RedoInsts.insert(T);
  }
  return true;
}

// Helper function of OptimizeXor(). It tries to simplify
// "Opnd1 ^ Opnd2 ^ ConstOpnd" into "R ^ C", where C would be 0, and R is a
// symbolic value.
//
// If it was successful, true is returned, and the "R" and "C" is returned
// via "Res" and "ConstOpnd", respectively (If the entire expression is
// evaluated to a constant, the Res is set to NULL); otherwise, false is
// returned, and both "Res" and "ConstOpnd" remain unchanged.
bool ReassociatePass::CombineXorOpnd(Instruction *I, XorOpnd *Opnd1,
                                     XorOpnd *Opnd2, APInt &ConstOpnd,
                                     Value *&Res) {
  Value *X = Opnd1->getSymbolicPart();
  if (X != Opnd2->getSymbolicPart()) {
    PassPrediction::PassPeeper(__FILE__, 3752); // if
    return false;
  }

  // This many instruction become dead.(At least "Opnd1 ^ Opnd2" will die.)
  int DeadInstNum = 1;
  if (Opnd1->getValue()->hasOneUse()) {
    PassPrediction::PassPeeper(__FILE__, 3753); // if
    DeadInstNum++;
  }
  if (Opnd2->getValue()->hasOneUse()) {
    PassPrediction::PassPeeper(__FILE__, 3754); // if
    DeadInstNum++;
  }

  // Xor-Rule 2:
  //  (x | c1) ^ (x & c2)
  //   = (x|c1) ^ (x&c2) ^ (c1 ^ c1) = ((x|c1) ^ c1) ^ (x & c2) ^ c1
  //   = (x & ~c1) ^ (x & c2) ^ c1               // Xor-Rule 1
  //   = (x & c3) ^ c1, where c3 = ~c1 ^ c2      // Xor-rule 3
  //
  if (Opnd1->isOrExpr() != Opnd2->isOrExpr()) {
    PassPrediction::PassPeeper(__FILE__, 3755); // if
    if (Opnd2->isOrExpr()) {
      PassPrediction::PassPeeper(__FILE__, 3756); // if
      std::swap(Opnd1, Opnd2);
    }

    const APInt &C1 = Opnd1->getConstPart();
    const APInt &C2 = Opnd2->getConstPart();
    APInt C3((~C1) ^ C2);

    // Do not increase code size!
    if (!C3.isNullValue() && !C3.isAllOnesValue()) {
      PassPrediction::PassPeeper(__FILE__, 3757); // if
      int NewInstNum = ConstOpnd.getBoolValue() ? 1 : 2;
      if (NewInstNum > DeadInstNum) {
        PassPrediction::PassPeeper(__FILE__, 3758); // if
        return false;
      }
    }

    Res = createAndInstr(I, X, C3);
    ConstOpnd ^= C1;

  } else if (Opnd1->isOrExpr()) {
    // Xor-Rule 3: (x | c1) ^ (x | c2) = (x & c3) ^ c3 where c3 = c1 ^ c2
    //
    PassPrediction::PassPeeper(__FILE__, 3759); // if
    const APInt &C1 = Opnd1->getConstPart();
    const APInt &C2 = Opnd2->getConstPart();
    APInt C3 = C1 ^ C2;

    // Do not increase code size
    if (!C3.isNullValue() && !C3.isAllOnesValue()) {
      PassPrediction::PassPeeper(__FILE__, 3761); // if
      int NewInstNum = ConstOpnd.getBoolValue() ? 1 : 2;
      if (NewInstNum > DeadInstNum) {
        PassPrediction::PassPeeper(__FILE__, 3762); // if
        return false;
      }
    }

    Res = createAndInstr(I, X, C3);
    ConstOpnd ^= C3;
  } else {
    // Xor-Rule 4: (x & c1) ^ (x & c2) = (x & (c1^c2))
    //
    PassPrediction::PassPeeper(__FILE__, 3760); // else
    const APInt &C1 = Opnd1->getConstPart();
    const APInt &C2 = Opnd2->getConstPart();
    APInt C3 = C1 ^ C2;
    Res = createAndInstr(I, X, C3);
  }

  // Put the original operands in the Redo list; hope they will be deleted
  // as dead code.
  if (Instruction *T = dyn_cast<Instruction>(Opnd1->getValue())) {
    PassPrediction::PassPeeper(__FILE__, 3763); // if
    RedoInsts.insert(T);
  }
  if (Instruction *T = dyn_cast<Instruction>(Opnd2->getValue())) {
    PassPrediction::PassPeeper(__FILE__, 3764); // if
    RedoInsts.insert(T);
  }

  return true;
}

/// Optimize a series of operands to an 'xor' instruction. If it can be reduced
/// to a single Value, it is returned, otherwise the Ops list is mutated as
/// necessary.
Value *ReassociatePass::OptimizeXor(Instruction *I,
                                    SmallVectorImpl<ValueEntry> &Ops) {
  if (Value *V = OptimizeAndOrXor(Instruction::Xor, Ops)) {
    PassPrediction::PassPeeper(__FILE__, 3765); // if
    return V;
  }

  if (Ops.size() == 1) {
    PassPrediction::PassPeeper(__FILE__, 3766); // if
    return nullptr;
  }

  SmallVector<XorOpnd, 8> Opnds;
  SmallVector<XorOpnd *, 8> OpndPtrs;
  Type *Ty = Ops[0].Op->getType();
  APInt ConstOpnd(Ty->getScalarSizeInBits(), 0);

  // Step 1: Convert ValueEntry to XorOpnd
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3767); // for
    Value *V = Ops[i].Op;
    const APInt *C;
    // TODO: Support non-splat vectors.
    if (match(V, PatternMatch::m_APInt(C))) {
      PassPrediction::PassPeeper(__FILE__, 3768); // if
      ConstOpnd ^= *C;
    } else {
      PassPrediction::PassPeeper(__FILE__, 3769); // else
      XorOpnd O(V);
      O.setSymbolicRank(getRank(O.getSymbolicPart()));
      Opnds.push_back(O);
    }
  }

  // NOTE: From this point on, do *NOT* add/delete element to/from "Opnds".
  //  It would otherwise invalidate the "Opnds"'s iterator, and hence invalidate
  //  the "OpndPtrs" as well. For the similar reason, do not fuse this loop
  //  with the previous loop --- the iterator of the "Opnds" may be invalidated
  //  when new elements are added to the vector.
  for (unsigned i = 0, e = Opnds.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3770); // for
    OpndPtrs.push_back(&Opnds[i]);
  }

  // Step 2: Sort the Xor-Operands in a way such that the operands containing
  //  the same symbolic value cluster together. For instance, the input operand
  //  sequence ("x | 123", "y & 456", "x & 789") will be sorted into:
  //  ("x | 123", "x & 789", "y & 456").
  //
  //  The purpose is twofold:
  //  1) Cluster together the operands sharing the same symbolic-value.
  //  2) Operand having smaller symbolic-value-rank is permuted earlier, which
  //     could potentially shorten crital path, and expose more loop-invariants.
  //     Note that values' rank are basically defined in RPO order (FIXME).
  //     So, if Rank(X) < Rank(Y) < Rank(Z), it means X is defined earlier
  //     than Y which is defined earlier than Z. Permute "x | 1", "Y & 2",
  //     "z" in the order of X-Y-Z is better than any other orders.
  std::stable_sort(OpndPtrs.begin(), OpndPtrs.end(),
                   [](XorOpnd *LHS, XorOpnd *RHS) {
                     return LHS->getSymbolicRank() < RHS->getSymbolicRank();
                   });

  // Step 3: Combine adjacent operands
  XorOpnd *PrevOpnd = nullptr;
  bool Changed = false;
  for (unsigned i = 0, e = Opnds.size(); i < e; i++) {
    PassPrediction::PassPeeper(__FILE__, 3771); // for
    XorOpnd *CurrOpnd = OpndPtrs[i];
    // The combined value
    Value *CV;

    // Step 3.1: Try simplifying "CurrOpnd ^ ConstOpnd"
    if (!ConstOpnd.isNullValue() &&
        CombineXorOpnd(I, CurrOpnd, ConstOpnd, CV)) {
      PassPrediction::PassPeeper(__FILE__, 3772); // if
      Changed = true;
      if (CV) {
        PassPrediction::PassPeeper(__FILE__, 3773); // if
        *CurrOpnd = XorOpnd(CV);
      } else {
        PassPrediction::PassPeeper(__FILE__, 3774); // else
        CurrOpnd->Invalidate();
        continue;
      }
    }

    if (!PrevOpnd ||
        CurrOpnd->getSymbolicPart() != PrevOpnd->getSymbolicPart()) {
      PassPrediction::PassPeeper(__FILE__, 3775); // if
      PrevOpnd = CurrOpnd;
      continue;
    }

    // step 3.2: When previous and current operands share the same symbolic
    //  value, try to simplify "PrevOpnd ^ CurrOpnd ^ ConstOpnd"
    //
    if (CombineXorOpnd(I, CurrOpnd, PrevOpnd, ConstOpnd, CV)) {
      // Remove previous operand
      PassPrediction::PassPeeper(__FILE__, 3776); // if
      PrevOpnd->Invalidate();
      if (CV) {
        PassPrediction::PassPeeper(__FILE__, 3777); // if
        *CurrOpnd = XorOpnd(CV);
        PrevOpnd = CurrOpnd;
      } else {
        PassPrediction::PassPeeper(__FILE__, 3778); // else
        CurrOpnd->Invalidate();
        PrevOpnd = nullptr;
      }
      Changed = true;
    }
  }

  // Step 4: Reassemble the Ops
  if (Changed) {
    PassPrediction::PassPeeper(__FILE__, 3779); // if
    Ops.clear();
    for (unsigned int i = 0, e = Opnds.size(); i < e; i++) {
      PassPrediction::PassPeeper(__FILE__, 3780); // for
      XorOpnd &O = Opnds[i];
      if (O.isInvalid()) {
        PassPrediction::PassPeeper(__FILE__, 3781); // if
        continue;
      }
      ValueEntry VE(getRank(O.getValue()), O.getValue());
      Ops.push_back(VE);
    }
    if (!ConstOpnd.isNullValue()) {
      PassPrediction::PassPeeper(__FILE__, 3782); // if
      Value *C = ConstantInt::get(Ty, ConstOpnd);
      ValueEntry VE(getRank(C), C);
      Ops.push_back(VE);
    }
    unsigned Sz = Ops.size();
    if (Sz == 1) {
      PassPrediction::PassPeeper(__FILE__, 3783); // if
      return Ops.back().Op;
    }
    if (Sz == 0) {
      assert(ConstOpnd.isNullValue());
      return ConstantInt::get(Ty, ConstOpnd);
    }
  }

  return nullptr;
}

/// Optimize a series of operands to an 'add' instruction.  This
/// optimizes based on identities.  If it can be reduced to a single Value, it
/// is returned, otherwise the Ops list is mutated as necessary.
Value *ReassociatePass::OptimizeAdd(Instruction *I,
                                    SmallVectorImpl<ValueEntry> &Ops) {
  // Scan the operand lists looking for X and -X pairs.  If we find any, we
  // can simplify expressions like X+-X == 0 and X+~X ==-1.  While we're at it,
  // scan for any
  // duplicates.  We want to canonicalize Y+Y+Y+Z -> 3*Y+Z.

  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3784); // for
    Value *TheOp = Ops[i].Op;
    // Check to see if we've seen this operand before.  If so, we factor all
    // instances of the operand together.  Due to our sorting criteria, we know
    // that these need to be next to each other in the vector.
    if (i + 1 != Ops.size() && Ops[i + 1].Op == TheOp) {
      // Rescan the list, remove all instances of this operand from the expr.
      PassPrediction::PassPeeper(__FILE__, 3785); // if
      unsigned NumFound = 0;
      do {
        PassPrediction::PassPeeper(__FILE__, 3786); // do-while
        Ops.erase(Ops.begin() + i);
        ++NumFound;
      } while (i != Ops.size() && Ops[i].Op == TheOp);

      DEBUG(dbgs() << "\nFACTORING [" << NumFound << "]: " << *TheOp << '\n');
      ++NumFactor;

      // Insert a new multiply.
      Type *Ty = TheOp->getType();
      Constant *C = Ty->isIntOrIntVectorTy() ? ConstantInt::get(Ty, NumFound)
                                             : ConstantFP::get(Ty, NumFound);
      Instruction *Mul = CreateMul(TheOp, C, "factor", I, I);

      // Now that we have inserted a multiply, optimize it. This allows us to
      // handle cases that require multiple factoring steps, such as this:
      // (X*2) + (X*2) + (X*2) -> (X*2)*3 -> X*6
      RedoInsts.insert(Mul);

      // If every add operand was a duplicate, return the multiply.
      if (Ops.empty()) {
        PassPrediction::PassPeeper(__FILE__, 3787); // if
        return Mul;
      }

      // Otherwise, we had some input that didn't have the dupe, such as
      // "A + A + B" -> "A*2 + B".  Add the new multiply to the list of
      // things being added by this operation.
      Ops.insert(Ops.begin(), ValueEntry(getRank(Mul), Mul));

      --i;
      e = Ops.size();
      continue;
    }

    // Check for X and -X or X and ~X in the operand list.
    if (!BinaryOperator::isNeg(TheOp) && !BinaryOperator::isFNeg(TheOp) &&
        !BinaryOperator::isNot(TheOp)) {
      PassPrediction::PassPeeper(__FILE__, 3788); // if
      continue;
    }

    Value *X = nullptr;
    if (BinaryOperator::isNeg(TheOp) || BinaryOperator::isFNeg(TheOp)) {
      PassPrediction::PassPeeper(__FILE__, 3789); // if
      X = BinaryOperator::getNegArgument(TheOp);
    } else if (BinaryOperator::isNot(TheOp)) {
      PassPrediction::PassPeeper(__FILE__, 3790); // if
      X = BinaryOperator::getNotArgument(TheOp);
    }

    unsigned FoundX = FindInOperandList(Ops, i, X);
    if (FoundX == i) {
      PassPrediction::PassPeeper(__FILE__, 3791); // if
      continue;
    }

    // Remove X and -X from the operand list.
    if (Ops.size() == 2 &&
        (BinaryOperator::isNeg(TheOp) || BinaryOperator::isFNeg(TheOp))) {
      PassPrediction::PassPeeper(__FILE__, 3792); // if
      return Constant::getNullValue(X->getType());
    }

    // Remove X and ~X from the operand list.
    if (Ops.size() == 2 && BinaryOperator::isNot(TheOp)) {
      PassPrediction::PassPeeper(__FILE__, 3793); // if
      return Constant::getAllOnesValue(X->getType());
    }

    Ops.erase(Ops.begin() + i);
    if (i < FoundX) {
      PassPrediction::PassPeeper(__FILE__, 3794); // if
      --FoundX;
    } else {
      PassPrediction::PassPeeper(__FILE__, 3795); // else
      --i; // Need to back up an extra one.
    }
    Ops.erase(Ops.begin() + FoundX);
    ++NumAnnihil;
    --i;    // Revisit element.
    e -= 2; // Removed two elements.

    // if X and ~X we append -1 to the operand list.
    if (BinaryOperator::isNot(TheOp)) {
      PassPrediction::PassPeeper(__FILE__, 3796); // if
      Value *V = Constant::getAllOnesValue(X->getType());
      Ops.insert(Ops.end(), ValueEntry(getRank(V), V));
      e += 1;
    }
  }

  // Scan the operand list, checking to see if there are any common factors
  // between operands.  Consider something like A*A+A*B*C+D.  We would like to
  // reassociate this to A*(A+B*C)+D, which reduces the number of multiplies.
  // To efficiently find this, we count the number of times a factor occurs
  // for any ADD operands that are MULs.
  DenseMap<Value *, unsigned> FactorOccurrences;

  // Keep track of each multiply we see, to avoid triggering on (X*4)+(X*4)
  // where they are actually the same multiply.
  unsigned MaxOcc = 0;
  Value *MaxOccVal = nullptr;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3797); // for
    BinaryOperator *BOp =
        isReassociableOp(Ops[i].Op, Instruction::Mul, Instruction::FMul);
    if (!BOp) {
      PassPrediction::PassPeeper(__FILE__, 3798); // if
      continue;
    }

    // Compute all of the factors of this added value.
    SmallVector<Value *, 8> Factors;
    FindSingleUseMultiplyFactors(BOp, Factors);
    assert(Factors.size() > 1 && "Bad linearize!");

    // Add one to FactorOccurrences for each unique factor in this op.
    SmallPtrSet<Value *, 8> Duplicates;
    for (unsigned i = 0, e = Factors.size(); i != e; ++i) {
      PassPrediction::PassPeeper(__FILE__, 3799); // for
      Value *Factor = Factors[i];
      if (!Duplicates.insert(Factor).second) {
        PassPrediction::PassPeeper(__FILE__, 3800); // if
        continue;
      }

      unsigned Occ = ++FactorOccurrences[Factor];
      if (Occ > MaxOcc) {
        PassPrediction::PassPeeper(__FILE__, 3801); // if
        MaxOcc = Occ;
        MaxOccVal = Factor;
      }

      // If Factor is a negative constant, add the negated value as a factor
      // because we can percolate the negate out.  Watch for minint, which
      // cannot be positivified.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Factor)) {
        PassPrediction::PassPeeper(__FILE__, 3802); // if
        if (CI->isNegative() && !CI->isMinValue(true)) {
          PassPrediction::PassPeeper(__FILE__, 3803); // if
          Factor = ConstantInt::get(CI->getContext(), -CI->getValue());
          if (!Duplicates.insert(Factor).second) {
            PassPrediction::PassPeeper(__FILE__, 3804); // if
            continue;
          }
          unsigned Occ = ++FactorOccurrences[Factor];
          if (Occ > MaxOcc) {
            PassPrediction::PassPeeper(__FILE__, 3805); // if
            MaxOcc = Occ;
            MaxOccVal = Factor;
          }
        }
      } else if (ConstantFP *CF = dyn_cast<ConstantFP>(Factor)) {
        PassPrediction::PassPeeper(__FILE__, 3806); // if
        if (CF->isNegative()) {
          PassPrediction::PassPeeper(__FILE__, 3807); // if
          APFloat F(CF->getValueAPF());
          F.changeSign();
          Factor = ConstantFP::get(CF->getContext(), F);
          if (!Duplicates.insert(Factor).second) {
            PassPrediction::PassPeeper(__FILE__, 3808); // if
            continue;
          }
          unsigned Occ = ++FactorOccurrences[Factor];
          if (Occ > MaxOcc) {
            PassPrediction::PassPeeper(__FILE__, 3809); // if
            MaxOcc = Occ;
            MaxOccVal = Factor;
          }
        }
      }
    }
  }

  // If any factor occurred more than one time, we can pull it out.
  if (MaxOcc > 1) {
    DEBUG(dbgs() << "\nFACTORING [" << MaxOcc << "]: " << *MaxOccVal << '\n');
    ++NumFactor;

    // Create a new instruction that uses the MaxOccVal twice.  If we don't do
    // this, we could otherwise run into situations where removing a factor
    // from an expression will drop a use of maxocc, and this can cause
    // RemoveFactorFromExpression on successive values to behave differently.
    Instruction *DummyInst =
        I->getType()->isIntOrIntVectorTy()
            ? BinaryOperator::CreateAdd(MaxOccVal, MaxOccVal)
            : BinaryOperator::CreateFAdd(MaxOccVal, MaxOccVal);

    SmallVector<WeakTrackingVH, 4> NewMulOps;
    for (unsigned i = 0; i != Ops.size(); ++i) {
      // Only try to remove factors from expressions we're allowed to.
      PassPrediction::PassPeeper(__FILE__, 3810); // for
      BinaryOperator *BOp =
          isReassociableOp(Ops[i].Op, Instruction::Mul, Instruction::FMul);
      if (!BOp) {
        PassPrediction::PassPeeper(__FILE__, 3811); // if
        continue;
      }

      if (Value *V = RemoveFactorFromExpression(Ops[i].Op, MaxOccVal)) {
        // The factorized operand may occur several times.  Convert them all in
        // one fell swoop.
        PassPrediction::PassPeeper(__FILE__, 3812); // if
        for (unsigned j = Ops.size(); j != i;) {
          PassPrediction::PassPeeper(__FILE__, 3813); // for
          --j;
          if (Ops[j].Op == Ops[i].Op) {
            PassPrediction::PassPeeper(__FILE__, 3814); // if
            NewMulOps.push_back(V);
            Ops.erase(Ops.begin() + j);
          }
        }
        --i;
      }
    }

    // No need for extra uses anymore.
    DummyInst->deleteValue();

    unsigned NumAddedValues = NewMulOps.size();
    Value *V = EmitAddTreeOfValues(I, NewMulOps);

    // Now that we have inserted the add tree, optimize it. This allows us to
    // handle cases that require multiple factoring steps, such as this:
    // A*A*B + A*A*C   -->   A*(A*B+A*C)   -->   A*(A*(B+C))
    assert(NumAddedValues > 1 && "Each occurrence should contribute a value");
    (void)NumAddedValues;
    if (Instruction *VI = dyn_cast<Instruction>(V)) {
      PassPrediction::PassPeeper(__FILE__, 3815); // if
      RedoInsts.insert(VI);
    }

    // Create the multiply.
    Instruction *V2 = CreateMul(V, MaxOccVal, "tmp", I, I);

    // Rerun associate on the multiply in case the inner expression turned into
    // a multiply.  We want to make sure that we keep things in canonical form.
    RedoInsts.insert(V2);

    // If every add operand included the factor (e.g. "A*B + A*C"), then the
    // entire result expression is just the multiply "A*(B+C)".
    if (Ops.empty()) {
      PassPrediction::PassPeeper(__FILE__, 3816); // if
      return V2;
    }

    // Otherwise, we had some input that didn't have the factor, such as
    // "A*B + A*C + D" -> "A*(B+C) + D".  Add the new multiply to the list of
    // things being added by this operation.
    Ops.insert(Ops.begin(), ValueEntry(getRank(V2), V2));
  }

  return nullptr;
}

/// \brief Build up a vector of value/power pairs factoring a product.
///
/// Given a series of multiplication operands, build a vector of factors and
/// the powers each is raised to when forming the final product. Sort them in
/// the order of descending power.
///
///      (x*x)          -> [(x, 2)]
///     ((x*x)*x)       -> [(x, 3)]
///   ((((x*y)*x)*y)*x) -> [(x, 3), (y, 2)]
///
/// \returns Whether any factors have a power greater than one.
static bool collectMultiplyFactors(SmallVectorImpl<ValueEntry> &Ops,
                                   SmallVectorImpl<Factor> &Factors) {
  // FIXME: Have Ops be (ValueEntry, Multiplicity) pairs, simplifying this.
  // Compute the sum of powers of simplifiable factors.
  unsigned FactorPowerSum = 0;
  for (unsigned Idx = 1, Size = Ops.size(); Idx < Size; ++Idx) {
    PassPrediction::PassPeeper(__FILE__, 3817); // for
    Value *Op = Ops[Idx - 1].Op;

    // Count the number of occurrences of this value.
    unsigned Count = 1;
    for (; Idx < Size && Ops[Idx].Op == Op; ++Idx) {
      PassPrediction::PassPeeper(__FILE__, 3818); // for
      ++Count;
    }
    // Track for simplification all factors which occur 2 or more times.
    if (Count > 1) {
      PassPrediction::PassPeeper(__FILE__, 3819); // if
      FactorPowerSum += Count;
    }
  }

  // We can only simplify factors if the sum of the powers of our simplifiable
  // factors is 4 or higher. When that is the case, we will *always* have
  // a simplification. This is an important invariant to prevent cyclicly
  // trying to simplify already minimal formations.
  if (FactorPowerSum < 4) {
    PassPrediction::PassPeeper(__FILE__, 3820); // if
    return false;
  }

  // Now gather the simplifiable factors, removing them from Ops.
  FactorPowerSum = 0;
  for (unsigned Idx = 1; Idx < Ops.size(); ++Idx) {
    PassPrediction::PassPeeper(__FILE__, 3821); // for
    Value *Op = Ops[Idx - 1].Op;

    // Count the number of occurrences of this value.
    unsigned Count = 1;
    for (; Idx < Ops.size() && Ops[Idx].Op == Op; ++Idx) {
      PassPrediction::PassPeeper(__FILE__, 3822); // for
      ++Count;
    }
    if (Count == 1) {
      PassPrediction::PassPeeper(__FILE__, 3823); // if
      continue;
    }
    // Move an even number of occurrences to Factors.
    Count &= ~1U;
    Idx -= Count;
    FactorPowerSum += Count;
    Factors.push_back(Factor(Op, Count));
    Ops.erase(Ops.begin() + Idx, Ops.begin() + Idx + Count);
  }

  // None of the adjustments above should have reduced the sum of factor powers
  // below our mininum of '4'.
  assert(FactorPowerSum >= 4);

  std::stable_sort(Factors.begin(), Factors.end(),
                   [](const Factor &LHS, const Factor &RHS) {
                     return LHS.Power > RHS.Power;
                   });
  return true;
}

/// \brief Build a tree of multiplies, computing the product of Ops.
static Value *buildMultiplyTree(IRBuilder<> &Builder,
                                SmallVectorImpl<Value *> &Ops) {
  if (Ops.size() == 1) {
    PassPrediction::PassPeeper(__FILE__, 3824); // if
    return Ops.back();
  }

  Value *LHS = Ops.pop_back_val();
  do {
    PassPrediction::PassPeeper(__FILE__, 3825); // do-while
    if (LHS->getType()->isIntOrIntVectorTy()) {
      PassPrediction::PassPeeper(__FILE__, 3826); // if
      LHS = Builder.CreateMul(LHS, Ops.pop_back_val());
    } else {
      PassPrediction::PassPeeper(__FILE__, 3827); // else
      LHS = Builder.CreateFMul(LHS, Ops.pop_back_val());
    }
  } while (!Ops.empty());

  return LHS;
}

/// \brief Build a minimal multiplication DAG for (a^x)*(b^y)*(c^z)*...
///
/// Given a vector of values raised to various powers, where no two values are
/// equal and the powers are sorted in decreasing order, compute the minimal
/// DAG of multiplies to compute the final product, and return that product
/// value.
Value *
ReassociatePass::buildMinimalMultiplyDAG(IRBuilder<> &Builder,
                                         SmallVectorImpl<Factor> &Factors) {
  assert(Factors[0].Power);
  SmallVector<Value *, 4> OuterProduct;
  for (unsigned LastIdx = 0, Idx = 1, Size = Factors.size();
       Idx < Size && Factors[Idx].Power > 0; ++Idx) {
    PassPrediction::PassPeeper(__FILE__, 3828); // for
    if (Factors[Idx].Power != Factors[LastIdx].Power) {
      PassPrediction::PassPeeper(__FILE__, 3829); // if
      LastIdx = Idx;
      continue;
    }

    // We want to multiply across all the factors with the same power so that
    // we can raise them to that power as a single entity. Build a mini tree
    // for that.
    SmallVector<Value *, 4> InnerProduct;
    InnerProduct.push_back(Factors[LastIdx].Base);
    do {
      PassPrediction::PassPeeper(__FILE__, 3830); // do-while
      InnerProduct.push_back(Factors[Idx].Base);
      ++Idx;
    } while (Idx < Size && Factors[Idx].Power == Factors[LastIdx].Power);

    // Reset the base value of the first factor to the new expression tree.
    // We'll remove all the factors with the same power in a second pass.
    Value *M = Factors[LastIdx].Base = buildMultiplyTree(Builder, InnerProduct);
    if (Instruction *MI = dyn_cast<Instruction>(M)) {
      PassPrediction::PassPeeper(__FILE__, 3831); // if
      RedoInsts.insert(MI);
    }

    LastIdx = Idx;
  }
  // Unique factors with equal powers -- we've folded them into the first one's
  // base.
  Factors.erase(std::unique(Factors.begin(), Factors.end(),
                            [](const Factor &LHS, const Factor &RHS) {
                              return LHS.Power == RHS.Power;
                            }),
                Factors.end());

  // Iteratively collect the base of each factor with an add power into the
  // outer product, and halve each power in preparation for squaring the
  // expression.
  for (unsigned Idx = 0, Size = Factors.size(); Idx != Size; ++Idx) {
    PassPrediction::PassPeeper(__FILE__, 3832); // for
    if (Factors[Idx].Power & 1) {
      PassPrediction::PassPeeper(__FILE__, 3833); // if
      OuterProduct.push_back(Factors[Idx].Base);
    }
    Factors[Idx].Power >>= 1;
  }
  if (Factors[0].Power) {
    PassPrediction::PassPeeper(__FILE__, 3834); // if
    Value *SquareRoot = buildMinimalMultiplyDAG(Builder, Factors);
    OuterProduct.push_back(SquareRoot);
    OuterProduct.push_back(SquareRoot);
  }
  if (OuterProduct.size() == 1) {
    PassPrediction::PassPeeper(__FILE__, 3835); // if
    return OuterProduct.front();
  }

  Value *V = buildMultiplyTree(Builder, OuterProduct);
  return V;
}

Value *ReassociatePass::OptimizeMul(BinaryOperator *I,
                                    SmallVectorImpl<ValueEntry> &Ops) {
  // We can only optimize the multiplies when there is a chain of more than
  // three, such that a balanced tree might require fewer total multiplies.
  if (Ops.size() < 4) {
    PassPrediction::PassPeeper(__FILE__, 3836); // if
    return nullptr;
  }

  // Try to turn linear trees of multiplies without other uses of the
  // intermediate stages into minimal multiply DAGs with perfect sub-expression
  // re-use.
  SmallVector<Factor, 4> Factors;
  if (!collectMultiplyFactors(Ops, Factors)) {
    PassPrediction::PassPeeper(__FILE__, 3837); // if
    return nullptr; // All distinct factors, so nothing left for us to do.
  }

  IRBuilder<> Builder(I);
  // The reassociate transformation for FP operations is performed only
  // if unsafe algebra is permitted by FastMathFlags. Propagate those flags
  // to the newly generated operations.
  if (auto FPI = dyn_cast<FPMathOperator>(I)) {
    PassPrediction::PassPeeper(__FILE__, 3838); // if
    Builder.setFastMathFlags(FPI->getFastMathFlags());
  }

  Value *V = buildMinimalMultiplyDAG(Builder, Factors);
  if (Ops.empty()) {
    PassPrediction::PassPeeper(__FILE__, 3839); // if
    return V;
  }

  ValueEntry NewEntry = ValueEntry(getRank(V), V);
  Ops.insert(std::lower_bound(Ops.begin(), Ops.end(), NewEntry), NewEntry);
  return nullptr;
}

Value *ReassociatePass::OptimizeExpression(BinaryOperator *I,
                                           SmallVectorImpl<ValueEntry> &Ops) {
  // Now that we have the linearized expression tree, try to optimize it.
  // Start by folding any constants that we found.
  Constant *Cst = nullptr;
  unsigned Opcode = I->getOpcode();
  while (!Ops.empty() && isa<Constant>(Ops.back().Op)) {
    PassPrediction::PassPeeper(__FILE__, 3840); // while
    Constant *C = cast<Constant>(Ops.pop_back_val().Op);
    Cst = Cst ? ConstantExpr::get(Opcode, C, Cst) : C;
  }
  // If there was nothing but constants then we are done.
  if (Ops.empty()) {
    PassPrediction::PassPeeper(__FILE__, 3841); // if
    return Cst;
  }

  // Put the combined constant back at the end of the operand list, except if
  // there is no point.  For example, an add of 0 gets dropped here, while a
  // multiplication by zero turns the whole expression into zero.
  if (Cst && Cst != ConstantExpr::getBinOpIdentity(Opcode, I->getType())) {
    PassPrediction::PassPeeper(__FILE__, 3842); // if
    if (Cst == ConstantExpr::getBinOpAbsorber(Opcode, I->getType())) {
      PassPrediction::PassPeeper(__FILE__, 3843); // if
      return Cst;
    }
    Ops.push_back(ValueEntry(0, Cst));
  }

  if (Ops.size() == 1) {
    PassPrediction::PassPeeper(__FILE__, 3844); // if
    return Ops[0].Op;
  }

  // Handle destructive annihilation due to identities between elements in the
  // argument list here.
  unsigned NumOps = Ops.size();
  switch (Opcode) {
  default:
    PassPrediction::PassPeeper(__FILE__, 3845); // break
    break;
  case Instruction::And:
    PassPrediction::PassPeeper(__FILE__, 3846); // case

  case Instruction::Or:
    PassPrediction::PassPeeper(__FILE__, 3847); // case

    if (Value *Result = OptimizeAndOrXor(Opcode, Ops)) {
      PassPrediction::PassPeeper(__FILE__, 3848); // if
      return Result;
    }
    PassPrediction::PassPeeper(__FILE__, 3849); // break
    break;

  case Instruction::Xor:
    PassPrediction::PassPeeper(__FILE__, 3850); // case

    if (Value *Result = OptimizeXor(I, Ops)) {
      PassPrediction::PassPeeper(__FILE__, 3851); // if
      return Result;
    }
    PassPrediction::PassPeeper(__FILE__, 3852); // break
    break;

  case Instruction::Add:
    PassPrediction::PassPeeper(__FILE__, 3853); // case

  case Instruction::FAdd:
    PassPrediction::PassPeeper(__FILE__, 3854); // case

    if (Value *Result = OptimizeAdd(I, Ops)) {
      PassPrediction::PassPeeper(__FILE__, 3855); // if
      return Result;
    }
    PassPrediction::PassPeeper(__FILE__, 3856); // break
    break;

  case Instruction::Mul:
    PassPrediction::PassPeeper(__FILE__, 3857); // case

  case Instruction::FMul:
    PassPrediction::PassPeeper(__FILE__, 3858); // case

    if (Value *Result = OptimizeMul(I, Ops)) {
      PassPrediction::PassPeeper(__FILE__, 3859); // if
      return Result;
    }
    PassPrediction::PassPeeper(__FILE__, 3860); // break
    break;
  }

  if (Ops.size() != NumOps) {
    PassPrediction::PassPeeper(__FILE__, 3861); // if
    return OptimizeExpression(I, Ops);
  }
  return nullptr;
}

// Remove dead instructions and if any operands are trivially dead add them to
// Insts so they will be removed as well.
void ReassociatePass::RecursivelyEraseDeadInsts(
    Instruction *I, SetVector<AssertingVH<Instruction>> &Insts) {
  assert(isInstructionTriviallyDead(I) && "Trivially dead instructions only!");
  SmallVector<Value *, 4> Ops(I->op_begin(), I->op_end());
  ValueRankMap.erase(I);
  Insts.remove(I);
  RedoInsts.remove(I);
  I->eraseFromParent();
  for (auto Op : Ops) {
    PassPrediction::PassPeeper(__FILE__, 3862); // for-range
    if (Instruction *OpInst = dyn_cast<Instruction>(Op)) {
      PassPrediction::PassPeeper(__FILE__, 3863); // if
      if (OpInst->use_empty()) {
        PassPrediction::PassPeeper(__FILE__, 3864); // if
        Insts.insert(OpInst);
      }
    }
  }
}

/// Zap the given instruction, adding interesting operands to the work list.
void ReassociatePass::EraseInst(Instruction *I) {
  assert(isInstructionTriviallyDead(I) && "Trivially dead instructions only!");
  DEBUG(dbgs() << "Erasing dead inst: "; I->dump());

  SmallVector<Value *, 8> Ops(I->op_begin(), I->op_end());
  // Erase the dead instruction.
  ValueRankMap.erase(I);
  RedoInsts.remove(I);
  I->eraseFromParent();
  // Optimize its operands.
  SmallPtrSet<Instruction *, 8> Visited; // Detect self-referential nodes.
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3865); // for
    if (Instruction *Op = dyn_cast<Instruction>(Ops[i])) {
      // If this is a node in an expression tree, climb to the expression root
      // and add that since that's where optimization actually happens.
      PassPrediction::PassPeeper(__FILE__, 3866); // if
      unsigned Opcode = Op->getOpcode();
      while (Op->hasOneUse() && Op->user_back()->getOpcode() == Opcode &&
             Visited.insert(Op).second) {
        PassPrediction::PassPeeper(__FILE__, 3867); // while
        Op = Op->user_back();
      }
      RedoInsts.insert(Op);
    }
  }

  MadeChange = true;
}

// Canonicalize expressions of the following form:
//  x + (-Constant * y) -> x - (Constant * y)
//  x - (-Constant * y) -> x + (Constant * y)
Instruction *ReassociatePass::canonicalizeNegConstExpr(Instruction *I) {
  if (!I->hasOneUse() || I->getType()->isVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 3868); // if
    return nullptr;
  }

  // Must be a fmul or fdiv instruction.
  unsigned Opcode = I->getOpcode();
  if (Opcode != Instruction::FMul && Opcode != Instruction::FDiv) {
    PassPrediction::PassPeeper(__FILE__, 3869); // if
    return nullptr;
  }

  auto *C0 = dyn_cast<ConstantFP>(I->getOperand(0));
  auto *C1 = dyn_cast<ConstantFP>(I->getOperand(1));

  // Both operands are constant, let it get constant folded away.
  if (C0 && C1) {
    PassPrediction::PassPeeper(__FILE__, 3870); // if
    return nullptr;
  }

  ConstantFP *CF = C0 ? C0 : C1;

  // Must have one constant operand.
  if (!CF) {
    PassPrediction::PassPeeper(__FILE__, 3871); // if
    return nullptr;
  }

  // Must be a negative ConstantFP.
  if (!CF->isNegative()) {
    PassPrediction::PassPeeper(__FILE__, 3872); // if
    return nullptr;
  }

  // User must be a binary operator with one or more uses.
  Instruction *User = I->user_back();
  if (!isa<BinaryOperator>(User) || User->use_empty()) {
    PassPrediction::PassPeeper(__FILE__, 3873); // if
    return nullptr;
  }

  unsigned UserOpcode = User->getOpcode();
  if (UserOpcode != Instruction::FAdd && UserOpcode != Instruction::FSub) {
    PassPrediction::PassPeeper(__FILE__, 3874); // if
    return nullptr;
  }

  // Subtraction is not commutative. Explicitly, the following transform is
  // not valid: (-Constant * y) - x  -> x + (Constant * y)
  if (!User->isCommutative() && User->getOperand(1) != I) {
    PassPrediction::PassPeeper(__FILE__, 3875); // if
    return nullptr;
  }

  // Don't canonicalize x + (-Constant * y) -> x - (Constant * y), if the
  // resulting subtract will be broken up later.  This can get us into an
  // infinite loop during reassociation.
  if (UserOpcode == Instruction::FAdd && ShouldBreakUpSubtract(User)) {
    PassPrediction::PassPeeper(__FILE__, 3876); // if
    return nullptr;
  }

  // Change the sign of the constant.
  APFloat Val = CF->getValueAPF();
  Val.changeSign();
  I->setOperand(C0 ? 0 : 1, ConstantFP::get(CF->getContext(), Val));

  // Canonicalize I to RHS to simplify the next bit of logic. E.g.,
  // ((-Const*y) + x) -> (x + (-Const*y)).
  if (User->getOperand(0) == I && User->isCommutative()) {
    PassPrediction::PassPeeper(__FILE__, 3877); // if
    cast<BinaryOperator>(User)->swapOperands();
  }

  Value *Op0 = User->getOperand(0);
  Value *Op1 = User->getOperand(1);
  BinaryOperator *NI;
  switch (UserOpcode) {
  default:
    llvm_unreachable("Unexpected Opcode!");
  case Instruction::FAdd:
    PassPrediction::PassPeeper(__FILE__, 3878); // case

    NI = BinaryOperator::CreateFSub(Op0, Op1);
    NI->setFastMathFlags(cast<FPMathOperator>(User)->getFastMathFlags());
    PassPrediction::PassPeeper(__FILE__, 3879); // break
    break;
  case Instruction::FSub:
    PassPrediction::PassPeeper(__FILE__, 3880); // case

    NI = BinaryOperator::CreateFAdd(Op0, Op1);
    NI->setFastMathFlags(cast<FPMathOperator>(User)->getFastMathFlags());
    PassPrediction::PassPeeper(__FILE__, 3881); // break
    break;
  }

  NI->insertBefore(User);
  NI->setName(User->getName());
  User->replaceAllUsesWith(NI);
  NI->setDebugLoc(I->getDebugLoc());
  RedoInsts.insert(I);
  MadeChange = true;
  return NI;
}

/// Inspect and optimize the given instruction. Note that erasing
/// instructions is not allowed.
void ReassociatePass::OptimizeInst(Instruction *I) {
  // Only consider operations that we understand.
  if (!isa<BinaryOperator>(I)) {
    PassPrediction::PassPeeper(__FILE__, 3882); // if
    return;
  }

  if (I->getOpcode() == Instruction::Shl &&
      isa<ConstantInt>(I->getOperand(1))) {
    // If an operand of this shift is a reassociable multiply, or if the shift
    // is used by a reassociable multiply or add, turn into a multiply.
    PassPrediction::PassPeeper(__FILE__, 3883); // if
    if (isReassociableOp(I->getOperand(0), Instruction::Mul) ||
        (I->hasOneUse() &&
         (isReassociableOp(I->user_back(), Instruction::Mul) ||
          isReassociableOp(I->user_back(), Instruction::Add)))) {
      PassPrediction::PassPeeper(__FILE__, 3884); // if
      Instruction *NI = ConvertShiftToMul(I);
      RedoInsts.insert(I);
      MadeChange = true;
      I = NI;
    }
  }

  // Canonicalize negative constants out of expressions.
  if (Instruction *Res = canonicalizeNegConstExpr(I)) {
    PassPrediction::PassPeeper(__FILE__, 3885); // if
    I = Res;
  }

  // Commute binary operators, to canonicalize the order of their operands.
  // This can potentially expose more CSE opportunities, and makes writing other
  // transformations simpler.
  if (I->isCommutative()) {
    PassPrediction::PassPeeper(__FILE__, 3886); // if
    canonicalizeOperands(I);
  }

  // Don't optimize floating point instructions that don't have unsafe algebra.
  if (I->getType()->isFPOrFPVectorTy() && !I->hasUnsafeAlgebra()) {
    PassPrediction::PassPeeper(__FILE__, 3887); // if
    return;
  }

  // Do not reassociate boolean (i1) expressions.  We want to preserve the
  // original order of evaluation for short-circuited comparisons that
  // SimplifyCFG has folded to AND/OR expressions.  If the expression
  // is not further optimized, it is likely to be transformed back to a
  // short-circuited form for code gen, and the source order may have been
  // optimized for the most likely conditions.
  if (I->getType()->isIntegerTy(1)) {
    PassPrediction::PassPeeper(__FILE__, 3888); // if
    return;
  }

  // If this is a subtract instruction which is not already in negate form,
  // see if we can convert it to X+-Y.
  if (I->getOpcode() == Instruction::Sub) {
    PassPrediction::PassPeeper(__FILE__, 3889); // if
    if (ShouldBreakUpSubtract(I)) {
      PassPrediction::PassPeeper(__FILE__, 3890); // if
      Instruction *NI = BreakUpSubtract(I, RedoInsts);
      RedoInsts.insert(I);
      MadeChange = true;
      I = NI;
    } else if (BinaryOperator::isNeg(I)) {
      // Otherwise, this is a negation.  See if the operand is a multiply tree
      // and if this is not an inner node of a multiply tree.
      PassPrediction::PassPeeper(__FILE__, 3891); // if
      if (isReassociableOp(I->getOperand(1), Instruction::Mul) &&
          (!I->hasOneUse() ||
           !isReassociableOp(I->user_back(), Instruction::Mul))) {
        PassPrediction::PassPeeper(__FILE__, 3892); // if
        Instruction *NI = LowerNegateToMultiply(I);
        // If the negate was simplified, revisit the users to see if we can
        // reassociate further.
        for (User *U : NI->users()) {
          PassPrediction::PassPeeper(__FILE__, 3893); // for-range
          if (BinaryOperator *Tmp = dyn_cast<BinaryOperator>(U)) {
            PassPrediction::PassPeeper(__FILE__, 3894); // if
            RedoInsts.insert(Tmp);
          }
        }
        RedoInsts.insert(I);
        MadeChange = true;
        I = NI;
      }
    }
  } else if (I->getOpcode() == Instruction::FSub) {
    PassPrediction::PassPeeper(__FILE__, 3895); // if
    if (ShouldBreakUpSubtract(I)) {
      PassPrediction::PassPeeper(__FILE__, 3896); // if
      Instruction *NI = BreakUpSubtract(I, RedoInsts);
      RedoInsts.insert(I);
      MadeChange = true;
      I = NI;
    } else if (BinaryOperator::isFNeg(I)) {
      // Otherwise, this is a negation.  See if the operand is a multiply tree
      // and if this is not an inner node of a multiply tree.
      PassPrediction::PassPeeper(__FILE__, 3897); // if
      if (isReassociableOp(I->getOperand(1), Instruction::FMul) &&
          (!I->hasOneUse() ||
           !isReassociableOp(I->user_back(), Instruction::FMul))) {
        // If the negate was simplified, revisit the users to see if we can
        // reassociate further.
        PassPrediction::PassPeeper(__FILE__, 3898); // if
        Instruction *NI = LowerNegateToMultiply(I);
        for (User *U : NI->users()) {
          PassPrediction::PassPeeper(__FILE__, 3899); // for-range
          if (BinaryOperator *Tmp = dyn_cast<BinaryOperator>(U)) {
            PassPrediction::PassPeeper(__FILE__, 3900); // if
            RedoInsts.insert(Tmp);
          }
        }
        RedoInsts.insert(I);
        MadeChange = true;
        I = NI;
      }
    }
  }

  // If this instruction is an associative binary operator, process it.
  if (!I->isAssociative()) {
    PassPrediction::PassPeeper(__FILE__, 3901); // if
    return;
  }
  BinaryOperator *BO = cast<BinaryOperator>(I);

  // If this is an interior node of a reassociable tree, ignore it until we
  // get to the root of the tree, to avoid N^2 analysis.
  unsigned Opcode = BO->getOpcode();
  if (BO->hasOneUse() && BO->user_back()->getOpcode() == Opcode) {
    // During the initial run we will get to the root of the tree.
    // But if we get here while we are redoing instructions, there is no
    // guarantee that the root will be visited. So Redo later
    PassPrediction::PassPeeper(__FILE__, 3902); // if
    if (BO->user_back() != BO &&
        BO->getParent() == BO->user_back()->getParent()) {
      PassPrediction::PassPeeper(__FILE__, 3903); // if
      RedoInsts.insert(BO->user_back());
    }
    return;
  }

  // If this is an add tree that is used by a sub instruction, ignore it
  // until we process the subtract.
  if (BO->hasOneUse() && BO->getOpcode() == Instruction::Add &&
      cast<Instruction>(BO->user_back())->getOpcode() == Instruction::Sub) {
    PassPrediction::PassPeeper(__FILE__, 3904); // if
    return;
  }
  if (BO->hasOneUse() && BO->getOpcode() == Instruction::FAdd &&
      cast<Instruction>(BO->user_back())->getOpcode() == Instruction::FSub) {
    PassPrediction::PassPeeper(__FILE__, 3905); // if
    return;
  }

  ReassociateExpression(BO);
}

void ReassociatePass::ReassociateExpression(BinaryOperator *I) {
  // First, walk the expression tree, linearizing the tree, collecting the
  // operand information.
  SmallVector<RepeatedValue, 8> Tree;
  MadeChange |= LinearizeExprTree(I, Tree);
  SmallVector<ValueEntry, 8> Ops;
  Ops.reserve(Tree.size());
  for (unsigned i = 0, e = Tree.size(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 3906); // for
    RepeatedValue E = Tree[i];
    Ops.append(E.second.getZExtValue(), ValueEntry(getRank(E.first), E.first));
  }

  DEBUG(dbgs() << "RAIn:\t"; PrintOps(I, Ops); dbgs() << '\n');

  // Now that we have linearized the tree to a list and have gathered all of
  // the operands and their ranks, sort the operands by their rank.  Use a
  // stable_sort so that values with equal ranks will have their relative
  // positions maintained (and so the compiler is deterministic).  Note that
  // this sorts so that the highest ranking values end up at the beginning of
  // the vector.
  std::stable_sort(Ops.begin(), Ops.end());

  // Now that we have the expression tree in a convenient
  // sorted form, optimize it globally if possible.
  if (Value *V = OptimizeExpression(I, Ops)) {
    PassPrediction::PassPeeper(__FILE__, 3907); // if
    if (V == I) {
      // Self-referential expression in unreachable code.
      PassPrediction::PassPeeper(__FILE__, 3908); // if
      return;
    }
    // This expression tree simplified to something that isn't a tree,
    // eliminate it.
    DEBUG(dbgs() << "Reassoc to scalar: " << *V << '\n');
    I->replaceAllUsesWith(V);
    if (Instruction *VI = dyn_cast<Instruction>(V)) {
      PassPrediction::PassPeeper(__FILE__, 3909); // if
      VI->setDebugLoc(I->getDebugLoc());
    }
    RedoInsts.insert(I);
    ++NumAnnihil;
    return;
  }

  // We want to sink immediates as deeply as possible except in the case where
  // this is a multiply tree used only by an add, and the immediate is a -1.
  // In this case we reassociate to put the negation on the outside so that we
  // can fold the negation into the add: (-X)*Y + Z -> Z-X*Y
  if (I->hasOneUse()) {
    PassPrediction::PassPeeper(__FILE__, 3910); // if
    if (I->getOpcode() == Instruction::Mul &&
        cast<Instruction>(I->user_back())->getOpcode() == Instruction::Add &&
        isa<ConstantInt>(Ops.back().Op) &&
        cast<ConstantInt>(Ops.back().Op)->isMinusOne()) {
      PassPrediction::PassPeeper(__FILE__, 3911); // if
      ValueEntry Tmp = Ops.pop_back_val();
      Ops.insert(Ops.begin(), Tmp);
    } else if (I->getOpcode() == Instruction::FMul &&
               cast<Instruction>(I->user_back())->getOpcode() ==
                   Instruction::FAdd &&
               isa<ConstantFP>(Ops.back().Op) &&
               cast<ConstantFP>(Ops.back().Op)->isExactlyValue(-1.0)) {
      PassPrediction::PassPeeper(__FILE__, 3912); // if
      ValueEntry Tmp = Ops.pop_back_val();
      Ops.insert(Ops.begin(), Tmp);
    }
  }

  DEBUG(dbgs() << "RAOut:\t"; PrintOps(I, Ops); dbgs() << '\n');

  if (Ops.size() == 1) {
    PassPrediction::PassPeeper(__FILE__, 3913); // if
    if (Ops[0].Op == I) {
      // Self-referential expression in unreachable code.
      PassPrediction::PassPeeper(__FILE__, 3914); // if
      return;
    }

    // This expression tree simplified to something that isn't a tree,
    // eliminate it.
    I->replaceAllUsesWith(Ops[0].Op);
    if (Instruction *OI = dyn_cast<Instruction>(Ops[0].Op)) {
      PassPrediction::PassPeeper(__FILE__, 3915); // if
      OI->setDebugLoc(I->getDebugLoc());
    }
    RedoInsts.insert(I);
    return;
  }

  // Now that we ordered and optimized the expressions, splat them back into
  // the expression tree, removing any unneeded nodes.
  RewriteExprTree(I, Ops);
}

PreservedAnalyses ReassociatePass::run(Function &F, FunctionAnalysisManager &) {
  // Get the functions basic blocks in Reverse Post Order. This order is used by
  // BuildRankMap to pre calculate ranks correctly. It also excludes dead basic
  // blocks (it has been seen that the analysis in this pass could hang when
  // analysing dead basic blocks).
  ReversePostOrderTraversal<Function *> RPOT(&F);

  // Calculate the rank map for F.
  BuildRankMap(F, RPOT);

  MadeChange = false;
  // Traverse the same blocks that was analysed by BuildRankMap.
  for (BasicBlock *BI : RPOT) {
    assert(RankMap.count(&*BI) && "BB should be ranked.");
    // Optimize every instruction in the basic block.
    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;) {
      PassPrediction::PassPeeper(__FILE__, 3916); // for
      if (isInstructionTriviallyDead(&*II)) {
        PassPrediction::PassPeeper(__FILE__, 3917); // if
        EraseInst(&*II++);
      } else {
        PassPrediction::PassPeeper(__FILE__, 3918); // else
        OptimizeInst(&*II);
        assert(II->getParent() == &*BI && "Moved to a different block!");
        ++II;
      }
    }

    // Make a copy of all the instructions to be redone so we can remove dead
    // instructions.
    SetVector<AssertingVH<Instruction>> ToRedo(RedoInsts);
    // Iterate over all instructions to be reevaluated and remove trivially dead
    // instructions. If any operand of the trivially dead instruction becomes
    // dead mark it for deletion as well. Continue this process until all
    // trivially dead instructions have been removed.
    while (!ToRedo.empty()) {
      PassPrediction::PassPeeper(__FILE__, 3919); // while
      Instruction *I = ToRedo.pop_back_val();
      if (isInstructionTriviallyDead(I)) {
        PassPrediction::PassPeeper(__FILE__, 3920); // if
        RecursivelyEraseDeadInsts(I, ToRedo);
        MadeChange = true;
      }
    }

    // Now that we have removed dead instructions, we can reoptimize the
    // remaining instructions.
    while (!RedoInsts.empty()) {
      PassPrediction::PassPeeper(__FILE__, 3921); // while
      Instruction *I = RedoInsts.pop_back_val();
      if (isInstructionTriviallyDead(I)) {
        PassPrediction::PassPeeper(__FILE__, 3922); // if
        EraseInst(I);
      } else {
        PassPrediction::PassPeeper(__FILE__, 3923); // else
        OptimizeInst(I);
      }
    }
  }

  // We are done with the rank map.
  RankMap.clear();
  ValueRankMap.clear();

  if (MadeChange) {
    PassPrediction::PassPeeper(__FILE__, 3924); // if
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<GlobalsAA>();
    return PA;
  }

  return PreservedAnalyses::all();
}

namespace {
class ReassociateLegacyPass : public FunctionPass {
  ReassociatePass Impl;

public:
  static char ID; // Pass identification, replacement for typeid
  ReassociateLegacyPass() : FunctionPass(ID) {
    initializeReassociateLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F)) {
      PassPrediction::PassPeeper(__FILE__, 3925); // if
      return false;
    }

    FunctionAnalysisManager DummyFAM;
    auto PA = Impl.run(F, DummyFAM);
    return !PA.areAllPreserved();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
} // namespace

char ReassociateLegacyPass::ID = 0;
INITIALIZE_PASS(ReassociateLegacyPass, "reassociate", "Reassociate expressions",
                false, false)

// Public interface to the Reassociate pass
FunctionPass *llvm::createReassociatePass() {
  return new ReassociateLegacyPass();
}
