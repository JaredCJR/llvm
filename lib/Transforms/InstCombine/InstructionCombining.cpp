#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG.  This pass is where
// algebraic simplification happens.
//
// This pass combines things like:
//    %Y = add i32 %X, 1
//    %Z = add i32 %Y, 1
// into:
//    %Z = add i32 %X, 2
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. Compare instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All cmp instructions on boolean values are replaced with logical ops
//    5. add X, X is represented as (X*2) => (X << 1)
//    6. Multiplies with a power-of-two constant argument are transformed into
//       shifts.
//   ... etc.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm-c/Initialization.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <climits>
using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "instcombine"

STATISTIC(NumCombined, "Number of insts combined");
STATISTIC(NumConstProp, "Number of constant folds");
STATISTIC(NumDeadInst, "Number of dead inst eliminated");
STATISTIC(NumSunkInst, "Number of instructions sunk");
STATISTIC(NumExpand, "Number of expansions");
STATISTIC(NumFactor, "Number of factorizations");
STATISTIC(NumReassoc, "Number of reassociations");

static cl::opt<bool>
    EnableExpensiveCombines("expensive-combines",
                            cl::desc("Enable expensive instruction combines"));

static cl::opt<unsigned> MaxArraySize(
    "instcombine-maxarray-size", cl::init(1024),
    cl::desc("Maximum array size considered when doing a combine"));

Value *InstCombiner::EmitGEPOffset(User *GEP) {
  return llvm::EmitGEPOffset(&Builder, DL, GEP);
}

/// Return true if it is desirable to convert an integer computation from a
/// given bit width to a new bit width.
/// We don't want to convert from a legal to an illegal type or from a smaller
/// to a larger illegal type. A width of '1' is always treated as a legal type
/// because i1 is a fundamental type in IR, and there are many specialized
/// optimizations for i1 types.
bool InstCombiner::shouldChangeType(unsigned FromWidth,
                                    unsigned ToWidth) const {
  bool FromLegal = FromWidth == 1 || DL.isLegalInteger(FromWidth);
  bool ToLegal = ToWidth == 1 || DL.isLegalInteger(ToWidth);

  // If this is a legal integer from type, and the result would be an illegal
  // type, don't do the transformation.
  if (FromLegal && !ToLegal) {
    PassPrediction::PassPeeper(__FILE__, 30); // if
    return false;
  }

  // Otherwise, if both are illegal, do not increase the size of the result. We
  // do allow things like i160 -> i64, but not i64 -> i160.
  if (!FromLegal && !ToLegal && ToWidth > FromWidth) {
    PassPrediction::PassPeeper(__FILE__, 31); // if
    return false;
  }

  return true;
}

/// Return true if it is desirable to convert a computation from 'From' to 'To'.
/// We don't want to convert from a legal to an illegal type or from a smaller
/// to a larger illegal type. i1 is always treated as a legal type because it is
/// a fundamental type in IR, and there are many specialized optimizations for
/// i1 types.
bool InstCombiner::shouldChangeType(Type *From, Type *To) const {
  assert(From->isIntegerTy() && To->isIntegerTy());

  unsigned FromWidth = From->getPrimitiveSizeInBits();
  unsigned ToWidth = To->getPrimitiveSizeInBits();
  return shouldChangeType(FromWidth, ToWidth);
}

// Return true, if No Signed Wrap should be maintained for I.
// The No Signed Wrap flag can be kept if the operation "B (I.getOpcode) C",
// where both B and C should be ConstantInts, results in a constant that does
// not overflow. This function only handles the Add and Sub opcodes. For
// all other opcodes, the function conservatively returns false.
static bool MaintainNoSignedWrap(BinaryOperator &I, Value *B, Value *C) {
  OverflowingBinaryOperator *OBO = dyn_cast<OverflowingBinaryOperator>(&I);
  if (!OBO || !OBO->hasNoSignedWrap()) {
    PassPrediction::PassPeeper(__FILE__, 32); // if
    return false;
  }

  // We reason about Add and Sub Only.
  Instruction::BinaryOps Opcode = I.getOpcode();
  if (Opcode != Instruction::Add && Opcode != Instruction::Sub) {
    PassPrediction::PassPeeper(__FILE__, 33); // if
    return false;
  }

  const APInt *BVal, *CVal;
  if (!match(B, m_APInt(BVal)) || !match(C, m_APInt(CVal))) {
    PassPrediction::PassPeeper(__FILE__, 34); // if
    return false;
  }

  bool Overflow = false;
  if (Opcode == Instruction::Add) {
    PassPrediction::PassPeeper(__FILE__, 35); // if
    (void)BVal->sadd_ov(*CVal, Overflow);
  } else {
    PassPrediction::PassPeeper(__FILE__, 36); // else
    (void)BVal->ssub_ov(*CVal, Overflow);
  }

  return !Overflow;
}

/// Conservatively clears subclassOptionalData after a reassociation or
/// commutation. We preserve fast-math flags when applicable as they can be
/// preserved.
static void ClearSubclassDataAfterReassociation(BinaryOperator &I) {
  FPMathOperator *FPMO = dyn_cast<FPMathOperator>(&I);
  if (!FPMO) {
    PassPrediction::PassPeeper(__FILE__, 37); // if
    I.clearSubclassOptionalData();
    return;
  }

  FastMathFlags FMF = I.getFastMathFlags();
  I.clearSubclassOptionalData();
  I.setFastMathFlags(FMF);
}

/// Combine constant operands of associative operations either before or after a
/// cast to eliminate one of the associative operations:
/// (op (cast (op X, C2)), C1) --> (cast (op X, op (C1, C2)))
/// (op (cast (op X, C2)), C1) --> (op (cast X), op (C1, C2))
static bool simplifyAssocCastAssoc(BinaryOperator *BinOp1) {
  auto *Cast = dyn_cast<CastInst>(BinOp1->getOperand(0));
  if (!Cast || !Cast->hasOneUse()) {
    PassPrediction::PassPeeper(__FILE__, 38); // if
    return false;
  }

  // TODO: Enhance logic for other casts and remove this check.
  auto CastOpcode = Cast->getOpcode();
  if (CastOpcode != Instruction::ZExt) {
    PassPrediction::PassPeeper(__FILE__, 39); // if
    return false;
  }

  // TODO: Enhance logic for other BinOps and remove this check.
  if (!BinOp1->isBitwiseLogicOp()) {
    PassPrediction::PassPeeper(__FILE__, 40); // if
    return false;
  }

  auto AssocOpcode = BinOp1->getOpcode();
  auto *BinOp2 = dyn_cast<BinaryOperator>(Cast->getOperand(0));
  if (!BinOp2 || !BinOp2->hasOneUse() || BinOp2->getOpcode() != AssocOpcode) {
    PassPrediction::PassPeeper(__FILE__, 41); // if
    return false;
  }

  Constant *C1, *C2;
  if (!match(BinOp1->getOperand(1), m_Constant(C1)) ||
      !match(BinOp2->getOperand(1), m_Constant(C2))) {
    PassPrediction::PassPeeper(__FILE__, 42); // if
    return false;
  }

  // TODO: This assumes a zext cast.
  // Eg, if it was a trunc, we'd cast C1 to the source type because casting C2
  // to the destination type might lose bits.

  // Fold the constants together in the destination type:
  // (op (cast (op X, C2)), C1) --> (op (cast X), FoldedC)
  Type *DestTy = C1->getType();
  Constant *CastC2 = ConstantExpr::getCast(CastOpcode, C2, DestTy);
  Constant *FoldedC = ConstantExpr::get(AssocOpcode, C1, CastC2);
  Cast->setOperand(0, BinOp2->getOperand(0));
  BinOp1->setOperand(1, FoldedC);
  return true;
}

/// This performs a few simplifications for operators that are associative or
/// commutative:
///
///  Commutative operators:
///
///  1. Order operands such that they are listed from right (least complex) to
///     left (most complex).  This puts constants before unary operators before
///     binary operators.
///
///  Associative operators:
///
///  2. Transform: "(A op B) op C" ==> "A op (B op C)" if "B op C" simplifies.
///  3. Transform: "A op (B op C)" ==> "(A op B) op C" if "A op B" simplifies.
///
///  Associative and commutative operators:
///
///  4. Transform: "(A op B) op C" ==> "(C op A) op B" if "C op A" simplifies.
///  5. Transform: "A op (B op C)" ==> "B op (C op A)" if "C op A" simplifies.
///  6. Transform: "(A op C1) op (B op C2)" ==> "(A op B) op (C1 op C2)"
///     if C1 and C2 are constants.
bool InstCombiner::SimplifyAssociativeOrCommutative(BinaryOperator &I) {
  Instruction::BinaryOps Opcode = I.getOpcode();
  bool Changed = false;

  do {
    // Order operands such that they are listed from right (least complex) to
    // left (most complex).  This puts constants before unary operators before
    // binary operators.
    PassPrediction::PassPeeper(__FILE__, 43); // do-while
    if (I.isCommutative() &&
        getComplexity(I.getOperand(0)) < getComplexity(I.getOperand(1))) {
      PassPrediction::PassPeeper(__FILE__, 44); // if
      Changed = !I.swapOperands();
    }

    BinaryOperator *Op0 = dyn_cast<BinaryOperator>(I.getOperand(0));
    BinaryOperator *Op1 = dyn_cast<BinaryOperator>(I.getOperand(1));

    if (I.isAssociative()) {
      // Transform: "(A op B) op C" ==> "A op (B op C)" if "B op C" simplifies.
      PassPrediction::PassPeeper(__FILE__, 45); // if
      if (Op0 && Op0->getOpcode() == Opcode) {
        PassPrediction::PassPeeper(__FILE__, 46); // if
        Value *A = Op0->getOperand(0);
        Value *B = Op0->getOperand(1);
        Value *C = I.getOperand(1);

        // Does "B op C" simplify?
        if (Value *V = SimplifyBinOp(Opcode, B, C, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "A op V".
          PassPrediction::PassPeeper(__FILE__, 47); // if
          I.setOperand(0, A);
          I.setOperand(1, V);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          if (MaintainNoSignedWrap(I, B, C) &&
              (!Op0 || (isa<BinaryOperator>(Op0) && Op0->hasNoSignedWrap()))) {
            // Note: this is only valid because SimplifyBinOp doesn't look at
            // the operands to Op0.
            PassPrediction::PassPeeper(__FILE__, 48); // if
            I.clearSubclassOptionalData();
            I.setHasNoSignedWrap(true);
          } else {
            PassPrediction::PassPeeper(__FILE__, 49); // else
            ClearSubclassDataAfterReassociation(I);
          }

          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "A op (B op C)" ==> "(A op B) op C" if "A op B" simplifies.
      if (Op1 && Op1->getOpcode() == Opcode) {
        PassPrediction::PassPeeper(__FILE__, 50); // if
        Value *A = I.getOperand(0);
        Value *B = Op1->getOperand(0);
        Value *C = Op1->getOperand(1);

        // Does "A op B" simplify?
        if (Value *V = SimplifyBinOp(Opcode, A, B, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "V op C".
          PassPrediction::PassPeeper(__FILE__, 51); // if
          I.setOperand(0, V);
          I.setOperand(1, C);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          ClearSubclassDataAfterReassociation(I);
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }
    }

    if (I.isAssociative() && I.isCommutative()) {
      PassPrediction::PassPeeper(__FILE__, 52); // if
      if (simplifyAssocCastAssoc(&I)) {
        PassPrediction::PassPeeper(__FILE__, 53); // if
        Changed = true;
        ++NumReassoc;
        continue;
      }

      // Transform: "(A op B) op C" ==> "(C op A) op B" if "C op A" simplifies.
      if (Op0 && Op0->getOpcode() == Opcode) {
        PassPrediction::PassPeeper(__FILE__, 54); // if
        Value *A = Op0->getOperand(0);
        Value *B = Op0->getOperand(1);
        Value *C = I.getOperand(1);

        // Does "C op A" simplify?
        if (Value *V = SimplifyBinOp(Opcode, C, A, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "V op B".
          PassPrediction::PassPeeper(__FILE__, 55); // if
          I.setOperand(0, V);
          I.setOperand(1, B);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          ClearSubclassDataAfterReassociation(I);
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "A op (B op C)" ==> "B op (C op A)" if "C op A" simplifies.
      if (Op1 && Op1->getOpcode() == Opcode) {
        PassPrediction::PassPeeper(__FILE__, 56); // if
        Value *A = I.getOperand(0);
        Value *B = Op1->getOperand(0);
        Value *C = Op1->getOperand(1);

        // Does "C op A" simplify?
        if (Value *V = SimplifyBinOp(Opcode, C, A, SQ.getWithInstruction(&I))) {
          // It simplifies to V.  Form "B op V".
          PassPrediction::PassPeeper(__FILE__, 57); // if
          I.setOperand(0, B);
          I.setOperand(1, V);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          ClearSubclassDataAfterReassociation(I);
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "(A op C1) op (B op C2)" ==> "(A op B) op (C1 op C2)"
      // if C1 and C2 are constants.
      if (Op0 && Op1 && Op0->getOpcode() == Opcode &&
          Op1->getOpcode() == Opcode && isa<Constant>(Op0->getOperand(1)) &&
          isa<Constant>(Op1->getOperand(1)) && Op0->hasOneUse() &&
          Op1->hasOneUse()) {
        PassPrediction::PassPeeper(__FILE__, 58); // if
        Value *A = Op0->getOperand(0);
        Constant *C1 = cast<Constant>(Op0->getOperand(1));
        Value *B = Op1->getOperand(0);
        Constant *C2 = cast<Constant>(Op1->getOperand(1));

        Constant *Folded = ConstantExpr::get(Opcode, C1, C2);
        BinaryOperator *New = BinaryOperator::Create(Opcode, A, B);
        if (isa<FPMathOperator>(New)) {
          PassPrediction::PassPeeper(__FILE__, 59); // if
          FastMathFlags Flags = I.getFastMathFlags();
          Flags &= Op0->getFastMathFlags();
          Flags &= Op1->getFastMathFlags();
          New->setFastMathFlags(Flags);
        }
        InsertNewInstWith(New, I);
        New->takeName(Op1);
        I.setOperand(0, New);
        I.setOperand(1, Folded);
        // Conservatively clear the optional flags, since they may not be
        // preserved by the reassociation.
        ClearSubclassDataAfterReassociation(I);

        Changed = true;
        continue;
      }
    }

    // No further simplifications.
    return Changed;
  } while (1);
}

/// Return whether "X LOp (Y ROp Z)" is always equal to
/// "(X LOp Y) ROp (X LOp Z)".
static bool LeftDistributesOverRight(Instruction::BinaryOps LOp,
                                     Instruction::BinaryOps ROp) {
  switch (LOp) {
  default:
    return false;

  case Instruction::And:
    PassPrediction::PassPeeper(__FILE__, 60); // case

    // And distributes over Or and Xor.
    switch (ROp) {
    default:
      return false;
    case Instruction::Or:
      PassPrediction::PassPeeper(__FILE__, 61); // case

    case Instruction::Xor:
      PassPrediction::PassPeeper(__FILE__, 62); // case

      return true;
    }

  case Instruction::Mul:
    PassPrediction::PassPeeper(__FILE__, 63); // case

    // Multiplication distributes over addition and subtraction.
    switch (ROp) {
    default:
      return false;
    case Instruction::Add:
      PassPrediction::PassPeeper(__FILE__, 64); // case

    case Instruction::Sub:
      PassPrediction::PassPeeper(__FILE__, 65); // case

      return true;
    }

  case Instruction::Or:
    PassPrediction::PassPeeper(__FILE__, 66); // case

    // Or distributes over And.
    switch (ROp) {
    default:
      return false;
    case Instruction::And:
      PassPrediction::PassPeeper(__FILE__, 67); // case

      return true;
    }
  }
}

/// Return whether "(X LOp Y) ROp Z" is always equal to
/// "(X ROp Z) LOp (Y ROp Z)".
static bool RightDistributesOverLeft(Instruction::BinaryOps LOp,
                                     Instruction::BinaryOps ROp) {
  if (Instruction::isCommutative(ROp)) {
    PassPrediction::PassPeeper(__FILE__, 68); // if
    return LeftDistributesOverRight(ROp, LOp);
  }

  switch (LOp) {
  default:
    return false;
  // (X >> Z) & (Y >> Z)  -> (X&Y) >> Z  for all shifts.
  // (X >> Z) | (Y >> Z)  -> (X|Y) >> Z  for all shifts.
  // (X >> Z) ^ (Y >> Z)  -> (X^Y) >> Z  for all shifts.
  case Instruction::And:
    PassPrediction::PassPeeper(__FILE__, 69); // case

  case Instruction::Or:
    PassPrediction::PassPeeper(__FILE__, 70); // case

  case Instruction::Xor:
    PassPrediction::PassPeeper(__FILE__, 71); // case

    switch (ROp) {
    default:
      return false;
    case Instruction::Shl:
      PassPrediction::PassPeeper(__FILE__, 72); // case

    case Instruction::LShr:
      PassPrediction::PassPeeper(__FILE__, 73); // case

    case Instruction::AShr:
      PassPrediction::PassPeeper(__FILE__, 74); // case

      return true;
    }
  }
  // TODO: It would be nice to handle division, aka "(X + Y)/Z = X/Z + Y/Z",
  // but this requires knowing that the addition does not overflow and other
  // such subtleties.
  return false;
}

/// This function returns identity value for given opcode, which can be used to
/// factor patterns like (X * 2) + X ==> (X * 2) + (X * 1) ==> X * (2 + 1).
static Value *getIdentityValue(Instruction::BinaryOps Opcode, Value *V) {
  if (isa<Constant>(V)) {
    PassPrediction::PassPeeper(__FILE__, 75); // if
    return nullptr;
  }

  return ConstantExpr::getBinOpIdentity(Opcode, V->getType());
}

/// This function factors binary ops which can be combined using distributive
/// laws. This function tries to transform 'Op' based TopLevelOpcode to enable
/// factorization e.g for ADD(SHL(X , 2), MUL(X, 5)), When this function called
/// with TopLevelOpcode == Instruction::Add and Op = SHL(X, 2), transforms
/// SHL(X, 2) to MUL(X, 4) i.e. returns Instruction::Mul with LHS set to 'X' and
/// RHS to 4.
static Instruction::BinaryOps
getBinOpsForFactorization(Instruction::BinaryOps TopLevelOpcode,
                          BinaryOperator *Op, Value *&LHS, Value *&RHS) {
  assert(Op && "Expected a binary operator");

  LHS = Op->getOperand(0);
  RHS = Op->getOperand(1);

  switch (TopLevelOpcode) {
  default:
    return Op->getOpcode();

  case Instruction::Add:
    PassPrediction::PassPeeper(__FILE__, 76); // case

  case Instruction::Sub:
    PassPrediction::PassPeeper(__FILE__, 77); // case

    if (Op->getOpcode() == Instruction::Shl) {
      PassPrediction::PassPeeper(__FILE__, 78); // if
      if (Constant *CST = dyn_cast<Constant>(Op->getOperand(1))) {
        // The multiplier is really 1 << CST.
        PassPrediction::PassPeeper(__FILE__, 79); // if
        RHS = ConstantExpr::getShl(ConstantInt::get(Op->getType(), 1), CST);
        return Instruction::Mul;
      }
    }
    return Op->getOpcode();
  }

  // TODO: We can add other conversions e.g. shr => div etc.
}

/// This tries to simplify binary operations by factorizing out common terms
/// (e. g. "(A*B)+(A*C)" -> "A*(B+C)").
Value *InstCombiner::tryFactorization(BinaryOperator &I,
                                      Instruction::BinaryOps InnerOpcode,
                                      Value *A, Value *B, Value *C, Value *D) {
  assert(A && B && C && D && "All values must be provided");

  Value *V = nullptr;
  Value *SimplifiedInst = nullptr;
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  Instruction::BinaryOps TopLevelOpcode = I.getOpcode();

  // Does "X op' Y" always equal "Y op' X"?
  bool InnerCommutative = Instruction::isCommutative(InnerOpcode);

  // Does "X op' (Y op Z)" always equal "(X op' Y) op (X op' Z)"?
  if (LeftDistributesOverRight(InnerOpcode, TopLevelOpcode)) {
    // Does the instruction have the form "(A op' B) op (A op' D)" or, in the
    // commutative case, "(A op' B) op (C op' A)"?
    PassPrediction::PassPeeper(__FILE__, 80); // if
    if (A == C || (InnerCommutative && A == D)) {
      PassPrediction::PassPeeper(__FILE__, 81); // if
      if (A != C) {
        PassPrediction::PassPeeper(__FILE__, 82); // if
        std::swap(C, D);
      }
      // Consider forming "A op' (B op D)".
      // If "B op D" simplifies then it can be formed with no cost.
      V = SimplifyBinOp(TopLevelOpcode, B, D, SQ.getWithInstruction(&I));
      // If "B op D" doesn't simplify then only go on if both of the existing
      // operations "A op' B" and "C op' D" will be zapped as no longer used.
      if (!V && LHS->hasOneUse() && RHS->hasOneUse()) {
        PassPrediction::PassPeeper(__FILE__, 83); // if
        V = Builder.CreateBinOp(TopLevelOpcode, B, D, RHS->getName());
      }
      if (V) {
        PassPrediction::PassPeeper(__FILE__, 84); // if
        SimplifiedInst = Builder.CreateBinOp(InnerOpcode, A, V);
      }
    }
  }

  // Does "(X op Y) op' Z" always equal "(X op' Z) op (Y op' Z)"?
  if (!SimplifiedInst &&
      RightDistributesOverLeft(TopLevelOpcode, InnerOpcode)) {
    // Does the instruction have the form "(A op' B) op (C op' B)" or, in the
    // commutative case, "(A op' B) op (B op' D)"?
    PassPrediction::PassPeeper(__FILE__, 85); // if
    if (B == D || (InnerCommutative && B == C)) {
      PassPrediction::PassPeeper(__FILE__, 86); // if
      if (B != D) {
        PassPrediction::PassPeeper(__FILE__, 87); // if
        std::swap(C, D);
      }
      // Consider forming "(A op C) op' B".
      // If "A op C" simplifies then it can be formed with no cost.
      V = SimplifyBinOp(TopLevelOpcode, A, C, SQ.getWithInstruction(&I));

      // If "A op C" doesn't simplify then only go on if both of the existing
      // operations "A op' B" and "C op' D" will be zapped as no longer used.
      if (!V && LHS->hasOneUse() && RHS->hasOneUse()) {
        PassPrediction::PassPeeper(__FILE__, 88); // if
        V = Builder.CreateBinOp(TopLevelOpcode, A, C, LHS->getName());
      }
      if (V) {
        PassPrediction::PassPeeper(__FILE__, 89); // if
        SimplifiedInst = Builder.CreateBinOp(InnerOpcode, V, B);
      }
    }
  }

  if (SimplifiedInst) {
    PassPrediction::PassPeeper(__FILE__, 90); // if
    ++NumFactor;
    SimplifiedInst->takeName(&I);

    // Check if we can add NSW flag to SimplifiedInst. If so, set NSW flag.
    // TODO: Check for NUW.
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(SimplifiedInst)) {
      PassPrediction::PassPeeper(__FILE__, 91); // if
      if (isa<OverflowingBinaryOperator>(SimplifiedInst)) {
        PassPrediction::PassPeeper(__FILE__, 92); // if
        bool HasNSW = false;
        if (isa<OverflowingBinaryOperator>(&I)) {
          PassPrediction::PassPeeper(__FILE__, 93); // if
          HasNSW = I.hasNoSignedWrap();
        }

        if (auto *LOBO = dyn_cast<OverflowingBinaryOperator>(LHS)) {
          PassPrediction::PassPeeper(__FILE__, 94); // if
          HasNSW &= LOBO->hasNoSignedWrap();
        }

        if (auto *ROBO = dyn_cast<OverflowingBinaryOperator>(RHS)) {
          PassPrediction::PassPeeper(__FILE__, 95); // if
          HasNSW &= ROBO->hasNoSignedWrap();
        }

        // We can propagate 'nsw' if we know that
        //  %Y = mul nsw i16 %X, C
        //  %Z = add nsw i16 %Y, %X
        // =>
        //  %Z = mul nsw i16 %X, C+1
        //
        // iff C+1 isn't INT_MIN
        const APInt *CInt;
        if (TopLevelOpcode == Instruction::Add &&
            InnerOpcode == Instruction::Mul) {
          PassPrediction::PassPeeper(__FILE__, 96); // if
          if (match(V, m_APInt(CInt)) && !CInt->isMinSignedValue()) {
            PassPrediction::PassPeeper(__FILE__, 97); // if
            BO->setHasNoSignedWrap(HasNSW);
          }
        }
      }
    }
  }
  return SimplifiedInst;
}

/// This tries to simplify binary operations which some other binary operation
/// distributes over either by factorizing out common terms
/// (eg "(A*B)+(A*C)" -> "A*(B+C)") or expanding out if this results in
/// simplifications (eg: "A & (B | C) -> (A&B) | (A&C)" if this is a win).
/// Returns the simplified value, or null if it didn't simplify.
Value *InstCombiner::SimplifyUsingDistributiveLaws(BinaryOperator &I) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS);
  Instruction::BinaryOps TopLevelOpcode = I.getOpcode();

  {
    // Factorization.
    Value *A, *B, *C, *D;
    Instruction::BinaryOps LHSOpcode, RHSOpcode;
    if (Op0) {
      PassPrediction::PassPeeper(__FILE__, 98); // if
      LHSOpcode = getBinOpsForFactorization(TopLevelOpcode, Op0, A, B);
    }
    if (Op1) {
      PassPrediction::PassPeeper(__FILE__, 99); // if
      RHSOpcode = getBinOpsForFactorization(TopLevelOpcode, Op1, C, D);
    }

    // The instruction has the form "(A op' B) op (C op' D)".  Try to factorize
    // a common term.
    if (Op0 && Op1 && LHSOpcode == RHSOpcode) {
      PassPrediction::PassPeeper(__FILE__, 100); // if
      if (Value *V = tryFactorization(I, LHSOpcode, A, B, C, D)) {
        PassPrediction::PassPeeper(__FILE__, 101); // if
        return V;
      }
    }

    // The instruction has the form "(A op' B) op (C)".  Try to factorize common
    // term.
    if (Op0) {
      PassPrediction::PassPeeper(__FILE__, 102); // if
      if (Value *Ident = getIdentityValue(LHSOpcode, RHS)) {
        PassPrediction::PassPeeper(__FILE__, 103); // if
        if (Value *V = tryFactorization(I, LHSOpcode, A, B, RHS, Ident)) {
          PassPrediction::PassPeeper(__FILE__, 104); // if
          return V;
        }
      }
    }

    // The instruction has the form "(B) op (C op' D)".  Try to factorize common
    // term.
    if (Op1) {
      PassPrediction::PassPeeper(__FILE__, 105); // if
      if (Value *Ident = getIdentityValue(RHSOpcode, LHS)) {
        PassPrediction::PassPeeper(__FILE__, 106); // if
        if (Value *V = tryFactorization(I, RHSOpcode, LHS, Ident, C, D)) {
          PassPrediction::PassPeeper(__FILE__, 107); // if
          return V;
        }
      }
    }
  }

  // Expansion.
  if (Op0 && RightDistributesOverLeft(Op0->getOpcode(), TopLevelOpcode)) {
    // The instruction has the form "(A op' B) op C".  See if expanding it out
    // to "(A op C) op' (B op C)" results in simplifications.
    PassPrediction::PassPeeper(__FILE__, 108); // if
    Value *A = Op0->getOperand(0), *B = Op0->getOperand(1), *C = RHS;
    Instruction::BinaryOps InnerOpcode = Op0->getOpcode(); // op'

    Value *L = SimplifyBinOp(TopLevelOpcode, A, C, SQ.getWithInstruction(&I));
    Value *R = SimplifyBinOp(TopLevelOpcode, B, C, SQ.getWithInstruction(&I));

    // Do "A op C" and "B op C" both simplify?
    if (L && R) {
      // They do! Return "L op' R".
      PassPrediction::PassPeeper(__FILE__, 109); // if
      ++NumExpand;
      C = Builder.CreateBinOp(InnerOpcode, L, R);
      C->takeName(&I);
      return C;
    }

    // Does "A op C" simplify to the identity value for the inner opcode?
    if (L && L == ConstantExpr::getBinOpIdentity(InnerOpcode, L->getType())) {
      // They do! Return "B op C".
      PassPrediction::PassPeeper(__FILE__, 110); // if
      ++NumExpand;
      C = Builder.CreateBinOp(TopLevelOpcode, B, C);
      C->takeName(&I);
      return C;
    }

    // Does "B op C" simplify to the identity value for the inner opcode?
    if (R && R == ConstantExpr::getBinOpIdentity(InnerOpcode, R->getType())) {
      // They do! Return "A op C".
      PassPrediction::PassPeeper(__FILE__, 111); // if
      ++NumExpand;
      C = Builder.CreateBinOp(TopLevelOpcode, A, C);
      C->takeName(&I);
      return C;
    }
  }

  if (Op1 && LeftDistributesOverRight(TopLevelOpcode, Op1->getOpcode())) {
    // The instruction has the form "A op (B op' C)".  See if expanding it out
    // to "(A op B) op' (A op C)" results in simplifications.
    PassPrediction::PassPeeper(__FILE__, 112); // if
    Value *A = LHS, *B = Op1->getOperand(0), *C = Op1->getOperand(1);
    Instruction::BinaryOps InnerOpcode = Op1->getOpcode(); // op'

    Value *L = SimplifyBinOp(TopLevelOpcode, A, B, SQ.getWithInstruction(&I));
    Value *R = SimplifyBinOp(TopLevelOpcode, A, C, SQ.getWithInstruction(&I));

    // Do "A op B" and "A op C" both simplify?
    if (L && R) {
      // They do! Return "L op' R".
      PassPrediction::PassPeeper(__FILE__, 113); // if
      ++NumExpand;
      A = Builder.CreateBinOp(InnerOpcode, L, R);
      A->takeName(&I);
      return A;
    }

    // Does "A op B" simplify to the identity value for the inner opcode?
    if (L && L == ConstantExpr::getBinOpIdentity(InnerOpcode, L->getType())) {
      // They do! Return "A op C".
      PassPrediction::PassPeeper(__FILE__, 114); // if
      ++NumExpand;
      A = Builder.CreateBinOp(TopLevelOpcode, A, C);
      A->takeName(&I);
      return A;
    }

    // Does "A op C" simplify to the identity value for the inner opcode?
    if (R && R == ConstantExpr::getBinOpIdentity(InnerOpcode, R->getType())) {
      // They do! Return "A op B".
      PassPrediction::PassPeeper(__FILE__, 115); // if
      ++NumExpand;
      A = Builder.CreateBinOp(TopLevelOpcode, A, B);
      A->takeName(&I);
      return A;
    }
  }

  // (op (select (a, c, b)), (select (a, d, b))) -> (select (a, (op c, d), 0))
  // (op (select (a, b, c)), (select (a, b, d))) -> (select (a, 0, (op c, d)))
  if (auto *SI0 = dyn_cast<SelectInst>(LHS)) {
    PassPrediction::PassPeeper(__FILE__, 116); // if
    if (auto *SI1 = dyn_cast<SelectInst>(RHS)) {
      PassPrediction::PassPeeper(__FILE__, 117); // if
      if (SI0->getCondition() == SI1->getCondition()) {
        PassPrediction::PassPeeper(__FILE__, 118); // if
        Value *SI = nullptr;
        if (Value *V = SimplifyBinOp(TopLevelOpcode, SI0->getFalseValue(),
                                     SI1->getFalseValue(),
                                     SQ.getWithInstruction(&I))) {
          PassPrediction::PassPeeper(__FILE__, 119); // if
          SI = Builder.CreateSelect(SI0->getCondition(),
                                    Builder.CreateBinOp(TopLevelOpcode,
                                                        SI0->getTrueValue(),
                                                        SI1->getTrueValue()),
                                    V);
        }
        if (Value *V =
                SimplifyBinOp(TopLevelOpcode, SI0->getTrueValue(),
                              SI1->getTrueValue(), SQ.getWithInstruction(&I))) {
          PassPrediction::PassPeeper(__FILE__, 120); // if
          SI = Builder.CreateSelect(SI0->getCondition(), V,
                                    Builder.CreateBinOp(TopLevelOpcode,
                                                        SI0->getFalseValue(),
                                                        SI1->getFalseValue()));
        }
        if (SI) {
          PassPrediction::PassPeeper(__FILE__, 121); // if
          SI->takeName(&I);
          return SI;
        }
      }
    }
  }

  return nullptr;
}

/// Given a 'sub' instruction, return the RHS of the instruction if the LHS is a
/// constant zero (which is the 'negate' form).
Value *InstCombiner::dyn_castNegVal(Value *V) const {
  if (BinaryOperator::isNeg(V)) {
    PassPrediction::PassPeeper(__FILE__, 122); // if
    return BinaryOperator::getNegArgument(V);
  }

  // Constants can be considered to be negated values if they can be folded.
  if (ConstantInt *C = dyn_cast<ConstantInt>(V)) {
    PassPrediction::PassPeeper(__FILE__, 123); // if
    return ConstantExpr::getNeg(C);
  }

  if (ConstantDataVector *C = dyn_cast<ConstantDataVector>(V)) {
    PassPrediction::PassPeeper(__FILE__, 124); // if
    if (C->getType()->getElementType()->isIntegerTy()) {
      PassPrediction::PassPeeper(__FILE__, 125); // if
      return ConstantExpr::getNeg(C);
    }
  }

  if (ConstantVector *CV = dyn_cast<ConstantVector>(V)) {
    PassPrediction::PassPeeper(__FILE__, 126); // if
    for (unsigned i = 0, e = CV->getNumOperands(); i != e; ++i) {
      PassPrediction::PassPeeper(__FILE__, 127); // for
      Constant *Elt = CV->getAggregateElement(i);
      if (!Elt) {
        PassPrediction::PassPeeper(__FILE__, 128); // if
        return nullptr;
      }

      if (isa<UndefValue>(Elt)) {
        PassPrediction::PassPeeper(__FILE__, 129); // if
        continue;
      }

      if (!isa<ConstantInt>(Elt)) {
        PassPrediction::PassPeeper(__FILE__, 130); // if
        return nullptr;
      }
    }
    return ConstantExpr::getNeg(CV);
  }

  return nullptr;
}

/// Given a 'fsub' instruction, return the RHS of the instruction if the LHS is
/// a constant negative zero (which is the 'negate' form).
Value *InstCombiner::dyn_castFNegVal(Value *V, bool IgnoreZeroSign) const {
  if (BinaryOperator::isFNeg(V, IgnoreZeroSign)) {
    PassPrediction::PassPeeper(__FILE__, 131); // if
    return BinaryOperator::getFNegArgument(V);
  }

  // Constants can be considered to be negated values if they can be folded.
  if (ConstantFP *C = dyn_cast<ConstantFP>(V)) {
    PassPrediction::PassPeeper(__FILE__, 132); // if
    return ConstantExpr::getFNeg(C);
  }

  if (ConstantDataVector *C = dyn_cast<ConstantDataVector>(V)) {
    PassPrediction::PassPeeper(__FILE__, 133); // if
    if (C->getType()->getElementType()->isFloatingPointTy()) {
      PassPrediction::PassPeeper(__FILE__, 134); // if
      return ConstantExpr::getFNeg(C);
    }
  }

  return nullptr;
}

static Value *foldOperationIntoSelectOperand(Instruction &I, Value *SO,
                                             InstCombiner::BuilderTy &Builder) {
  if (auto *Cast = dyn_cast<CastInst>(&I)) {
    PassPrediction::PassPeeper(__FILE__, 135); // if
    return Builder.CreateCast(Cast->getOpcode(), SO, I.getType());
  }

  assert(I.isBinaryOp() && "Unexpected opcode for select folding");

  // Figure out if the constant is the left or the right argument.
  bool ConstIsRHS = isa<Constant>(I.getOperand(1));
  Constant *ConstOperand = cast<Constant>(I.getOperand(ConstIsRHS));

  if (auto *SOC = dyn_cast<Constant>(SO)) {
    PassPrediction::PassPeeper(__FILE__, 136); // if
    if (ConstIsRHS) {
      PassPrediction::PassPeeper(__FILE__, 137); // if
      return ConstantExpr::get(I.getOpcode(), SOC, ConstOperand);
    }
    return ConstantExpr::get(I.getOpcode(), ConstOperand, SOC);
  }

  Value *Op0 = SO, *Op1 = ConstOperand;
  if (!ConstIsRHS) {
    PassPrediction::PassPeeper(__FILE__, 138); // if
    std::swap(Op0, Op1);
  }

  auto *BO = cast<BinaryOperator>(&I);
  Value *RI =
      Builder.CreateBinOp(BO->getOpcode(), Op0, Op1, SO->getName() + ".op");
  auto *FPInst = dyn_cast<Instruction>(RI);
  if (FPInst && isa<FPMathOperator>(FPInst)) {
    PassPrediction::PassPeeper(__FILE__, 139); // if
    FPInst->copyFastMathFlags(BO);
  }
  return RI;
}

Instruction *InstCombiner::FoldOpIntoSelect(Instruction &Op, SelectInst *SI) {
  // Don't modify shared select instructions.
  if (!SI->hasOneUse()) {
    PassPrediction::PassPeeper(__FILE__, 140); // if
    return nullptr;
  }

  Value *TV = SI->getTrueValue();
  Value *FV = SI->getFalseValue();
  if (!(isa<Constant>(TV) || isa<Constant>(FV))) {
    PassPrediction::PassPeeper(__FILE__, 141); // if
    return nullptr;
  }

  // Bool selects with constant operands can be folded to logical ops.
  if (SI->getType()->isIntOrIntVectorTy(1)) {
    PassPrediction::PassPeeper(__FILE__, 142); // if
    return nullptr;
  }

  // If it's a bitcast involving vectors, make sure it has the same number of
  // elements on both sides.
  if (auto *BC = dyn_cast<BitCastInst>(&Op)) {
    PassPrediction::PassPeeper(__FILE__, 143); // if
    VectorType *DestTy = dyn_cast<VectorType>(BC->getDestTy());
    VectorType *SrcTy = dyn_cast<VectorType>(BC->getSrcTy());

    // Verify that either both or neither are vectors.
    if ((SrcTy == nullptr) != (DestTy == nullptr)) {
      PassPrediction::PassPeeper(__FILE__, 144); // if
      return nullptr;
    }

    // If vectors, verify that they have the same number of elements.
    if (SrcTy && SrcTy->getNumElements() != DestTy->getNumElements()) {
      PassPrediction::PassPeeper(__FILE__, 145); // if
      return nullptr;
    }
  }

  // Test if a CmpInst instruction is used exclusively by a select as
  // part of a minimum or maximum operation. If so, refrain from doing
  // any other folding. This helps out other analyses which understand
  // non-obfuscated minimum and maximum idioms, such as ScalarEvolution
  // and CodeGen. And in this case, at least one of the comparison
  // operands has at least one user besides the compare (the select),
  // which would often largely negate the benefit of folding anyway.
  if (auto *CI = dyn_cast<CmpInst>(SI->getCondition())) {
    PassPrediction::PassPeeper(__FILE__, 146); // if
    if (CI->hasOneUse()) {
      PassPrediction::PassPeeper(__FILE__, 147); // if
      Value *Op0 = CI->getOperand(0), *Op1 = CI->getOperand(1);
      if ((SI->getOperand(1) == Op0 && SI->getOperand(2) == Op1) ||
          (SI->getOperand(2) == Op0 && SI->getOperand(1) == Op1)) {
        PassPrediction::PassPeeper(__FILE__, 148); // if
        return nullptr;
      }
    }
  }

  Value *NewTV = foldOperationIntoSelectOperand(Op, TV, Builder);
  Value *NewFV = foldOperationIntoSelectOperand(Op, FV, Builder);
  return SelectInst::Create(SI->getCondition(), NewTV, NewFV, "", nullptr, SI);
}

static Value *foldOperationIntoPhiValue(BinaryOperator *I, Value *InV,
                                        InstCombiner::BuilderTy &Builder) {
  bool ConstIsRHS = isa<Constant>(I->getOperand(1));
  Constant *C = cast<Constant>(I->getOperand(ConstIsRHS));

  if (auto *InC = dyn_cast<Constant>(InV)) {
    PassPrediction::PassPeeper(__FILE__, 149); // if
    if (ConstIsRHS) {
      PassPrediction::PassPeeper(__FILE__, 150); // if
      return ConstantExpr::get(I->getOpcode(), InC, C);
    }
    return ConstantExpr::get(I->getOpcode(), C, InC);
  }

  Value *Op0 = InV, *Op1 = C;
  if (!ConstIsRHS) {
    PassPrediction::PassPeeper(__FILE__, 151); // if
    std::swap(Op0, Op1);
  }

  Value *RI = Builder.CreateBinOp(I->getOpcode(), Op0, Op1, "phitmp");
  auto *FPInst = dyn_cast<Instruction>(RI);
  if (FPInst && isa<FPMathOperator>(FPInst)) {
    PassPrediction::PassPeeper(__FILE__, 152); // if
    FPInst->copyFastMathFlags(I);
  }
  return RI;
}

Instruction *InstCombiner::foldOpIntoPhi(Instruction &I, PHINode *PN) {
  unsigned NumPHIValues = PN->getNumIncomingValues();
  if (NumPHIValues == 0) {
    PassPrediction::PassPeeper(__FILE__, 153); // if
    return nullptr;
  }

  // We normally only transform phis with a single use.  However, if a PHI has
  // multiple uses and they are all the same operation, we can fold *all* of the
  // uses into the PHI.
  if (!PN->hasOneUse()) {
    // Walk the use list for the instruction, comparing them to I.
    PassPrediction::PassPeeper(__FILE__, 154); // if
    for (User *U : PN->users()) {
      PassPrediction::PassPeeper(__FILE__, 155); // for-range
      Instruction *UI = cast<Instruction>(U);
      if (UI != &I && !I.isIdenticalTo(UI)) {
        PassPrediction::PassPeeper(__FILE__, 156); // if
        return nullptr;
      }
    }
    // Otherwise, we can replace *all* users with the new PHI we form.
  }

  // Check to see if all of the operands of the PHI are simple constants
  // (constantint/constantfp/undef).  If there is one non-constant value,
  // remember the BB it is in.  If there is more than one or if *it* is a PHI,
  // bail out.  We don't do arbitrary constant expressions here because moving
  // their computation can be expensive without a cost model.
  BasicBlock *NonConstBB = nullptr;
  for (unsigned i = 0; i != NumPHIValues; ++i) {
    PassPrediction::PassPeeper(__FILE__, 157); // for
    Value *InVal = PN->getIncomingValue(i);
    if (isa<Constant>(InVal) && !isa<ConstantExpr>(InVal)) {
      PassPrediction::PassPeeper(__FILE__, 158); // if
      continue;
    }

    if (isa<PHINode>(InVal)) {
      PassPrediction::PassPeeper(__FILE__, 159); // if
      return nullptr;                            // Itself a phi.
    }
    if (NonConstBB) {
      PassPrediction::PassPeeper(__FILE__, 160); // if
      return nullptr; // More than one non-const value.
    }

    NonConstBB = PN->getIncomingBlock(i);

    // If the InVal is an invoke at the end of the pred block, then we can't
    // insert a computation after it without breaking the edge.
    if (InvokeInst *II = dyn_cast<InvokeInst>(InVal)) {
      PassPrediction::PassPeeper(__FILE__, 161); // if
      if (II->getParent() == NonConstBB) {
        PassPrediction::PassPeeper(__FILE__, 162); // if
        return nullptr;
      }
    }

    // If the incoming non-constant value is in I's block, we will remove one
    // instruction, but insert another equivalent one, leading to infinite
    // instcombine.
    if (isPotentiallyReachable(I.getParent(), NonConstBB, &DT, LI)) {
      PassPrediction::PassPeeper(__FILE__, 163); // if
      return nullptr;
    }
  }

  // If there is exactly one non-constant value, we can insert a copy of the
  // operation in that block.  However, if this is a critical edge, we would be
  // inserting the computation on some other paths (e.g. inside a loop).  Only
  // do this if the pred block is unconditionally branching into the phi block.
  if (NonConstBB != nullptr) {
    PassPrediction::PassPeeper(__FILE__, 164); // if
    BranchInst *BI = dyn_cast<BranchInst>(NonConstBB->getTerminator());
    if (!BI || !BI->isUnconditional()) {
      PassPrediction::PassPeeper(__FILE__, 165); // if
      return nullptr;
    }
  }

  // Okay, we can do the transformation: create the new PHI node.
  PHINode *NewPN = PHINode::Create(I.getType(), PN->getNumIncomingValues());
  InsertNewInstBefore(NewPN, *PN);
  NewPN->takeName(PN);

  // If we are going to have to insert a new computation, do so right before the
  // predecessor's terminator.
  if (NonConstBB) {
    PassPrediction::PassPeeper(__FILE__, 166); // if
    Builder.SetInsertPoint(NonConstBB->getTerminator());
  }

  // Next, add all of the operands to the PHI.
  if (SelectInst *SI = dyn_cast<SelectInst>(&I)) {
    // We only currently try to fold the condition of a select when it is a phi,
    // not the true/false values.
    PassPrediction::PassPeeper(__FILE__, 167); // if
    Value *TrueV = SI->getTrueValue();
    Value *FalseV = SI->getFalseValue();
    BasicBlock *PhiTransBB = PN->getParent();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      PassPrediction::PassPeeper(__FILE__, 168); // for
      BasicBlock *ThisBB = PN->getIncomingBlock(i);
      Value *TrueVInPred = TrueV->DoPHITranslation(PhiTransBB, ThisBB);
      Value *FalseVInPred = FalseV->DoPHITranslation(PhiTransBB, ThisBB);
      Value *InV = nullptr;
      // Beware of ConstantExpr:  it may eventually evaluate to getNullValue,
      // even if currently isNullValue gives false.
      Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i));
      // For vector constants, we cannot use isNullValue to fold into
      // FalseVInPred versus TrueVInPred. When we have individual nonzero
      // elements in the vector, we will incorrectly fold InC to
      // `TrueVInPred`.
      if (InC && !isa<ConstantExpr>(InC) && isa<ConstantInt>(InC)) {
        PassPrediction::PassPeeper(__FILE__, 169); // if
        InV = InC->isNullValue() ? FalseVInPred : TrueVInPred;
      } else {
        // Generate the select in the same block as PN's current incoming block.
        // Note: ThisBB need not be the NonConstBB because vector constants
        // which are constants by definition are handled here.
        // FIXME: This can lead to an increase in IR generation because we might
        // generate selects for vector constant phi operand, that could not be
        // folded to TrueVInPred or FalseVInPred as done for ConstantInt. For
        // non-vector phis, this transformation was always profitable because
        // the select would be generated exactly once in the NonConstBB.
        PassPrediction::PassPeeper(__FILE__, 170); // else
        Builder.SetInsertPoint(ThisBB->getTerminator());
        InV = Builder.CreateSelect(PN->getIncomingValue(i), TrueVInPred,
                                   FalseVInPred, "phitmp");
      }
      NewPN->addIncoming(InV, ThisBB);
    }
  } else if (CmpInst *CI = dyn_cast<CmpInst>(&I)) {
    PassPrediction::PassPeeper(__FILE__, 171); // if
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      PassPrediction::PassPeeper(__FILE__, 172); // for
      Value *InV = nullptr;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        PassPrediction::PassPeeper(__FILE__, 173); // if
        InV = ConstantExpr::getCompare(CI->getPredicate(), InC, C);
      } else if (isa<ICmpInst>(CI)) {
        PassPrediction::PassPeeper(__FILE__, 174); // if
        InV = Builder.CreateICmp(CI->getPredicate(), PN->getIncomingValue(i), C,
                                 "phitmp");
      } else {
        PassPrediction::PassPeeper(__FILE__, 175); // else
        InV = Builder.CreateFCmp(CI->getPredicate(), PN->getIncomingValue(i), C,
                                 "phitmp");
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    PassPrediction::PassPeeper(__FILE__, 176); // if
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      PassPrediction::PassPeeper(__FILE__, 178); // for
      Value *InV =
          foldOperationIntoPhiValue(BO, PN->getIncomingValue(i), Builder);
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else {
    PassPrediction::PassPeeper(__FILE__, 177); // else
    CastInst *CI = cast<CastInst>(&I);
    Type *RetTy = CI->getType();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      PassPrediction::PassPeeper(__FILE__, 179); // for
      Value *InV;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        PassPrediction::PassPeeper(__FILE__, 180); // if
        InV = ConstantExpr::getCast(CI->getOpcode(), InC, RetTy);
      } else {
        PassPrediction::PassPeeper(__FILE__, 181); // else
        InV = Builder.CreateCast(CI->getOpcode(), PN->getIncomingValue(i),
                                 I.getType(), "phitmp");
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  }

  for (auto UI = PN->user_begin(), E = PN->user_end(); UI != E;) {
    PassPrediction::PassPeeper(__FILE__, 182); // for
    Instruction *User = cast<Instruction>(*UI++);
    if (User == &I) {
      PassPrediction::PassPeeper(__FILE__, 183); // if
      continue;
    }
    replaceInstUsesWith(*User, NewPN);
    eraseInstFromFunction(*User);
  }
  return replaceInstUsesWith(I, NewPN);
}

Instruction *InstCombiner::foldOpWithConstantIntoOperand(BinaryOperator &I) {
  assert(isa<Constant>(I.getOperand(1)) && "Unexpected operand type");

  if (auto *Sel = dyn_cast<SelectInst>(I.getOperand(0))) {
    PassPrediction::PassPeeper(__FILE__, 184); // if
    if (Instruction *NewSel = FoldOpIntoSelect(I, Sel)) {
      PassPrediction::PassPeeper(__FILE__, 185); // if
      return NewSel;
    }
  } else if (auto *PN = dyn_cast<PHINode>(I.getOperand(0))) {
    PassPrediction::PassPeeper(__FILE__, 186); // if
    if (Instruction *NewPhi = foldOpIntoPhi(I, PN)) {
      PassPrediction::PassPeeper(__FILE__, 187); // if
      return NewPhi;
    }
  }
  return nullptr;
}

/// Given a pointer type and a constant offset, determine whether or not there
/// is a sequence of GEP indices into the pointed type that will land us at the
/// specified offset. If so, fill them into NewIndices and return the resultant
/// element type, otherwise return null.
Type *InstCombiner::FindElementAtOffset(PointerType *PtrTy, int64_t Offset,
                                        SmallVectorImpl<Value *> &NewIndices) {
  Type *Ty = PtrTy->getElementType();
  if (!Ty->isSized()) {
    PassPrediction::PassPeeper(__FILE__, 188); // if
    return nullptr;
  }

  // Start with the index over the outer type.  Note that the type size
  // might be zero (even if the offset isn't zero) if the indexed type
  // is something like [0 x {int, int}]
  Type *IntPtrTy = DL.getIntPtrType(PtrTy);
  int64_t FirstIdx = 0;
  if (int64_t TySize = DL.getTypeAllocSize(Ty)) {
    PassPrediction::PassPeeper(__FILE__, 189); // if
    FirstIdx = Offset / TySize;
    Offset -= FirstIdx * TySize;

    // Handle hosts where % returns negative instead of values [0..TySize).
    if (Offset < 0) {
      PassPrediction::PassPeeper(__FILE__, 190); // if
      --FirstIdx;
      Offset += TySize;
      assert(Offset >= 0);
    }
    assert((uint64_t)Offset < (uint64_t)TySize && "Out of range offset");
  }

  NewIndices.push_back(ConstantInt::get(IntPtrTy, FirstIdx));

  // Index into the types.  If we fail, set OrigBase to null.
  while (Offset) {
    // Indexing into tail padding between struct/array elements.
    PassPrediction::PassPeeper(__FILE__, 191); // while
    if (uint64_t(Offset * 8) >= DL.getTypeSizeInBits(Ty)) {
      PassPrediction::PassPeeper(__FILE__, 192); // if
      return nullptr;
    }

    if (StructType *STy = dyn_cast<StructType>(Ty)) {
      PassPrediction::PassPeeper(__FILE__, 193); // if
      const StructLayout *SL = DL.getStructLayout(STy);
      assert(Offset < (int64_t)SL->getSizeInBytes() &&
             "Offset must stay within the indexed type");

      unsigned Elt = SL->getElementContainingOffset(Offset);
      NewIndices.push_back(
          ConstantInt::get(Type::getInt32Ty(Ty->getContext()), Elt));

      Offset -= SL->getElementOffset(Elt);
      Ty = STy->getElementType(Elt);
    } else if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
      PassPrediction::PassPeeper(__FILE__, 194); // if
      uint64_t EltSize = DL.getTypeAllocSize(AT->getElementType());
      assert(EltSize && "Cannot index into a zero-sized array");
      NewIndices.push_back(ConstantInt::get(IntPtrTy, Offset / EltSize));
      Offset %= EltSize;
      Ty = AT->getElementType();
    } else {
      // Otherwise, we can't index into the middle of this atomic type, bail.
      PassPrediction::PassPeeper(__FILE__, 195); // else
      return nullptr;
    }
  }

  return Ty;
}

static bool shouldMergeGEPs(GEPOperator &GEP, GEPOperator &Src) {
  // If this GEP has only 0 indices, it is the same pointer as
  // Src. If Src is not a trivial GEP too, don't combine
  // the indices.
  if (GEP.hasAllZeroIndices() && !Src.hasAllZeroIndices() && !Src.hasOneUse()) {
    PassPrediction::PassPeeper(__FILE__, 196); // if
    return false;
  }
  return true;
}

/// Return a value X such that Val = X * Scale, or null if none.
/// If the multiplication is known not to overflow, then NoSignedWrap is set.
Value *InstCombiner::Descale(Value *Val, APInt Scale, bool &NoSignedWrap) {
  assert(isa<IntegerType>(Val->getType()) && "Can only descale integers!");
  assert(cast<IntegerType>(Val->getType())->getBitWidth() ==
             Scale.getBitWidth() &&
         "Scale not compatible with value!");

  // If Val is zero or Scale is one then Val = Val * Scale.
  if (match(Val, m_Zero()) || Scale == 1) {
    PassPrediction::PassPeeper(__FILE__, 197); // if
    NoSignedWrap = true;
    return Val;
  }

  // If Scale is zero then it does not divide Val.
  if (Scale.isMinValue()) {
    PassPrediction::PassPeeper(__FILE__, 198); // if
    return nullptr;
  }

  // Look through chains of multiplications, searching for a constant that is
  // divisible by Scale.  For example, descaling X*(Y*(Z*4)) by a factor of 4
  // will find the constant factor 4 and produce X*(Y*Z).  Descaling X*(Y*8) by
  // a factor of 4 will produce X*(Y*2).  The principle of operation is to bore
  // down from Val:
  //
  //     Val = M1 * X          ||   Analysis starts here and works down
  //      M1 = M2 * Y          ||   Doesn't descend into terms with more
  //      M2 =  Z * 4          \/   than one use
  //
  // Then to modify a term at the bottom:
  //
  //     Val = M1 * X
  //      M1 =  Z * Y          ||   Replaced M2 with Z
  //
  // Then to work back up correcting nsw flags.

  // Op - the term we are currently analyzing.  Starts at Val then drills down.
  // Replaced with its descaled value before exiting from the drill down loop.
  Value *Op = Val;

  // Parent - initially null, but after drilling down notes where Op came from.
  // In the example above, Parent is (Val, 0) when Op is M1, because M1 is the
  // 0'th operand of Val.
  std::pair<Instruction *, unsigned> Parent;

  // Set if the transform requires a descaling at deeper levels that doesn't
  // overflow.
  bool RequireNoSignedWrap = false;

  // Log base 2 of the scale. Negative if not a power of 2.
  int32_t logScale = Scale.exactLogBase2();

  for (;; Op = Parent.first->getOperand(Parent.second)) { // Drill down

    PassPrediction::PassPeeper(__FILE__, 199); // for
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op)) {
      // If Op is a constant divisible by Scale then descale to the quotient.
      PassPrediction::PassPeeper(__FILE__, 200); // if
      APInt Quotient(Scale), Remainder(Scale);   // Init ensures right bitwidth.
      APInt::sdivrem(CI->getValue(), Scale, Quotient, Remainder);
      if (!Remainder.isMinValue()) {
        // Not divisible by Scale.
        PassPrediction::PassPeeper(__FILE__, 201); // if
        return nullptr;
      }
      // Replace with the quotient in the parent.
      Op = ConstantInt::get(CI->getType(), Quotient);
      NoSignedWrap = true;
      PassPrediction::PassPeeper(__FILE__, 202); // break
      break;
    }

    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op)) {

      PassPrediction::PassPeeper(__FILE__, 203); // if
      if (BO->getOpcode() == Instruction::Mul) {
        // Multiplication.
        PassPrediction::PassPeeper(__FILE__, 204); // if
        NoSignedWrap = BO->hasNoSignedWrap();
        if (RequireNoSignedWrap && !NoSignedWrap) {
          PassPrediction::PassPeeper(__FILE__, 205); // if
          return nullptr;
        }

        // There are three cases for multiplication: multiplication by exactly
        // the scale, multiplication by a constant different to the scale, and
        // multiplication by something else.
        Value *LHS = BO->getOperand(0);
        Value *RHS = BO->getOperand(1);

        if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
          // Multiplication by a constant.
          PassPrediction::PassPeeper(__FILE__, 206); // if
          if (CI->getValue() == Scale) {
            // Multiplication by exactly the scale, replace the multiplication
            // by its left-hand side in the parent.
            PassPrediction::PassPeeper(__FILE__, 207); // if
            Op = LHS;
            PassPrediction::PassPeeper(__FILE__, 208); // break
            break;
          }

          // Otherwise drill down into the constant.
          if (!Op->hasOneUse()) {
            PassPrediction::PassPeeper(__FILE__, 209); // if
            return nullptr;
          }

          Parent = std::make_pair(BO, 1);
          continue;
        }

        // Multiplication by something else. Drill down into the left-hand side
        // since that's where the reassociate pass puts the good stuff.
        if (!Op->hasOneUse()) {
          PassPrediction::PassPeeper(__FILE__, 210); // if
          return nullptr;
        }

        Parent = std::make_pair(BO, 0);
        continue;
      }

      if (logScale > 0 && BO->getOpcode() == Instruction::Shl &&
          isa<ConstantInt>(BO->getOperand(1))) {
        // Multiplication by a power of 2.
        PassPrediction::PassPeeper(__FILE__, 211); // if
        NoSignedWrap = BO->hasNoSignedWrap();
        if (RequireNoSignedWrap && !NoSignedWrap) {
          PassPrediction::PassPeeper(__FILE__, 212); // if
          return nullptr;
        }

        Value *LHS = BO->getOperand(0);
        int32_t Amt = cast<ConstantInt>(BO->getOperand(1))
                          ->getLimitedValue(Scale.getBitWidth());
        // Op = LHS << Amt.

        if (Amt == logScale) {
          // Multiplication by exactly the scale, replace the multiplication
          // by its left-hand side in the parent.
          PassPrediction::PassPeeper(__FILE__, 213); // if
          Op = LHS;
          PassPrediction::PassPeeper(__FILE__, 214); // break
          break;
        }
        if (Amt < logScale || !Op->hasOneUse()) {
          PassPrediction::PassPeeper(__FILE__, 215); // if
          return nullptr;
        }

        // Multiplication by more than the scale.  Reduce the multiplying amount
        // by the scale in the parent.
        Parent = std::make_pair(BO, 1);
        Op = ConstantInt::get(BO->getType(), Amt - logScale);
        PassPrediction::PassPeeper(__FILE__, 216); // break
        break;
      }
    }

    if (!Op->hasOneUse()) {
      PassPrediction::PassPeeper(__FILE__, 217); // if
      return nullptr;
    }

    if (CastInst *Cast = dyn_cast<CastInst>(Op)) {
      PassPrediction::PassPeeper(__FILE__, 218); // if
      if (Cast->getOpcode() == Instruction::SExt) {
        // Op is sign-extended from a smaller type, descale in the smaller type.
        PassPrediction::PassPeeper(__FILE__, 219); // if
        unsigned SmallSize = Cast->getSrcTy()->getPrimitiveSizeInBits();
        APInt SmallScale = Scale.trunc(SmallSize);
        // Suppose Op = sext X, and we descale X as Y * SmallScale.  We want to
        // descale Op as (sext Y) * Scale.  In order to have
        //   sext (Y * SmallScale) = (sext Y) * Scale
        // some conditions need to hold however: SmallScale must sign-extend to
        // Scale and the multiplication Y * SmallScale should not overflow.
        if (SmallScale.sext(Scale.getBitWidth()) != Scale) {
          // SmallScale does not sign-extend to Scale.
          PassPrediction::PassPeeper(__FILE__, 220); // if
          return nullptr;
        }
        assert(SmallScale.exactLogBase2() == logScale);
        // Require that Y * SmallScale must not overflow.
        RequireNoSignedWrap = true;

        // Drill down through the cast.
        Parent = std::make_pair(Cast, 0);
        Scale = SmallScale;
        continue;
      }

      if (Cast->getOpcode() == Instruction::Trunc) {
        // Op is truncated from a larger type, descale in the larger type.
        // Suppose Op = trunc X, and we descale X as Y * sext Scale.  Then
        //   trunc (Y * sext Scale) = (trunc Y) * Scale
        // always holds.  However (trunc Y) * Scale may overflow even if
        // trunc (Y * sext Scale) does not, so nsw flags need to be cleared
        // from this point up in the expression (see later).
        PassPrediction::PassPeeper(__FILE__, 221); // if
        if (RequireNoSignedWrap) {
          PassPrediction::PassPeeper(__FILE__, 222); // if
          return nullptr;
        }

        // Drill down through the cast.
        unsigned LargeSize = Cast->getSrcTy()->getPrimitiveSizeInBits();
        Parent = std::make_pair(Cast, 0);
        Scale = Scale.sext(LargeSize);
        if (logScale + 1 ==
            (int32_t)Cast->getType()->getPrimitiveSizeInBits()) {
          PassPrediction::PassPeeper(__FILE__, 223); // if
          logScale = -1;
        }
        assert(Scale.exactLogBase2() == logScale);
        continue;
      }
    }

    // Unsupported expression, bail out.
    return nullptr;
  }

  // If Op is zero then Val = Op * Scale.
  if (match(Op, m_Zero())) {
    PassPrediction::PassPeeper(__FILE__, 224); // if
    NoSignedWrap = true;
    return Op;
  }

  // We know that we can successfully descale, so from here on we can safely
  // modify the IR.  Op holds the descaled version of the deepest term in the
  // expression.  NoSignedWrap is 'true' if multiplying Op by Scale is known
  // not to overflow.

  if (!Parent.first) {
    // The expression only had one term.
    PassPrediction::PassPeeper(__FILE__, 225); // if
    return Op;
  }

  // Rewrite the parent using the descaled version of its operand.
  assert(Parent.first->hasOneUse() && "Drilled down when more than one use!");
  assert(Op != Parent.first->getOperand(Parent.second) &&
         "Descaling was a no-op?");
  Parent.first->setOperand(Parent.second, Op);
  Worklist.Add(Parent.first);

  // Now work back up the expression correcting nsw flags.  The logic is based
  // on the following observation: if X * Y is known not to overflow as a signed
  // multiplication, and Y is replaced by a value Z with smaller absolute value,
  // then X * Z will not overflow as a signed multiplication either.  As we work
  // our way up, having NoSignedWrap 'true' means that the descaled value at the
  // current level has strictly smaller absolute value than the original.
  Instruction *Ancestor = Parent.first;
  do {
    PassPrediction::PassPeeper(__FILE__, 226); // do-while
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Ancestor)) {
      // If the multiplication wasn't nsw then we can't say anything about the
      // value of the descaled multiplication, and we have to clear nsw flags
      // from this point on up.
      PassPrediction::PassPeeper(__FILE__, 227); // if
      bool OpNoSignedWrap = BO->hasNoSignedWrap();
      NoSignedWrap &= OpNoSignedWrap;
      if (NoSignedWrap != OpNoSignedWrap) {
        PassPrediction::PassPeeper(__FILE__, 228); // if
        BO->setHasNoSignedWrap(NoSignedWrap);
        Worklist.Add(Ancestor);
      }
    } else if (Ancestor->getOpcode() == Instruction::Trunc) {
      // The fact that the descaled input to the trunc has smaller absolute
      // value than the original input doesn't tell us anything useful about
      // the absolute values of the truncations.
      PassPrediction::PassPeeper(__FILE__, 229); // if
      NoSignedWrap = false;
    }
    assert((Ancestor->getOpcode() != Instruction::SExt || NoSignedWrap) &&
           "Failed to keep proper track of nsw flags while drilling down?");

    if (Ancestor == Val) {
      // Got to the top, all done!
      PassPrediction::PassPeeper(__FILE__, 230); // if
      return Val;
    }

    // Move up one level in the expression.
    assert(Ancestor->hasOneUse() && "Drilled down when more than one use!");
    Ancestor = Ancestor->user_back();
  } while (1);
}

/// \brief Creates node of binary operation with the same attributes as the
/// specified one but with other operands.
static Value *CreateBinOpAsGiven(BinaryOperator &Inst, Value *LHS, Value *RHS,
                                 InstCombiner::BuilderTy &B) {
  Value *BO = B.CreateBinOp(Inst.getOpcode(), LHS, RHS);
  // If LHS and RHS are constant, BO won't be a binary operator.
  if (BinaryOperator *NewBO = dyn_cast<BinaryOperator>(BO)) {
    PassPrediction::PassPeeper(__FILE__, 231); // if
    NewBO->copyIRFlags(&Inst);
  }
  return BO;
}

/// \brief Makes transformation of binary operation specific for vector types.
/// \param Inst Binary operator to transform.
/// \return Pointer to node that must replace the original binary operator, or
///         null pointer if no transformation was made.
Value *InstCombiner::SimplifyVectorOp(BinaryOperator &Inst) {
  if (!Inst.getType()->isVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 232); // if
    return nullptr;
  }

  // It may not be safe to reorder shuffles and things like div, urem, etc.
  // because we may trap when executing those ops on unknown vector elements.
  // See PR20059.
  if (!isSafeToSpeculativelyExecute(&Inst)) {
    PassPrediction::PassPeeper(__FILE__, 233); // if
    return nullptr;
  }

  unsigned VWidth = cast<VectorType>(Inst.getType())->getNumElements();
  Value *LHS = Inst.getOperand(0), *RHS = Inst.getOperand(1);
  assert(cast<VectorType>(LHS->getType())->getNumElements() == VWidth);
  assert(cast<VectorType>(RHS->getType())->getNumElements() == VWidth);

  // If both arguments of the binary operation are shuffles that use the same
  // mask and shuffle within a single vector, move the shuffle after the binop:
  //   Op(shuffle(v1, m), shuffle(v2, m)) -> shuffle(Op(v1, v2), m)
  auto *LShuf = dyn_cast<ShuffleVectorInst>(LHS);
  auto *RShuf = dyn_cast<ShuffleVectorInst>(RHS);
  if (LShuf && RShuf && LShuf->getMask() == RShuf->getMask() &&
      isa<UndefValue>(LShuf->getOperand(1)) &&
      isa<UndefValue>(RShuf->getOperand(1)) &&
      LShuf->getOperand(0)->getType() == RShuf->getOperand(0)->getType()) {
    PassPrediction::PassPeeper(__FILE__, 234); // if
    Value *NewBO = CreateBinOpAsGiven(Inst, LShuf->getOperand(0),
                                      RShuf->getOperand(0), Builder);
    return Builder.CreateShuffleVector(NewBO, UndefValue::get(NewBO->getType()),
                                       LShuf->getMask());
  }

  // If one argument is a shuffle within one vector, the other is a constant,
  // try moving the shuffle after the binary operation.
  ShuffleVectorInst *Shuffle = nullptr;
  Constant *C1 = nullptr;
  if (isa<ShuffleVectorInst>(LHS)) {
    PassPrediction::PassPeeper(__FILE__, 235); // if
    Shuffle = cast<ShuffleVectorInst>(LHS);
  }
  if (isa<ShuffleVectorInst>(RHS)) {
    PassPrediction::PassPeeper(__FILE__, 236); // if
    Shuffle = cast<ShuffleVectorInst>(RHS);
  }
  if (isa<Constant>(LHS)) {
    PassPrediction::PassPeeper(__FILE__, 237); // if
    C1 = cast<Constant>(LHS);
  }
  if (isa<Constant>(RHS)) {
    PassPrediction::PassPeeper(__FILE__, 238); // if
    C1 = cast<Constant>(RHS);
  }
  if (Shuffle && C1 &&
      (isa<ConstantVector>(C1) || isa<ConstantDataVector>(C1)) &&
      isa<UndefValue>(Shuffle->getOperand(1)) &&
      Shuffle->getType() == Shuffle->getOperand(0)->getType()) {
    PassPrediction::PassPeeper(__FILE__, 239); // if
    SmallVector<int, 16> ShMask = Shuffle->getShuffleMask();
    // Find constant C2 that has property:
    //   shuffle(C2, ShMask) = C1
    // If such constant does not exist (example: ShMask=<0,0> and C1=<1,2>)
    // reorder is not possible.
    SmallVector<Constant *, 16> C2M(
        VWidth, UndefValue::get(C1->getType()->getScalarType()));
    bool MayChange = true;
    for (unsigned I = 0; I < VWidth; ++I) {
      PassPrediction::PassPeeper(__FILE__, 240); // for
      if (ShMask[I] >= 0) {
        assert(ShMask[I] < (int)VWidth);
        if (!isa<UndefValue>(C2M[ShMask[I]])) {
          PassPrediction::PassPeeper(__FILE__, 241); // if
          MayChange = false;
          PassPrediction::PassPeeper(__FILE__, 242); // break
          break;
        }
        C2M[ShMask[I]] = C1->getAggregateElement(I);
      }
    }
    if (MayChange) {
      PassPrediction::PassPeeper(__FILE__, 243); // if
      Constant *C2 = ConstantVector::get(C2M);
      Value *NewLHS = isa<Constant>(LHS) ? C2 : Shuffle->getOperand(0);
      Value *NewRHS = isa<Constant>(LHS) ? Shuffle->getOperand(0) : C2;
      Value *NewBO = CreateBinOpAsGiven(Inst, NewLHS, NewRHS, Builder);
      return Builder.CreateShuffleVector(NewBO, UndefValue::get(Inst.getType()),
                                         Shuffle->getMask());
    }
  }

  return nullptr;
}

Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  SmallVector<Value *, 8> Ops(GEP.op_begin(), GEP.op_end());

  if (Value *V = SimplifyGEPInst(GEP.getSourceElementType(), Ops,
                                 SQ.getWithInstruction(&GEP))) {
    PassPrediction::PassPeeper(__FILE__, 244); // if
    return replaceInstUsesWith(GEP, V);
  }

  Value *PtrOp = GEP.getOperand(0);

  // Eliminate unneeded casts for indices, and replace indices which displace
  // by multiples of a zero size type with zero.
  bool MadeChange = false;
  Type *IntPtrTy =
      DL.getIntPtrType(GEP.getPointerOperandType()->getScalarType());

  gep_type_iterator GTI = gep_type_begin(GEP);
  for (User::op_iterator I = GEP.op_begin() + 1, E = GEP.op_end(); I != E;
       ++I, ++GTI) {
    // Skip indices into struct types.
    PassPrediction::PassPeeper(__FILE__, 245); // for
    if (GTI.isStruct()) {
      PassPrediction::PassPeeper(__FILE__, 246); // if
      continue;
    }

    // Index type should have the same width as IntPtr
    Type *IndexTy = (*I)->getType();
    Type *NewIndexType =
        IndexTy->isVectorTy()
            ? VectorType::get(IntPtrTy, IndexTy->getVectorNumElements())
            : IntPtrTy;

    // If the element type has zero size then any index over it is equivalent
    // to an index of zero, so replace it with zero if it is not zero already.
    Type *EltTy = GTI.getIndexedType();
    if (EltTy->isSized() && DL.getTypeAllocSize(EltTy) == 0) {
      PassPrediction::PassPeeper(__FILE__, 247); // if
      if (!isa<Constant>(*I) || !cast<Constant>(*I)->isNullValue()) {
        PassPrediction::PassPeeper(__FILE__, 248); // if
        *I = Constant::getNullValue(NewIndexType);
        MadeChange = true;
      }
    }

    if (IndexTy != NewIndexType) {
      // If we are using a wider index than needed for this platform, shrink
      // it to what we need.  If narrower, sign-extend it to what we need.
      // This explicit cast can make subsequent optimizations more obvious.
      PassPrediction::PassPeeper(__FILE__, 249); // if
      *I = Builder.CreateIntCast(*I, NewIndexType, true);
      MadeChange = true;
    }
  }
  if (MadeChange) {
    PassPrediction::PassPeeper(__FILE__, 250); // if
    return &GEP;
  }

  // Check to see if the inputs to the PHI node are getelementptr instructions.
  if (PHINode *PN = dyn_cast<PHINode>(PtrOp)) {
    PassPrediction::PassPeeper(__FILE__, 251); // if
    GetElementPtrInst *Op1 = dyn_cast<GetElementPtrInst>(PN->getOperand(0));
    if (!Op1) {
      PassPrediction::PassPeeper(__FILE__, 252); // if
      return nullptr;
    }

    // Don't fold a GEP into itself through a PHI node. This can only happen
    // through the back-edge of a loop. Folding a GEP into itself means that
    // the value of the previous iteration needs to be stored in the meantime,
    // thus requiring an additional register variable to be live, but not
    // actually achieving anything (the GEP still needs to be executed once per
    // loop iteration).
    if (Op1 == &GEP) {
      PassPrediction::PassPeeper(__FILE__, 253); // if
      return nullptr;
    }

    int DI = -1;

    for (auto I = PN->op_begin() + 1, E = PN->op_end(); I != E; ++I) {
      PassPrediction::PassPeeper(__FILE__, 254); // for
      GetElementPtrInst *Op2 = dyn_cast<GetElementPtrInst>(*I);
      if (!Op2 || Op1->getNumOperands() != Op2->getNumOperands()) {
        PassPrediction::PassPeeper(__FILE__, 255); // if
        return nullptr;
      }

      // As for Op1 above, don't try to fold a GEP into itself.
      if (Op2 == &GEP) {
        PassPrediction::PassPeeper(__FILE__, 256); // if
        return nullptr;
      }

      // Keep track of the type as we walk the GEP.
      Type *CurTy = nullptr;

      for (unsigned J = 0, F = Op1->getNumOperands(); J != F; ++J) {
        PassPrediction::PassPeeper(__FILE__, 257); // for
        if (Op1->getOperand(J)->getType() != Op2->getOperand(J)->getType()) {
          PassPrediction::PassPeeper(__FILE__, 258); // if
          return nullptr;
        }

        if (Op1->getOperand(J) != Op2->getOperand(J)) {
          PassPrediction::PassPeeper(__FILE__, 259); // if
          if (DI == -1) {
            // We have not seen any differences yet in the GEPs feeding the
            // PHI yet, so we record this one if it is allowed to be a
            // variable.

            // The first two arguments can vary for any GEP, the rest have to be
            // static for struct slots
            PassPrediction::PassPeeper(__FILE__, 260); // if
            if (J > 1 && CurTy->isStructTy()) {
              PassPrediction::PassPeeper(__FILE__, 262); // if
              return nullptr;
            }

            DI = J;
          } else {
            // The GEP is different by more than one input. While this could be
            // extended to support GEPs that vary by more than one variable it
            // doesn't make sense since it greatly increases the complexity and
            // would result in an R+R+R addressing mode which no backend
            // directly supports and would need to be broken into several
            // simpler instructions anyway.
            PassPrediction::PassPeeper(__FILE__, 261); // else
            return nullptr;
          }
        }

        // Sink down a layer of the type for the next iteration.
        if (J > 0) {
          PassPrediction::PassPeeper(__FILE__, 263); // if
          if (J == 1) {
            PassPrediction::PassPeeper(__FILE__, 264); // if
            CurTy = Op1->getSourceElementType();
          } else if (CompositeType *CT = dyn_cast<CompositeType>(CurTy)) {
            PassPrediction::PassPeeper(__FILE__, 265); // if
            CurTy = CT->getTypeAtIndex(Op1->getOperand(J));
          } else {
            PassPrediction::PassPeeper(__FILE__, 266); // else
            CurTy = nullptr;
          }
        }
      }
    }

    // If not all GEPs are identical we'll have to create a new PHI node.
    // Check that the old PHI node has only one use so that it will get
    // removed.
    if (DI != -1 && !PN->hasOneUse()) {
      PassPrediction::PassPeeper(__FILE__, 267); // if
      return nullptr;
    }

    GetElementPtrInst *NewGEP = cast<GetElementPtrInst>(Op1->clone());
    if (DI == -1) {
      // All the GEPs feeding the PHI are identical. Clone one down into our
      // BB so that it can be merged with the current GEP.
      PassPrediction::PassPeeper(__FILE__, 268); // if
      GEP.getParent()->getInstList().insert(
          GEP.getParent()->getFirstInsertionPt(), NewGEP);
    } else {
      // All the GEPs feeding the PHI differ at a single offset. Clone a GEP
      // into the current block so it can be merged, and create a new PHI to
      // set that index.
      PassPrediction::PassPeeper(__FILE__, 269); // else
      PHINode *NewPN;
      {
        IRBuilderBase::InsertPointGuard Guard(Builder);
        Builder.SetInsertPoint(PN);
        NewPN = Builder.CreatePHI(Op1->getOperand(DI)->getType(),
                                  PN->getNumOperands());
      }

      for (auto &I : PN->operands()) {
        PassPrediction::PassPeeper(__FILE__, 270); // for-range
        NewPN->addIncoming(cast<GEPOperator>(I)->getOperand(DI),
                           PN->getIncomingBlock(I));
      }

      NewGEP->setOperand(DI, NewPN);
      GEP.getParent()->getInstList().insert(
          GEP.getParent()->getFirstInsertionPt(), NewGEP);
      NewGEP->setOperand(DI, NewPN);
    }

    GEP.setOperand(0, NewGEP);
    PtrOp = NewGEP;
  }

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  if (GEPOperator *Src = dyn_cast<GEPOperator>(PtrOp)) {
    PassPrediction::PassPeeper(__FILE__, 271); // if
    if (!shouldMergeGEPs(*cast<GEPOperator>(&GEP), *Src)) {
      PassPrediction::PassPeeper(__FILE__, 272); // if
      return nullptr;
    }

    // Note that if our source is a gep chain itself then we wait for that
    // chain to be resolved before we perform this transformation.  This
    // avoids us creating a TON of code in some cases.
    if (GEPOperator *SrcGEP = dyn_cast<GEPOperator>(Src->getOperand(0))) {
      PassPrediction::PassPeeper(__FILE__, 273); // if
      if (SrcGEP->getNumOperands() == 2 && shouldMergeGEPs(*Src, *SrcGEP)) {
        PassPrediction::PassPeeper(__FILE__, 274); // if
        return nullptr; // Wait until our source is folded to completion.
      }
    }

    SmallVector<Value *, 8> Indices;

    // Find out whether the last index in the source GEP is a sequential idx.
    bool EndsWithSequential = false;
    for (gep_type_iterator I = gep_type_begin(*Src), E = gep_type_end(*Src);
         I != E; ++I) {
      PassPrediction::PassPeeper(__FILE__, 275); // for
      EndsWithSequential = I.isSequential();
    }

    // Can we combine the two pointer arithmetics offsets?
    if (EndsWithSequential) {
      // Replace: gep (gep %P, long B), long A, ...
      // With:    T = long A+B; gep %P, T, ...
      //
      PassPrediction::PassPeeper(__FILE__, 276); // if
      Value *SO1 = Src->getOperand(Src->getNumOperands() - 1);
      Value *GO1 = GEP.getOperand(1);

      // If they aren't the same type, then the input hasn't been processed
      // by the loop above yet (which canonicalizes sequential index types to
      // intptr_t).  Just avoid transforming this until the input has been
      // normalized.
      if (SO1->getType() != GO1->getType()) {
        PassPrediction::PassPeeper(__FILE__, 277); // if
        return nullptr;
      }

      Value *Sum =
          SimplifyAddInst(GO1, SO1, false, false, SQ.getWithInstruction(&GEP));
      // Only do the combine when we are sure the cost after the
      // merge is never more than that before the merge.
      if (Sum == nullptr) {
        PassPrediction::PassPeeper(__FILE__, 278); // if
        return nullptr;
      }

      // Update the GEP in place if possible.
      if (Src->getNumOperands() == 2) {
        PassPrediction::PassPeeper(__FILE__, 279); // if
        GEP.setOperand(0, Src->getOperand(0));
        GEP.setOperand(1, Sum);
        return &GEP;
      }
      Indices.append(Src->op_begin() + 1, Src->op_end() - 1);
      Indices.push_back(Sum);
      Indices.append(GEP.op_begin() + 2, GEP.op_end());
    } else if (isa<Constant>(*GEP.idx_begin()) &&
               cast<Constant>(*GEP.idx_begin())->isNullValue() &&
               Src->getNumOperands() != 1) {
      // Otherwise we can do the fold if the first index of the GEP is a zero
      PassPrediction::PassPeeper(__FILE__, 280); // if
      Indices.append(Src->op_begin() + 1, Src->op_end());
      Indices.append(GEP.idx_begin() + 1, GEP.idx_end());
    }

    if (!Indices.empty()) {
      PassPrediction::PassPeeper(__FILE__, 281); // if
      return GEP.isInBounds() && Src->isInBounds()
                 ? GetElementPtrInst::CreateInBounds(
                       Src->getSourceElementType(), Src->getOperand(0), Indices,
                       GEP.getName())
                 : GetElementPtrInst::Create(Src->getSourceElementType(),
                                             Src->getOperand(0), Indices,
                                             GEP.getName());
    }
  }

  if (GEP.getNumIndices() == 1) {
    PassPrediction::PassPeeper(__FILE__, 282); // if
    unsigned AS = GEP.getPointerAddressSpace();
    if (GEP.getOperand(1)->getType()->getScalarSizeInBits() ==
        DL.getPointerSizeInBits(AS)) {
      PassPrediction::PassPeeper(__FILE__, 283); // if
      Type *Ty = GEP.getSourceElementType();
      uint64_t TyAllocSize = DL.getTypeAllocSize(Ty);

      bool Matched = false;
      uint64_t C;
      Value *V = nullptr;
      if (TyAllocSize == 1) {
        PassPrediction::PassPeeper(__FILE__, 284); // if
        V = GEP.getOperand(1);
        Matched = true;
      } else if (match(GEP.getOperand(1),
                       m_AShr(m_Value(V), m_ConstantInt(C)))) {
        PassPrediction::PassPeeper(__FILE__, 285); // if
        if (TyAllocSize == 1ULL << C) {
          PassPrediction::PassPeeper(__FILE__, 286); // if
          Matched = true;
        }
      } else if (match(GEP.getOperand(1),
                       m_SDiv(m_Value(V), m_ConstantInt(C)))) {
        PassPrediction::PassPeeper(__FILE__, 287); // if
        if (TyAllocSize == C) {
          PassPrediction::PassPeeper(__FILE__, 288); // if
          Matched = true;
        }
      }

      if (Matched) {
        // Canonicalize (gep i8* X, -(ptrtoint Y))
        // to (inttoptr (sub (ptrtoint X), (ptrtoint Y)))
        // The GEP pattern is emitted by the SCEV expander for certain kinds of
        // pointer arithmetic.
        PassPrediction::PassPeeper(__FILE__, 289); // if
        if (match(V, m_Neg(m_PtrToInt(m_Value())))) {
          PassPrediction::PassPeeper(__FILE__, 290); // if
          Operator *Index = cast<Operator>(V);
          Value *PtrToInt = Builder.CreatePtrToInt(PtrOp, Index->getType());
          Value *NewSub = Builder.CreateSub(PtrToInt, Index->getOperand(1));
          return CastInst::Create(Instruction::IntToPtr, NewSub, GEP.getType());
        }
        // Canonicalize (gep i8* X, (ptrtoint Y)-(ptrtoint X))
        // to (bitcast Y)
        Value *Y;
        if (match(V, m_Sub(m_PtrToInt(m_Value(Y)),
                           m_PtrToInt(m_Specific(GEP.getOperand(0)))))) {
          PassPrediction::PassPeeper(__FILE__, 291); // if
          return CastInst::CreatePointerBitCastOrAddrSpaceCast(Y,
                                                               GEP.getType());
        }
      }
    }
  }

  // We do not handle pointer-vector geps here.
  if (GEP.getType()->isVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 292); // if
    return nullptr;
  }

  // Handle gep(bitcast x) and gep(gep x, 0, 0, 0).
  Value *StrippedPtr = PtrOp->stripPointerCasts();
  PointerType *StrippedPtrTy = cast<PointerType>(StrippedPtr->getType());

  if (StrippedPtr != PtrOp) {
    PassPrediction::PassPeeper(__FILE__, 293); // if
    bool HasZeroPointerIndex = false;
    if (ConstantInt *C = dyn_cast<ConstantInt>(GEP.getOperand(1))) {
      PassPrediction::PassPeeper(__FILE__, 294); // if
      HasZeroPointerIndex = C->isZero();
    }

    // Transform: GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ...
    // into     : GEP [10 x i8]* X, i32 0, ...
    //
    // Likewise, transform: GEP (bitcast i8* X to [0 x i8]*), i32 0, ...
    //           into     : GEP i8* X, ...
    //
    // This occurs when the program declares an array extern like "int X[];"
    if (HasZeroPointerIndex) {
      PassPrediction::PassPeeper(__FILE__, 295); // if
      if (ArrayType *CATy = dyn_cast<ArrayType>(GEP.getSourceElementType())) {
        // GEP (bitcast i8* X to [0 x i8]*), i32 0, ... ?
        PassPrediction::PassPeeper(__FILE__, 296); // if
        if (CATy->getElementType() == StrippedPtrTy->getElementType()) {
          // -> GEP i8* X, ...
          PassPrediction::PassPeeper(__FILE__, 297); // if
          SmallVector<Value *, 8> Idx(GEP.idx_begin() + 1, GEP.idx_end());
          GetElementPtrInst *Res = GetElementPtrInst::Create(
              StrippedPtrTy->getElementType(), StrippedPtr, Idx, GEP.getName());
          Res->setIsInBounds(GEP.isInBounds());
          if (StrippedPtrTy->getAddressSpace() == GEP.getAddressSpace()) {
            PassPrediction::PassPeeper(__FILE__, 298); // if
            return Res;
          }
          // Insert Res, and create an addrspacecast.
          // e.g.,
          // GEP (addrspacecast i8 addrspace(1)* X to [0 x i8]*), i32 0, ...
          // ->
          // %0 = GEP i8 addrspace(1)* X, ...
          // addrspacecast i8 addrspace(1)* %0 to i8*
          return new AddrSpaceCastInst(Builder.Insert(Res), GEP.getType());
        }

        if (ArrayType *XATy =
                dyn_cast<ArrayType>(StrippedPtrTy->getElementType())) {
          // GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ... ?
          PassPrediction::PassPeeper(__FILE__, 299); // if
          if (CATy->getElementType() == XATy->getElementType()) {
            // -> GEP [10 x i8]* X, i32 0, ...
            // At this point, we know that the cast source type is a pointer
            // to an array of the same type as the destination pointer
            // array.  Because the array type is never stepped over (there
            // is a leading zero) we can fold the cast into this GEP.
            PassPrediction::PassPeeper(__FILE__, 300); // if
            if (StrippedPtrTy->getAddressSpace() == GEP.getAddressSpace()) {
              PassPrediction::PassPeeper(__FILE__, 301); // if
              GEP.setOperand(0, StrippedPtr);
              GEP.setSourceElementType(XATy);
              return &GEP;
            }
            // Cannot replace the base pointer directly because StrippedPtr's
            // address space is different. Instead, create a new GEP followed by
            // an addrspacecast.
            // e.g.,
            // GEP (addrspacecast [10 x i8] addrspace(1)* X to [0 x i8]*),
            //   i32 0, ...
            // ->
            // %0 = GEP [10 x i8] addrspace(1)* X, ...
            // addrspacecast i8 addrspace(1)* %0 to i8*
            SmallVector<Value *, 8> Idx(GEP.idx_begin(), GEP.idx_end());
            Value *NewGEP = GEP.isInBounds()
                                ? Builder.CreateInBoundsGEP(
                                      nullptr, StrippedPtr, Idx, GEP.getName())
                                : Builder.CreateGEP(nullptr, StrippedPtr, Idx,
                                                    GEP.getName());
            return new AddrSpaceCastInst(NewGEP, GEP.getType());
          }
        }
      }
    } else if (GEP.getNumOperands() == 2) {
      // Transform things like:
      // %t = getelementptr i32* bitcast ([2 x i32]* %str to i32*), i32 %V
      // into:  %t1 = getelementptr [2 x i32]* %str, i32 0, i32 %V; bitcast
      PassPrediction::PassPeeper(__FILE__, 302); // if
      Type *SrcElTy = StrippedPtrTy->getElementType();
      Type *ResElTy = GEP.getSourceElementType();
      if (SrcElTy->isArrayTy() &&
          DL.getTypeAllocSize(SrcElTy->getArrayElementType()) ==
              DL.getTypeAllocSize(ResElTy)) {
        PassPrediction::PassPeeper(__FILE__, 303); // if
        Type *IdxType = DL.getIntPtrType(GEP.getType());
        Value *Idx[2] = {Constant::getNullValue(IdxType), GEP.getOperand(1)};
        Value *NewGEP =
            GEP.isInBounds()
                ? Builder.CreateInBoundsGEP(nullptr, StrippedPtr, Idx,
                                            GEP.getName())
                : Builder.CreateGEP(nullptr, StrippedPtr, Idx, GEP.getName());

        // V and GEP are both pointer types --> BitCast
        return CastInst::CreatePointerBitCastOrAddrSpaceCast(NewGEP,
                                                             GEP.getType());
      }

      // Transform things like:
      // %V = mul i64 %N, 4
      // %t = getelementptr i8* bitcast (i32* %arr to i8*), i32 %V
      // into:  %t1 = getelementptr i32* %arr, i32 %N; bitcast
      if (ResElTy->isSized() && SrcElTy->isSized()) {
        // Check that changing the type amounts to dividing the index by a scale
        // factor.
        PassPrediction::PassPeeper(__FILE__, 304); // if
        uint64_t ResSize = DL.getTypeAllocSize(ResElTy);
        uint64_t SrcSize = DL.getTypeAllocSize(SrcElTy);
        if (ResSize && SrcSize % ResSize == 0) {
          PassPrediction::PassPeeper(__FILE__, 305); // if
          Value *Idx = GEP.getOperand(1);
          unsigned BitWidth = Idx->getType()->getPrimitiveSizeInBits();
          uint64_t Scale = SrcSize / ResSize;

          // Earlier transforms ensure that the index has type IntPtrType, which
          // considerably simplifies the logic by eliminating implicit casts.
          assert(Idx->getType() == DL.getIntPtrType(GEP.getType()) &&
                 "Index not cast to pointer width?");

          bool NSW;
          if (Value *NewIdx = Descale(Idx, APInt(BitWidth, Scale), NSW)) {
            // Successfully decomposed Idx as NewIdx * Scale, form a new GEP.
            // If the multiplication NewIdx * Scale may overflow then the new
            // GEP may not be "inbounds".
            PassPrediction::PassPeeper(__FILE__, 306); // if
            Value *NewGEP =
                GEP.isInBounds() && NSW
                    ? Builder.CreateInBoundsGEP(nullptr, StrippedPtr, NewIdx,
                                                GEP.getName())
                    : Builder.CreateGEP(nullptr, StrippedPtr, NewIdx,
                                        GEP.getName());

            // The NewGEP must be pointer typed, so must the old one -> BitCast
            return CastInst::CreatePointerBitCastOrAddrSpaceCast(NewGEP,
                                                                 GEP.getType());
          }
        }
      }

      // Similarly, transform things like:
      // getelementptr i8* bitcast ([100 x double]* X to i8*), i32 %tmp
      //   (where tmp = 8*tmp2) into:
      // getelementptr [100 x double]* %arr, i32 0, i32 %tmp2; bitcast
      if (ResElTy->isSized() && SrcElTy->isSized() && SrcElTy->isArrayTy()) {
        // Check that changing to the array element type amounts to dividing the
        // index by a scale factor.
        PassPrediction::PassPeeper(__FILE__, 307); // if
        uint64_t ResSize = DL.getTypeAllocSize(ResElTy);
        uint64_t ArrayEltSize =
            DL.getTypeAllocSize(SrcElTy->getArrayElementType());
        if (ResSize && ArrayEltSize % ResSize == 0) {
          PassPrediction::PassPeeper(__FILE__, 308); // if
          Value *Idx = GEP.getOperand(1);
          unsigned BitWidth = Idx->getType()->getPrimitiveSizeInBits();
          uint64_t Scale = ArrayEltSize / ResSize;

          // Earlier transforms ensure that the index has type IntPtrType, which
          // considerably simplifies the logic by eliminating implicit casts.
          assert(Idx->getType() == DL.getIntPtrType(GEP.getType()) &&
                 "Index not cast to pointer width?");

          bool NSW;
          if (Value *NewIdx = Descale(Idx, APInt(BitWidth, Scale), NSW)) {
            // Successfully decomposed Idx as NewIdx * Scale, form a new GEP.
            // If the multiplication NewIdx * Scale may overflow then the new
            // GEP may not be "inbounds".
            PassPrediction::PassPeeper(__FILE__, 309); // if
            Value *Off[2] = {
                Constant::getNullValue(DL.getIntPtrType(GEP.getType())),
                NewIdx};

            Value *NewGEP = GEP.isInBounds() && NSW
                                ? Builder.CreateInBoundsGEP(
                                      SrcElTy, StrippedPtr, Off, GEP.getName())
                                : Builder.CreateGEP(SrcElTy, StrippedPtr, Off,
                                                    GEP.getName());
            // The NewGEP must be pointer typed, so must the old one -> BitCast
            return CastInst::CreatePointerBitCastOrAddrSpaceCast(NewGEP,
                                                                 GEP.getType());
          }
        }
      }
    }
  }

  // addrspacecast between types is canonicalized as a bitcast, then an
  // addrspacecast. To take advantage of the below bitcast + struct GEP, look
  // through the addrspacecast.
  if (AddrSpaceCastInst *ASC = dyn_cast<AddrSpaceCastInst>(PtrOp)) {
    //   X = bitcast A addrspace(1)* to B addrspace(1)*
    //   Y = addrspacecast A addrspace(1)* to B addrspace(2)*
    //   Z = gep Y, <...constant indices...>
    // Into an addrspacecasted GEP of the struct.
    PassPrediction::PassPeeper(__FILE__, 310); // if
    if (BitCastInst *BC = dyn_cast<BitCastInst>(ASC->getOperand(0))) {
      PassPrediction::PassPeeper(__FILE__, 311); // if
      PtrOp = BC;
    }
  }

  /// See if we can simplify:
  ///   X = bitcast A* to B*
  ///   Y = gep X, <...constant indices...>
  /// into a gep of the original struct.  This is important for SROA and alias
  /// analysis of unions.  If "A" is also a bitcast, wait for A/X to be merged.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(PtrOp)) {
    PassPrediction::PassPeeper(__FILE__, 312); // if
    Value *Operand = BCI->getOperand(0);
    PointerType *OpType = cast<PointerType>(Operand->getType());
    unsigned OffsetBits = DL.getPointerTypeSizeInBits(GEP.getType());
    APInt Offset(OffsetBits, 0);
    if (!isa<BitCastInst>(Operand) &&
        GEP.accumulateConstantOffset(DL, Offset)) {

      // If this GEP instruction doesn't move the pointer, just replace the GEP
      // with a bitcast of the real input to the dest type.
      PassPrediction::PassPeeper(__FILE__, 313); // if
      if (!Offset) {
        // If the bitcast is of an allocation, and the allocation will be
        // converted to match the type of the cast, don't touch this.
        PassPrediction::PassPeeper(__FILE__, 314); // if
        if (isa<AllocaInst>(Operand) || isAllocationFn(Operand, &TLI)) {
          // See if the bitcast simplifies, if so, don't nuke this GEP yet.
          PassPrediction::PassPeeper(__FILE__, 315); // if
          if (Instruction *I = visitBitCast(*BCI)) {
            PassPrediction::PassPeeper(__FILE__, 316); // if
            if (I != BCI) {
              PassPrediction::PassPeeper(__FILE__, 317); // if
              I->takeName(BCI);
              BCI->getParent()->getInstList().insert(BCI->getIterator(), I);
              replaceInstUsesWith(*BCI, I);
            }
            return &GEP;
          }
        }

        if (Operand->getType()->getPointerAddressSpace() !=
            GEP.getAddressSpace()) {
          PassPrediction::PassPeeper(__FILE__, 318); // if
          return new AddrSpaceCastInst(Operand, GEP.getType());
        }
        return new BitCastInst(Operand, GEP.getType());
      }

      // Otherwise, if the offset is non-zero, we need to find out if there is a
      // field at Offset in 'A's type.  If so, we can pull the cast through the
      // GEP.
      SmallVector<Value *, 8> NewIndices;
      if (FindElementAtOffset(OpType, Offset.getSExtValue(), NewIndices)) {
        PassPrediction::PassPeeper(__FILE__, 319); // if
        Value *NGEP =
            GEP.isInBounds()
                ? Builder.CreateInBoundsGEP(nullptr, Operand, NewIndices)
                : Builder.CreateGEP(nullptr, Operand, NewIndices);

        if (NGEP->getType() == GEP.getType()) {
          PassPrediction::PassPeeper(__FILE__, 320); // if
          return replaceInstUsesWith(GEP, NGEP);
        }
        NGEP->takeName(&GEP);

        if (NGEP->getType()->getPointerAddressSpace() !=
            GEP.getAddressSpace()) {
          PassPrediction::PassPeeper(__FILE__, 321); // if
          return new AddrSpaceCastInst(NGEP, GEP.getType());
        }
        return new BitCastInst(NGEP, GEP.getType());
      }
    }
  }

  if (!GEP.isInBounds()) {
    PassPrediction::PassPeeper(__FILE__, 322); // if
    unsigned PtrWidth =
        DL.getPointerSizeInBits(PtrOp->getType()->getPointerAddressSpace());
    APInt BasePtrOffset(PtrWidth, 0);
    Value *UnderlyingPtrOp =
        PtrOp->stripAndAccumulateInBoundsConstantOffsets(DL, BasePtrOffset);
    if (auto *AI = dyn_cast<AllocaInst>(UnderlyingPtrOp)) {
      PassPrediction::PassPeeper(__FILE__, 323); // if
      if (GEP.accumulateConstantOffset(DL, BasePtrOffset) &&
          BasePtrOffset.isNonNegative()) {
        PassPrediction::PassPeeper(__FILE__, 324); // if
        APInt AllocSize(PtrWidth, DL.getTypeAllocSize(AI->getAllocatedType()));
        if (BasePtrOffset.ule(AllocSize)) {
          PassPrediction::PassPeeper(__FILE__, 325); // if
          return GetElementPtrInst::CreateInBounds(
              PtrOp, makeArrayRef(Ops).slice(1), GEP.getName());
        }
      }
    }
  }

  return nullptr;
}

static bool isNeverEqualToUnescapedAlloc(Value *V, const TargetLibraryInfo *TLI,
                                         Instruction *AI) {
  if (isa<ConstantPointerNull>(V)) {
    PassPrediction::PassPeeper(__FILE__, 326); // if
    return true;
  }
  if (auto *LI = dyn_cast<LoadInst>(V)) {
    PassPrediction::PassPeeper(__FILE__, 327); // if
    return isa<GlobalVariable>(LI->getPointerOperand());
  }
  // Two distinct allocations will never be equal.
  // We rely on LookThroughBitCast in isAllocLikeFn being false, since looking
  // through bitcasts of V can cause
  // the result statement below to be true, even when AI and V (ex:
  // i8* ->i32* ->i8* of AI) are the same allocations.
  return isAllocLikeFn(V, TLI) && V != AI;
}

static bool isAllocSiteRemovable(Instruction *AI,
                                 SmallVectorImpl<WeakTrackingVH> &Users,
                                 const TargetLibraryInfo *TLI) {
  SmallVector<Instruction *, 4> Worklist;
  Worklist.push_back(AI);

  do {
    PassPrediction::PassPeeper(__FILE__, 328); // do-while
    Instruction *PI = Worklist.pop_back_val();
    for (User *U : PI->users()) {
      PassPrediction::PassPeeper(__FILE__, 329); // for-range
      Instruction *I = cast<Instruction>(U);
      switch (I->getOpcode()) {
      default:
        // Give up the moment we see something we can't handle.
        return false;

      case Instruction::AddrSpaceCast:
        PassPrediction::PassPeeper(__FILE__, 330); // case

      case Instruction::BitCast:
        PassPrediction::PassPeeper(__FILE__, 331); // case

      case Instruction::GetElementPtr:
        PassPrediction::PassPeeper(__FILE__, 332); // case

        Users.emplace_back(I);
        Worklist.push_back(I);
        continue;

      case Instruction::ICmp:
        PassPrediction::PassPeeper(__FILE__, 333); // case
        {
          ICmpInst *ICI = cast<ICmpInst>(I);
          // We can fold eq/ne comparisons with null to false/true,
          // respectively. We also fold comparisons in some conditions provided
          // the alloc has not escaped (see isNeverEqualToUnescapedAlloc).
          if (!ICI->isEquality()) {
            PassPrediction::PassPeeper(__FILE__, 334); // if
            return false;
          }
          unsigned OtherIndex = (ICI->getOperand(0) == PI) ? 1 : 0;
          if (!isNeverEqualToUnescapedAlloc(ICI->getOperand(OtherIndex), TLI,
                                            AI)) {
            PassPrediction::PassPeeper(__FILE__, 335); // if
            return false;
          }
          Users.emplace_back(I);
          continue;
        }

      case Instruction::Call:
        PassPrediction::PassPeeper(__FILE__, 336); // case

        // Ignore no-op and store intrinsics.
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
          PassPrediction::PassPeeper(__FILE__, 337); // if
          switch (II->getIntrinsicID()) {
          default:
            return false;

          case Intrinsic::memmove:
            PassPrediction::PassPeeper(__FILE__, 338); // case

          case Intrinsic::memcpy:
            PassPrediction::PassPeeper(__FILE__, 339); // case

          case Intrinsic::memset:
            PassPrediction::PassPeeper(__FILE__, 340); // case
            {
              MemIntrinsic *MI = cast<MemIntrinsic>(II);
              if (MI->isVolatile() || MI->getRawDest() != PI) {
                PassPrediction::PassPeeper(__FILE__, 341); // if
                return false;
              }
              LLVM_FALLTHROUGH;
            }
          case Intrinsic::dbg_declare:
            PassPrediction::PassPeeper(__FILE__, 342); // case

          case Intrinsic::dbg_value:
            PassPrediction::PassPeeper(__FILE__, 343); // case

          case Intrinsic::invariant_start:
            PassPrediction::PassPeeper(__FILE__, 344); // case

          case Intrinsic::invariant_end:
            PassPrediction::PassPeeper(__FILE__, 345); // case

          case Intrinsic::lifetime_start:
            PassPrediction::PassPeeper(__FILE__, 346); // case

          case Intrinsic::lifetime_end:
            PassPrediction::PassPeeper(__FILE__, 347); // case

          case Intrinsic::objectsize:
            PassPrediction::PassPeeper(__FILE__, 348); // case

            Users.emplace_back(I);
            continue;
          }
        }

        if (isFreeCall(I, TLI)) {
          PassPrediction::PassPeeper(__FILE__, 349); // if
          Users.emplace_back(I);
          continue;
        }
        return false;

      case Instruction::Store:
        PassPrediction::PassPeeper(__FILE__, 350); // case
        {
          StoreInst *SI = cast<StoreInst>(I);
          if (SI->isVolatile() || SI->getPointerOperand() != PI) {
            PassPrediction::PassPeeper(__FILE__, 351); // if
            return false;
          }
          Users.emplace_back(I);
          continue;
        }
      }
      llvm_unreachable("missing a return?");
    }
  } while (!Worklist.empty());
  return true;
}

Instruction *InstCombiner::visitAllocSite(Instruction &MI) {
  // If we have a malloc call which is only used in any amount of comparisons
  // to null and free calls, delete the calls and replace the comparisons with
  // true or false as appropriate.
  SmallVector<WeakTrackingVH, 64> Users;
  if (isAllocSiteRemovable(&MI, Users, &TLI)) {
    PassPrediction::PassPeeper(__FILE__, 352); // if
    for (unsigned i = 0, e = Users.size(); i != e; ++i) {
      // Lowering all @llvm.objectsize calls first because they may
      // use a bitcast/GEP of the alloca we are removing.
      PassPrediction::PassPeeper(__FILE__, 353); // for
      if (!Users[i]) {
        PassPrediction::PassPeeper(__FILE__, 354); // if
        continue;
      }

      Instruction *I = cast<Instruction>(&*Users[i]);

      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
        PassPrediction::PassPeeper(__FILE__, 355); // if
        if (II->getIntrinsicID() == Intrinsic::objectsize) {
          PassPrediction::PassPeeper(__FILE__, 356); // if
          ConstantInt *Result = lowerObjectSizeCall(II, DL, &TLI,
                                                    /*MustSucceed=*/true);
          replaceInstUsesWith(*I, Result);
          eraseInstFromFunction(*I);
          Users[i] = nullptr; // Skip examining in the next loop.
        }
      }
    }
    for (unsigned i = 0, e = Users.size(); i != e; ++i) {
      PassPrediction::PassPeeper(__FILE__, 357); // for
      if (!Users[i]) {
        PassPrediction::PassPeeper(__FILE__, 358); // if
        continue;
      }

      Instruction *I = cast<Instruction>(&*Users[i]);

      if (ICmpInst *C = dyn_cast<ICmpInst>(I)) {
        PassPrediction::PassPeeper(__FILE__, 359); // if
        replaceInstUsesWith(*C,
                            ConstantInt::get(Type::getInt1Ty(C->getContext()),
                                             C->isFalseWhenEqual()));
      } else if (isa<BitCastInst>(I) || isa<GetElementPtrInst>(I) ||
                 isa<AddrSpaceCastInst>(I)) {
        PassPrediction::PassPeeper(__FILE__, 360); // if
        replaceInstUsesWith(*I, UndefValue::get(I->getType()));
      }
      eraseInstFromFunction(*I);
    }

    if (InvokeInst *II = dyn_cast<InvokeInst>(&MI)) {
      // Replace invoke with a NOP intrinsic to maintain the original CFG
      PassPrediction::PassPeeper(__FILE__, 361); // if
      Module *M = II->getModule();
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::donothing);
      InvokeInst::Create(F, II->getNormalDest(), II->getUnwindDest(), None, "",
                         II->getParent());
    }
    return eraseInstFromFunction(MI);
  }
  return nullptr;
}

/// \brief Move the call to free before a NULL test.
///
/// Check if this free is accessed after its argument has been test
/// against NULL (property 0).
/// If yes, it is legal to move this call in its predecessor block.
///
/// The move is performed only if the block containing the call to free
/// will be removed, i.e.:
/// 1. it has only one predecessor P, and P has two successors
/// 2. it contains the call and an unconditional branch
/// 3. its successor is the same as its predecessor's successor
///
/// The profitability is out-of concern here and this function should
/// be called only if the caller knows this transformation would be
/// profitable (e.g., for code size).
static Instruction *tryToMoveFreeBeforeNullTest(CallInst &FI) {
  Value *Op = FI.getArgOperand(0);
  BasicBlock *FreeInstrBB = FI.getParent();
  BasicBlock *PredBB = FreeInstrBB->getSinglePredecessor();

  // Validate part of constraint #1: Only one predecessor
  // FIXME: We can extend the number of predecessor, but in that case, we
  //        would duplicate the call to free in each predecessor and it may
  //        not be profitable even for code size.
  if (!PredBB) {
    PassPrediction::PassPeeper(__FILE__, 362); // if
    return nullptr;
  }

  // Validate constraint #2: Does this block contains only the call to
  //                         free and an unconditional branch?
  // FIXME: We could check if we can speculate everything in the
  //        predecessor block
  if (FreeInstrBB->size() != 2) {
    PassPrediction::PassPeeper(__FILE__, 363); // if
    return nullptr;
  }
  BasicBlock *SuccBB;
  if (!match(FreeInstrBB->getTerminator(), m_UnconditionalBr(SuccBB))) {
    PassPrediction::PassPeeper(__FILE__, 364); // if
    return nullptr;
  }

  // Validate the rest of constraint #1 by matching on the pred branch.
  TerminatorInst *TI = PredBB->getTerminator();
  BasicBlock *TrueBB, *FalseBB;
  ICmpInst::Predicate Pred;
  if (!match(TI,
             m_Br(m_ICmp(Pred, m_Specific(Op), m_Zero()), TrueBB, FalseBB))) {
    PassPrediction::PassPeeper(__FILE__, 365); // if
    return nullptr;
  }
  if (Pred != ICmpInst::ICMP_EQ && Pred != ICmpInst::ICMP_NE) {
    PassPrediction::PassPeeper(__FILE__, 366); // if
    return nullptr;
  }

  // Validate constraint #3: Ensure the null case just falls through.
  if (SuccBB != (Pred == ICmpInst::ICMP_EQ ? TrueBB : FalseBB)) {
    PassPrediction::PassPeeper(__FILE__, 367); // if
    return nullptr;
  }
  assert(FreeInstrBB == (Pred == ICmpInst::ICMP_EQ ? FalseBB : TrueBB) &&
         "Broken CFG: missing edge from predecessor to successor");

  FI.moveBefore(TI);
  return &FI;
}

Instruction *InstCombiner::visitFree(CallInst &FI) {
  Value *Op = FI.getArgOperand(0);

  // free undef -> unreachable.
  if (isa<UndefValue>(Op)) {
    // Insert a new store to null because we cannot modify the CFG here.
    PassPrediction::PassPeeper(__FILE__, 368); // if
    Builder.CreateStore(ConstantInt::getTrue(FI.getContext()),
                        UndefValue::get(Type::getInt1PtrTy(FI.getContext())));
    return eraseInstFromFunction(FI);
  }

  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op)) {
    PassPrediction::PassPeeper(__FILE__, 369); // if
    return eraseInstFromFunction(FI);
  }

  // If we optimize for code size, try to move the call to free before the null
  // test so that simplify cfg can remove the empty block and dead code
  // elimination the branch. I.e., helps to turn something like:
  // if (foo) free(foo);
  // into
  // free(foo);
  if (MinimizeSize) {
    PassPrediction::PassPeeper(__FILE__, 370); // if
    if (Instruction *I = tryToMoveFreeBeforeNullTest(FI)) {
      PassPrediction::PassPeeper(__FILE__, 371); // if
      return I;
    }
  }

  return nullptr;
}

Instruction *InstCombiner::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() == 0) {              // ret void
    PassPrediction::PassPeeper(__FILE__, 372); // if
    return nullptr;
  }

  Value *ResultOp = RI.getOperand(0);
  Type *VTy = ResultOp->getType();
  if (!VTy->isIntegerTy()) {
    PassPrediction::PassPeeper(__FILE__, 373); // if
    return nullptr;
  }

  // There might be assume intrinsics dominating this return that completely
  // determine the value. If so, constant fold it.
  KnownBits Known = computeKnownBits(ResultOp, 0, &RI);
  if (Known.isConstant()) {
    PassPrediction::PassPeeper(__FILE__, 374); // if
    RI.setOperand(0, Constant::getIntegerValue(VTy, Known.getConstant()));
  }

  return nullptr;
}

Instruction *InstCombiner::visitBranchInst(BranchInst &BI) {
  // Change br (not X), label True, label False to: br X, label False, True
  Value *X = nullptr;
  BasicBlock *TrueDest;
  BasicBlock *FalseDest;
  if (match(&BI, m_Br(m_Not(m_Value(X)), TrueDest, FalseDest)) &&
      !isa<Constant>(X)) {
    // Swap Destinations and condition...
    PassPrediction::PassPeeper(__FILE__, 375); // if
    BI.setCondition(X);
    BI.swapSuccessors();
    return &BI;
  }

  // If the condition is irrelevant, remove the use so that other
  // transforms on the condition become more effective.
  if (BI.isConditional() && BI.getSuccessor(0) == BI.getSuccessor(1) &&
      !isa<UndefValue>(BI.getCondition())) {
    PassPrediction::PassPeeper(__FILE__, 376); // if
    BI.setCondition(UndefValue::get(BI.getCondition()->getType()));
    return &BI;
  }

  // Canonicalize, for example, icmp_ne -> icmp_eq or fcmp_one -> fcmp_oeq.
  CmpInst::Predicate Pred;
  if (match(&BI, m_Br(m_OneUse(m_Cmp(Pred, m_Value(), m_Value())), TrueDest,
                      FalseDest)) &&
      !isCanonicalPredicate(Pred)) {
    // Swap destinations and condition.
    PassPrediction::PassPeeper(__FILE__, 377); // if
    CmpInst *Cond = cast<CmpInst>(BI.getCondition());
    Cond->setPredicate(CmpInst::getInversePredicate(Pred));
    BI.swapSuccessors();
    Worklist.Add(Cond);
    return &BI;
  }

  return nullptr;
}

Instruction *InstCombiner::visitSwitchInst(SwitchInst &SI) {
  Value *Cond = SI.getCondition();
  Value *Op0;
  ConstantInt *AddRHS;
  if (match(Cond, m_Add(m_Value(Op0), m_ConstantInt(AddRHS)))) {
    // Change 'switch (X+4) case 1:' into 'switch (X) case -3'.
    PassPrediction::PassPeeper(__FILE__, 378); // if
    for (auto Case : SI.cases()) {
      PassPrediction::PassPeeper(__FILE__, 379); // for-range
      Constant *NewCase = ConstantExpr::getSub(Case.getCaseValue(), AddRHS);
      assert(isa<ConstantInt>(NewCase) &&
             "Result of expression should be constant");
      Case.setValue(cast<ConstantInt>(NewCase));
    }
    SI.setCondition(Op0);
    return &SI;
  }

  KnownBits Known = computeKnownBits(Cond, 0, &SI);
  unsigned LeadingKnownZeros = Known.countMinLeadingZeros();
  unsigned LeadingKnownOnes = Known.countMinLeadingOnes();

  // Compute the number of leading bits we can ignore.
  // TODO: A better way to determine this would use ComputeNumSignBits().
  for (auto &C : SI.cases()) {
    PassPrediction::PassPeeper(__FILE__, 380); // for-range
    LeadingKnownZeros = std::min(
        LeadingKnownZeros, C.getCaseValue()->getValue().countLeadingZeros());
    LeadingKnownOnes = std::min(
        LeadingKnownOnes, C.getCaseValue()->getValue().countLeadingOnes());
  }

  unsigned NewWidth =
      Known.getBitWidth() - std::max(LeadingKnownZeros, LeadingKnownOnes);

  // Shrink the condition operand if the new type is smaller than the old type.
  // This may produce a non-standard type for the switch, but that's ok because
  // the backend should extend back to a legal type for the target.
  if (NewWidth > 0 && NewWidth < Known.getBitWidth()) {
    PassPrediction::PassPeeper(__FILE__, 381); // if
    IntegerType *Ty = IntegerType::get(SI.getContext(), NewWidth);
    Builder.SetInsertPoint(&SI);
    Value *NewCond = Builder.CreateTrunc(Cond, Ty, "trunc");
    SI.setCondition(NewCond);

    for (auto Case : SI.cases()) {
      PassPrediction::PassPeeper(__FILE__, 382); // for-range
      APInt TruncatedCase = Case.getCaseValue()->getValue().trunc(NewWidth);
      Case.setValue(ConstantInt::get(SI.getContext(), TruncatedCase));
    }
    return &SI;
  }

  return nullptr;
}

Instruction *InstCombiner::visitExtractValueInst(ExtractValueInst &EV) {
  Value *Agg = EV.getAggregateOperand();

  if (!EV.hasIndices()) {
    PassPrediction::PassPeeper(__FILE__, 383); // if
    return replaceInstUsesWith(EV, Agg);
  }

  if (Value *V = SimplifyExtractValueInst(Agg, EV.getIndices(),
                                          SQ.getWithInstruction(&EV))) {
    PassPrediction::PassPeeper(__FILE__, 384); // if
    return replaceInstUsesWith(EV, V);
  }

  if (InsertValueInst *IV = dyn_cast<InsertValueInst>(Agg)) {
    // We're extracting from an insertvalue instruction, compare the indices
    PassPrediction::PassPeeper(__FILE__, 385); // if
    const unsigned *exti, *exte, *insi, *inse;
    for (exti = EV.idx_begin(), insi = IV->idx_begin(), exte = EV.idx_end(),
        inse = IV->idx_end();
         exti != exte && insi != inse; ++exti, ++insi) {
      PassPrediction::PassPeeper(__FILE__, 386); // for
      if (*insi != *exti) {
        // The insert and extract both reference distinctly different elements.
        // This means the extract is not influenced by the insert, and we can
        // replace the aggregate operand of the extract with the aggregate
        // operand of the insert. i.e., replace
        // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
        // %E = extractvalue { i32, { i32 } } %I, 0
        // with
        // %E = extractvalue { i32, { i32 } } %A, 0
        PassPrediction::PassPeeper(__FILE__, 387); // if
        return ExtractValueInst::Create(IV->getAggregateOperand(),
                                        EV.getIndices());
      }
    }
    if (exti == exte && insi == inse) {
      // Both iterators are at the end: Index lists are identical. Replace
      // %B = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %C = extractvalue { i32, { i32 } } %B, 1, 0
      // with "i32 42"
      PassPrediction::PassPeeper(__FILE__, 388); // if
      return replaceInstUsesWith(EV, IV->getInsertedValueOperand());
    }
    if (exti == exte) {
      // The extract list is a prefix of the insert list. i.e. replace
      // %I = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %E = extractvalue { i32, { i32 } } %I, 1
      // with
      // %X = extractvalue { i32, { i32 } } %A, 1
      // %E = insertvalue { i32 } %X, i32 42, 0
      // by switching the order of the insert and extract (though the
      // insertvalue should be left in, since it may have other uses).
      PassPrediction::PassPeeper(__FILE__, 389); // if
      Value *NewEV = Builder.CreateExtractValue(IV->getAggregateOperand(),
                                                EV.getIndices());
      return InsertValueInst::Create(NewEV, IV->getInsertedValueOperand(),
                                     makeArrayRef(insi, inse));
    }
    if (insi == inse) {
      // The insert list is a prefix of the extract list
      // We can simply remove the common indices from the extract and make it
      // operate on the inserted value instead of the insertvalue result.
      // i.e., replace
      // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
      // %E = extractvalue { i32, { i32 } } %I, 1, 0
      // with
      // %E extractvalue { i32 } { i32 42 }, 0
      PassPrediction::PassPeeper(__FILE__, 390); // if
      return ExtractValueInst::Create(IV->getInsertedValueOperand(),
                                      makeArrayRef(exti, exte));
    }
  }
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Agg)) {
    // We're extracting from an intrinsic, see if we're the only user, which
    // allows us to simplify multiple result intrinsics to simpler things that
    // just get one value.
    PassPrediction::PassPeeper(__FILE__, 391); // if
    if (II->hasOneUse()) {
      // Check if we're grabbing the overflow bit or the result of a 'with
      // overflow' intrinsic.  If it's the latter we can remove the intrinsic
      // and replace it with a traditional binary instruction.
      PassPrediction::PassPeeper(__FILE__, 392); // if
      switch (II->getIntrinsicID()) {
      case Intrinsic::uadd_with_overflow:
        PassPrediction::PassPeeper(__FILE__, 393); // case

      case Intrinsic::sadd_with_overflow:
        PassPrediction::PassPeeper(__FILE__, 394); // case

        if (*EV.idx_begin() == 0) {                  // Normal result.
          PassPrediction::PassPeeper(__FILE__, 395); // if
          Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
          replaceInstUsesWith(*II, UndefValue::get(II->getType()));
          eraseInstFromFunction(*II);
          return BinaryOperator::CreateAdd(LHS, RHS);
        }

        // If the normal result of the add is dead, and the RHS is a constant,
        // we can transform this into a range comparison.
        // overflow = uadd a, -4  -->  overflow = icmp ugt a, 3
        if (II->getIntrinsicID() == Intrinsic::uadd_with_overflow) {
          PassPrediction::PassPeeper(__FILE__, 396); // if
          if (ConstantInt *CI = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
            PassPrediction::PassPeeper(__FILE__, 397); // if
            return new ICmpInst(ICmpInst::ICMP_UGT, II->getArgOperand(0),
                                ConstantExpr::getNot(CI));
          }
        }
        PassPrediction::PassPeeper(__FILE__, 398); // break
        break;
      case Intrinsic::usub_with_overflow:
        PassPrediction::PassPeeper(__FILE__, 399); // case

      case Intrinsic::ssub_with_overflow:
        PassPrediction::PassPeeper(__FILE__, 400); // case

        if (*EV.idx_begin() == 0) {                  // Normal result.
          PassPrediction::PassPeeper(__FILE__, 401); // if
          Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
          replaceInstUsesWith(*II, UndefValue::get(II->getType()));
          eraseInstFromFunction(*II);
          return BinaryOperator::CreateSub(LHS, RHS);
        }
        PassPrediction::PassPeeper(__FILE__, 402); // break
        break;
      case Intrinsic::umul_with_overflow:
        PassPrediction::PassPeeper(__FILE__, 403); // case

      case Intrinsic::smul_with_overflow:
        PassPrediction::PassPeeper(__FILE__, 404); // case

        if (*EV.idx_begin() == 0) {                  // Normal result.
          PassPrediction::PassPeeper(__FILE__, 405); // if
          Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
          replaceInstUsesWith(*II, UndefValue::get(II->getType()));
          eraseInstFromFunction(*II);
          return BinaryOperator::CreateMul(LHS, RHS);
        }
        PassPrediction::PassPeeper(__FILE__, 406); // break
        break;
      default:
        PassPrediction::PassPeeper(__FILE__, 407); // break
        break;
      }
    }
  }
  if (LoadInst *L = dyn_cast<LoadInst>(Agg)) {
    // If the (non-volatile) load only has one use, we can rewrite this to a
    // load from a GEP. This reduces the size of the load. If a load is used
    // only by extractvalue instructions then this either must have been
    // optimized before, or it is a struct with padding, in which case we
    // don't want to do the transformation as it loses padding knowledge.
    PassPrediction::PassPeeper(__FILE__, 408); // if
    if (L->isSimple() && L->hasOneUse()) {
      // extractvalue has integer indices, getelementptr has Value*s. Convert.
      PassPrediction::PassPeeper(__FILE__, 409); // if
      SmallVector<Value *, 4> Indices;
      // Prefix an i32 0 since we need the first element.
      Indices.push_back(Builder.getInt32(0));
      for (ExtractValueInst::idx_iterator I = EV.idx_begin(), E = EV.idx_end();
           I != E; ++I) {
        PassPrediction::PassPeeper(__FILE__, 410); // for
        Indices.push_back(Builder.getInt32(*I));
      }

      // We need to insert these at the location of the old load, not at that of
      // the extractvalue.
      Builder.SetInsertPoint(L);
      Value *GEP = Builder.CreateInBoundsGEP(L->getType(),
                                             L->getPointerOperand(), Indices);
      Instruction *NL = Builder.CreateLoad(GEP);
      // Whatever aliasing information we had for the orignal load must also
      // hold for the smaller load, so propagate the annotations.
      AAMDNodes Nodes;
      L->getAAMetadata(Nodes);
      NL->setAAMetadata(Nodes);
      // Returning the load directly will cause the main loop to insert it in
      // the wrong spot, so use replaceInstUsesWith().
      return replaceInstUsesWith(EV, NL);
    }
  }
  // We could simplify extracts from other values. Note that nested extracts may
  // already be simplified implicitly by the above: extract (extract (insert) )
  // will be translated into extract ( insert ( extract ) ) first and then just
  // the value inserted, if appropriate. Similarly for extracts from single-use
  // loads: extract (extract (load)) will be translated to extract (load (gep))
  // and if again single-use then via load (gep (gep)) to load (gep).
  // However, double extracts from e.g. function arguments or return values
  // aren't handled yet.
  return nullptr;
}

/// Return 'true' if the given typeinfo will match anything.
static bool isCatchAll(EHPersonality Personality, Constant *TypeInfo) {
  switch (Personality) {
  case EHPersonality::GNU_C:
    PassPrediction::PassPeeper(__FILE__, 411); // case

  case EHPersonality::GNU_C_SjLj:
    PassPrediction::PassPeeper(__FILE__, 412); // case

  case EHPersonality::Rust:
    PassPrediction::PassPeeper(__FILE__, 413); // case

    // The GCC C EH and Rust personality only exists to support cleanups, so
    // it's not clear what the semantics of catch clauses are.
    return false;
  case EHPersonality::Unknown:
    PassPrediction::PassPeeper(__FILE__, 414); // case

    return false;
  case EHPersonality::GNU_Ada:
    PassPrediction::PassPeeper(__FILE__, 415); // case

    // While __gnat_all_others_value will match any Ada exception, it doesn't
    // match foreign exceptions (or didn't, before gcc-4.7).
    return false;
  case EHPersonality::GNU_CXX:
    PassPrediction::PassPeeper(__FILE__, 416); // case

  case EHPersonality::GNU_CXX_SjLj:
    PassPrediction::PassPeeper(__FILE__, 417); // case

  case EHPersonality::GNU_ObjC:
    PassPrediction::PassPeeper(__FILE__, 418); // case

  case EHPersonality::MSVC_X86SEH:
    PassPrediction::PassPeeper(__FILE__, 419); // case

  case EHPersonality::MSVC_Win64SEH:
    PassPrediction::PassPeeper(__FILE__, 420); // case

  case EHPersonality::MSVC_CXX:
    PassPrediction::PassPeeper(__FILE__, 421); // case

  case EHPersonality::CoreCLR:
    PassPrediction::PassPeeper(__FILE__, 422); // case

    return TypeInfo->isNullValue();
  }
  llvm_unreachable("invalid enum");
}

static bool shorter_filter(const Value *LHS, const Value *RHS) {
  return cast<ArrayType>(LHS->getType())->getNumElements() <
         cast<ArrayType>(RHS->getType())->getNumElements();
}

Instruction *InstCombiner::visitLandingPadInst(LandingPadInst &LI) {
  // The logic here should be correct for any real-world personality function.
  // However if that turns out not to be true, the offending logic can always
  // be conditioned on the personality function, like the catch-all logic is.
  EHPersonality Personality =
      classifyEHPersonality(LI.getParent()->getParent()->getPersonalityFn());

  // Simplify the list of clauses, eg by removing repeated catch clauses
  // (these are often created by inlining).
  bool MakeNewInstruction = false; // If true, recreate using the following:
  SmallVector<Constant *, 16> NewClauses; // - Clauses for the new instruction;
  bool CleanupFlag = LI.isCleanup();      // - The new instruction is a cleanup.

  SmallPtrSet<Value *, 16> AlreadyCaught; // Typeinfos known caught already.
  for (unsigned i = 0, e = LI.getNumClauses(); i != e; ++i) {
    PassPrediction::PassPeeper(__FILE__, 423); // for
    bool isLastClause = i + 1 == e;
    if (LI.isCatch(i)) {
      // A catch clause.
      PassPrediction::PassPeeper(__FILE__, 424); // if
      Constant *CatchClause = LI.getClause(i);
      Constant *TypeInfo = CatchClause->stripPointerCasts();

      // If we already saw this clause, there is no point in having a second
      // copy of it.
      if (AlreadyCaught.insert(TypeInfo).second) {
        // This catch clause was not already seen.
        PassPrediction::PassPeeper(__FILE__, 425); // if
        NewClauses.push_back(CatchClause);
      } else {
        // Repeated catch clause - drop the redundant copy.
        PassPrediction::PassPeeper(__FILE__, 426); // else
        MakeNewInstruction = true;
      }

      // If this is a catch-all then there is no point in keeping any following
      // clauses or marking the landingpad as having a cleanup.
      if (isCatchAll(Personality, TypeInfo)) {
        PassPrediction::PassPeeper(__FILE__, 427); // if
        if (!isLastClause) {
          PassPrediction::PassPeeper(__FILE__, 428); // if
          MakeNewInstruction = true;
        }
        CleanupFlag = false;
        PassPrediction::PassPeeper(__FILE__, 429); // break
        break;
      }
    } else {
      // A filter clause.  If any of the filter elements were already caught
      // then they can be dropped from the filter.  It is tempting to try to
      // exploit the filter further by saying that any typeinfo that does not
      // occur in the filter can't be caught later (and thus can be dropped).
      // However this would be wrong, since typeinfos can match without being
      // equal (for example if one represents a C++ class, and the other some
      // class derived from it).
      assert(LI.isFilter(i) && "Unsupported landingpad clause!");
      Constant *FilterClause = LI.getClause(i);
      ArrayType *FilterType = cast<ArrayType>(FilterClause->getType());
      unsigned NumTypeInfos = FilterType->getNumElements();

      // An empty filter catches everything, so there is no point in keeping any
      // following clauses or marking the landingpad as having a cleanup.  By
      // dealing with this case here the following code is made a bit simpler.
      if (!NumTypeInfos) {
        PassPrediction::PassPeeper(__FILE__, 430); // if
        NewClauses.push_back(FilterClause);
        if (!isLastClause) {
          PassPrediction::PassPeeper(__FILE__, 431); // if
          MakeNewInstruction = true;
        }
        CleanupFlag = false;
        PassPrediction::PassPeeper(__FILE__, 432); // break
        break;
      }

      bool MakeNewFilter = false;                // If true, make a new filter.
      SmallVector<Constant *, 16> NewFilterElts; // New elements.
      if (isa<ConstantAggregateZero>(FilterClause)) {
        // Not an empty filter - it contains at least one null typeinfo.
        assert(NumTypeInfos > 0 && "Should have handled empty filter already!");
        Constant *TypeInfo =
            Constant::getNullValue(FilterType->getElementType());
        // If this typeinfo is a catch-all then the filter can never match.
        if (isCatchAll(Personality, TypeInfo)) {
          // Throw the filter away.
          PassPrediction::PassPeeper(__FILE__, 434); // if
          MakeNewInstruction = true;
          continue;
        }

        // There is no point in having multiple copies of this typeinfo, so
        // discard all but the first copy if there is more than one.
        NewFilterElts.push_back(TypeInfo);
        if (NumTypeInfos > 1) {
          PassPrediction::PassPeeper(__FILE__, 435); // if
          MakeNewFilter = true;
        }
      } else {
        PassPrediction::PassPeeper(__FILE__, 433); // else
        ConstantArray *Filter = cast<ConstantArray>(FilterClause);
        SmallPtrSet<Value *, 16> SeenInFilter; // For uniquing the elements.
        NewFilterElts.reserve(NumTypeInfos);

        // Remove any filter elements that were already caught or that already
        // occurred in the filter.  While there, see if any of the elements are
        // catch-alls.  If so, the filter can be discarded.
        bool SawCatchAll = false;
        for (unsigned j = 0; j != NumTypeInfos; ++j) {
          PassPrediction::PassPeeper(__FILE__, 436); // for
          Constant *Elt = Filter->getOperand(j);
          Constant *TypeInfo = Elt->stripPointerCasts();
          if (isCatchAll(Personality, TypeInfo)) {
            // This element is a catch-all.  Bail out, noting this fact.
            PassPrediction::PassPeeper(__FILE__, 437); // if
            SawCatchAll = true;
            PassPrediction::PassPeeper(__FILE__, 438); // break
            break;
          }

          // Even if we've seen a type in a catch clause, we don't want to
          // remove it from the filter.  An unexpected type handler may be
          // set up for a call site which throws an exception of the same
          // type caught.  In order for the exception thrown by the unexpected
          // handler to propagate correctly, the filter must be correctly
          // described for the call site.
          //
          // Example:
          //
          // void unexpected() { throw 1;}
          // void foo() throw (int) {
          //   std::set_unexpected(unexpected);
          //   try {
          //     throw 2.0;
          //   } catch (int i) {}
          // }

          // There is no point in having multiple copies of the same typeinfo in
          // a filter, so only add it if we didn't already.
          if (SeenInFilter.insert(TypeInfo).second) {
            PassPrediction::PassPeeper(__FILE__, 439); // if
            NewFilterElts.push_back(cast<Constant>(Elt));
          }
        }
        // A filter containing a catch-all cannot match anything by definition.
        if (SawCatchAll) {
          // Throw the filter away.
          PassPrediction::PassPeeper(__FILE__, 440); // if
          MakeNewInstruction = true;
          continue;
        }

        // If we dropped something from the filter, make a new one.
        if (NewFilterElts.size() < NumTypeInfos) {
          PassPrediction::PassPeeper(__FILE__, 441); // if
          MakeNewFilter = true;
        }
      }
      if (MakeNewFilter) {
        PassPrediction::PassPeeper(__FILE__, 442); // if
        FilterType =
            ArrayType::get(FilterType->getElementType(), NewFilterElts.size());
        FilterClause = ConstantArray::get(FilterType, NewFilterElts);
        MakeNewInstruction = true;
      }

      NewClauses.push_back(FilterClause);

      // If the new filter is empty then it will catch everything so there is
      // no point in keeping any following clauses or marking the landingpad
      // as having a cleanup.  The case of the original filter being empty was
      // already handled above.
      if (MakeNewFilter && !NewFilterElts.size()) {
        assert(MakeNewInstruction && "New filter but not a new instruction!");
        CleanupFlag = false;
        PassPrediction::PassPeeper(__FILE__, 443); // break
        break;
      }
    }
  }

  // If several filters occur in a row then reorder them so that the shortest
  // filters come first (those with the smallest number of elements).  This is
  // advantageous because shorter filters are more likely to match, speeding up
  // unwinding, but mostly because it increases the effectiveness of the other
  // filter optimizations below.
  for (unsigned i = 0, e = NewClauses.size(); i + 1 < e;) {
    PassPrediction::PassPeeper(__FILE__, 444); // for
    unsigned j;
    // Find the maximal 'j' s.t. the range [i, j) consists entirely of filters.
    for (j = i; j != e; ++j) {
      PassPrediction::PassPeeper(__FILE__, 445); // for
      if (!isa<ArrayType>(NewClauses[j]->getType())) {
        PassPrediction::PassPeeper(__FILE__, 446); // if
        break;
      }
    }

    // Check whether the filters are already sorted by length.  We need to know
    // if sorting them is actually going to do anything so that we only make a
    // new landingpad instruction if it does.
    for (unsigned k = i; k + 1 < j; ++k) {
      PassPrediction::PassPeeper(__FILE__, 447); // for
      if (shorter_filter(NewClauses[k + 1], NewClauses[k])) {
        // Not sorted, so sort the filters now.  Doing an unstable sort would be
        // correct too but reordering filters pointlessly might confuse users.
        PassPrediction::PassPeeper(__FILE__, 448); // if
        std::stable_sort(NewClauses.begin() + i, NewClauses.begin() + j,
                         shorter_filter);
        MakeNewInstruction = true;
        PassPrediction::PassPeeper(__FILE__, 449); // break
        break;
      }
    }

    // Look for the next batch of filters.
    i = j + 1;
  }

  // If typeinfos matched if and only if equal, then the elements of a filter L
  // that occurs later than a filter F could be replaced by the intersection of
  // the elements of F and L.  In reality two typeinfos can match without being
  // equal (for example if one represents a C++ class, and the other some class
  // derived from it) so it would be wrong to perform this transform in general.
  // However the transform is correct and useful if F is a subset of L.  In that
  // case L can be replaced by F, and thus removed altogether since repeating a
  // filter is pointless.  So here we look at all pairs of filters F and L where
  // L follows F in the list of clauses, and remove L if every element of F is
  // an element of L.  This can occur when inlining C++ functions with exception
  // specifications.
  for (unsigned i = 0; i + 1 < NewClauses.size(); ++i) {
    // Examine each filter in turn.
    PassPrediction::PassPeeper(__FILE__, 450); // for
    Value *Filter = NewClauses[i];
    ArrayType *FTy = dyn_cast<ArrayType>(Filter->getType());
    if (!FTy) {
      // Not a filter - skip it.
      PassPrediction::PassPeeper(__FILE__, 451); // if
      continue;
    }
    unsigned FElts = FTy->getNumElements();
    // Examine each filter following this one.  Doing this backwards means that
    // we don't have to worry about filters disappearing under us when removed.
    for (unsigned j = NewClauses.size() - 1; j != i; --j) {
      PassPrediction::PassPeeper(__FILE__, 452); // for
      Value *LFilter = NewClauses[j];
      ArrayType *LTy = dyn_cast<ArrayType>(LFilter->getType());
      if (!LTy) {
        // Not a filter - skip it.
        PassPrediction::PassPeeper(__FILE__, 453); // if
        continue;
      }
      // If Filter is a subset of LFilter, i.e. every element of Filter is also
      // an element of LFilter, then discard LFilter.
      SmallVectorImpl<Constant *>::iterator J = NewClauses.begin() + j;
      // If Filter is empty then it is a subset of LFilter.
      if (!FElts) {
        // Discard LFilter.
        PassPrediction::PassPeeper(__FILE__, 454); // if
        NewClauses.erase(J);
        MakeNewInstruction = true;
        // Move on to the next filter.
        continue;
      }
      unsigned LElts = LTy->getNumElements();
      // If Filter is longer than LFilter then it cannot be a subset of it.
      if (FElts > LElts) {
        // Move on to the next filter.
        PassPrediction::PassPeeper(__FILE__, 455); // if
        continue;
      }
      // At this point we know that LFilter has at least one element.
      if (isa<ConstantAggregateZero>(LFilter)) { // LFilter only contains zeros.
        // Filter is a subset of LFilter iff Filter contains only zeros (as we
        // already know that Filter is not longer than LFilter).
        PassPrediction::PassPeeper(__FILE__, 456); // if
        if (isa<ConstantAggregateZero>(Filter)) {
          assert(FElts <= LElts && "Should have handled this case earlier!");
          // Discard LFilter.
          NewClauses.erase(J);
          MakeNewInstruction = true;
        }
        // Move on to the next filter.
        continue;
      }
      ConstantArray *LArray = cast<ConstantArray>(LFilter);
      if (isa<ConstantAggregateZero>(Filter)) { // Filter only contains zeros.
        // Since Filter is non-empty and contains only zeros, it is a subset of
        // LFilter iff LFilter contains a zero.
        assert(FElts > 0 && "Should have eliminated the empty filter earlier!");
        for (unsigned l = 0; l != LElts; ++l) {
          PassPrediction::PassPeeper(__FILE__, 457); // for
          if (LArray->getOperand(l)->isNullValue()) {
            // LFilter contains a zero - discard it.
            PassPrediction::PassPeeper(__FILE__, 458); // if
            NewClauses.erase(J);
            MakeNewInstruction = true;
            PassPrediction::PassPeeper(__FILE__, 459); // break
            break;
          }
        }
        // Move on to the next filter.
        continue;
      }
      // At this point we know that both filters are ConstantArrays.  Loop over
      // operands to see whether every element of Filter is also an element of
      // LFilter.  Since filters tend to be short this is probably faster than
      // using a method that scales nicely.
      ConstantArray *FArray = cast<ConstantArray>(Filter);
      bool AllFound = true;
      for (unsigned f = 0; f != FElts; ++f) {
        PassPrediction::PassPeeper(__FILE__, 460); // for
        Value *FTypeInfo = FArray->getOperand(f)->stripPointerCasts();
        AllFound = false;
        for (unsigned l = 0; l != LElts; ++l) {
          PassPrediction::PassPeeper(__FILE__, 461); // for
          Value *LTypeInfo = LArray->getOperand(l)->stripPointerCasts();
          if (LTypeInfo == FTypeInfo) {
            PassPrediction::PassPeeper(__FILE__, 462); // if
            AllFound = true;
            PassPrediction::PassPeeper(__FILE__, 463); // break
            break;
          }
        }
        if (!AllFound) {
          PassPrediction::PassPeeper(__FILE__, 464); // if
          break;
        }
      }
      if (AllFound) {
        // Discard LFilter.
        PassPrediction::PassPeeper(__FILE__, 465); // if
        NewClauses.erase(J);
        MakeNewInstruction = true;
      }
      // Move on to the next filter.
    }
  }

  // If we changed any of the clauses, replace the old landingpad instruction
  // with a new one.
  if (MakeNewInstruction) {
    PassPrediction::PassPeeper(__FILE__, 466); // if
    LandingPadInst *NLI =
        LandingPadInst::Create(LI.getType(), NewClauses.size());
    for (unsigned i = 0, e = NewClauses.size(); i != e; ++i) {
      PassPrediction::PassPeeper(__FILE__, 467); // for
      NLI->addClause(NewClauses[i]);
    }
    // A landing pad with no clauses must have the cleanup flag set.  It is
    // theoretically possible, though highly unlikely, that we eliminated all
    // clauses.  If so, force the cleanup flag to true.
    if (NewClauses.empty()) {
      PassPrediction::PassPeeper(__FILE__, 468); // if
      CleanupFlag = true;
    }
    NLI->setCleanup(CleanupFlag);
    return NLI;
  }

  // Even if none of the clauses changed, we may nonetheless have understood
  // that the cleanup flag is pointless.  Clear it if so.
  if (LI.isCleanup() != CleanupFlag) {
    assert(!CleanupFlag && "Adding a cleanup, not removing one?!");
    LI.setCleanup(CleanupFlag);
    return &LI;
  }

  return nullptr;
}

/// Try to move the specified instruction from its current block into the
/// beginning of DestBlock, which can only happen if it's safe to move the
/// instruction past all of the instructions between it and the end of its
/// block.
static bool TryToSinkInstruction(Instruction *I, BasicBlock *DestBlock) {
  assert(I->hasOneUse() && "Invariants didn't hold!");

  // Cannot move control-flow-involving, volatile loads, vaarg, etc.
  if (isa<PHINode>(I) || I->isEHPad() || I->mayHaveSideEffects() ||
      isa<TerminatorInst>(I)) {
    PassPrediction::PassPeeper(__FILE__, 469); // if
    return false;
  }

  // Do not sink alloca instructions out of the entry block.
  if (isa<AllocaInst>(I) &&
      I->getParent() == &DestBlock->getParent()->getEntryBlock()) {
    PassPrediction::PassPeeper(__FILE__, 470); // if
    return false;
  }

  // Do not sink into catchswitch blocks.
  if (isa<CatchSwitchInst>(DestBlock->getTerminator())) {
    PassPrediction::PassPeeper(__FILE__, 471); // if
    return false;
  }

  // Do not sink convergent call instructions.
  if (auto *CI = dyn_cast<CallInst>(I)) {
    PassPrediction::PassPeeper(__FILE__, 472); // if
    if (CI->isConvergent()) {
      PassPrediction::PassPeeper(__FILE__, 473); // if
      return false;
    }
  }
  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (I->mayReadFromMemory()) {
    PassPrediction::PassPeeper(__FILE__, 474); // if
    for (BasicBlock::iterator Scan = I->getIterator(),
                              E = I->getParent()->end();
         Scan != E; ++Scan) {
      PassPrediction::PassPeeper(__FILE__, 475); // for
      if (Scan->mayWriteToMemory()) {
        PassPrediction::PassPeeper(__FILE__, 476); // if
        return false;
      }
    }
  }

  BasicBlock::iterator InsertPos = DestBlock->getFirstInsertionPt();
  I->moveBefore(&*InsertPos);
  ++NumSunkInst;
  return true;
}

bool InstCombiner::run() {
  while (!Worklist.isEmpty()) {
    PassPrediction::PassPeeper(__FILE__, 477); // while
    Instruction *I = Worklist.RemoveOne();
    if (I == nullptr) {
      PassPrediction::PassPeeper(__FILE__, 478); // if
      continue;                                  // skip null values.
    }

    // Check to see if we can DCE the instruction.
    if (isInstructionTriviallyDead(I, &TLI)) {
      DEBUG(dbgs() << "IC: DCE: " << *I << '\n');
      eraseInstFromFunction(*I);
      ++NumDeadInst;
      MadeIRChange = true;
      continue;
    }

    // Instruction isn't dead, see if we can constant propagate it.
    if (!I->use_empty() &&
        (I->getNumOperands() == 0 || isa<Constant>(I->getOperand(0)))) {
      PassPrediction::PassPeeper(__FILE__, 479); // if
      if (Constant *C = ConstantFoldInstruction(I, DL, &TLI)) {
        DEBUG(dbgs() << "IC: ConstFold to: " << *C << " from: " << *I << '\n');

        // Add operands to the worklist.
        replaceInstUsesWith(*I, C);
        ++NumConstProp;
        if (isInstructionTriviallyDead(I, &TLI)) {
          PassPrediction::PassPeeper(__FILE__, 480); // if
          eraseInstFromFunction(*I);
        }
        MadeIRChange = true;
        continue;
      }
    }

    // In general, it is possible for computeKnownBits to determine all bits in
    // a value even when the operands are not all constants.
    Type *Ty = I->getType();
    if (ExpensiveCombines && !I->use_empty() && Ty->isIntOrIntVectorTy()) {
      PassPrediction::PassPeeper(__FILE__, 481); // if
      KnownBits Known = computeKnownBits(I, /*Depth*/ 0, I);
      if (Known.isConstant()) {
        PassPrediction::PassPeeper(__FILE__, 482); // if
        Constant *C = ConstantInt::get(Ty, Known.getConstant());
        DEBUG(dbgs() << "IC: ConstFold (all bits known) to: " << *C
                     << " from: " << *I << '\n');

        // Add operands to the worklist.
        replaceInstUsesWith(*I, C);
        ++NumConstProp;
        if (isInstructionTriviallyDead(I, &TLI)) {
          PassPrediction::PassPeeper(__FILE__, 483); // if
          eraseInstFromFunction(*I);
        }
        MadeIRChange = true;
        continue;
      }
    }

    // See if we can trivially sink this instruction to a successor basic block.
    if (I->hasOneUse()) {
      PassPrediction::PassPeeper(__FILE__, 484); // if
      BasicBlock *BB = I->getParent();
      Instruction *UserInst = cast<Instruction>(*I->user_begin());
      BasicBlock *UserParent;

      // Get the block the use occurs in.
      if (PHINode *PN = dyn_cast<PHINode>(UserInst)) {
        PassPrediction::PassPeeper(__FILE__, 485); // if
        UserParent = PN->getIncomingBlock(*I->use_begin());
      } else {
        PassPrediction::PassPeeper(__FILE__, 486); // else
        UserParent = UserInst->getParent();
      }

      if (UserParent != BB) {
        PassPrediction::PassPeeper(__FILE__, 487); // if
        bool UserIsSuccessor = false;
        // See if the user is one of our successors.
        for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E;
             ++SI) {
          PassPrediction::PassPeeper(__FILE__, 488); // for
          if (*SI == UserParent) {
            PassPrediction::PassPeeper(__FILE__, 489); // if
            UserIsSuccessor = true;
            PassPrediction::PassPeeper(__FILE__, 490); // break
            break;
          }
        }

        // If the user is one of our immediate successors, and if that successor
        // only has us as a predecessors (we'd have to split the critical edge
        // otherwise), we can keep going.
        if (UserIsSuccessor && UserParent->getUniquePredecessor()) {
          // Okay, the CFG is simple enough, try to sink this instruction.
          PassPrediction::PassPeeper(__FILE__, 491); // if
          if (TryToSinkInstruction(I, UserParent)) {
            DEBUG(dbgs() << "IC: Sink: " << *I << '\n');
            MadeIRChange = true;
            // We'll add uses of the sunk instruction below, but since sinking
            // can expose opportunities for it's *operands* add them to the
            // worklist
            for (Use &U : I->operands()) {
              PassPrediction::PassPeeper(__FILE__, 492); // for-range
              if (Instruction *OpI = dyn_cast<Instruction>(U.get())) {
                PassPrediction::PassPeeper(__FILE__, 493); // if
                Worklist.Add(OpI);
              }
            }
          }
        }
      }
    }

    // Now that we have an instruction, try combining it to simplify it.
    Builder.SetInsertPoint(I);
    Builder.SetCurrentDebugLocation(I->getDebugLoc());

#ifndef NDEBUG
    std::string OrigI;
#endif
    DEBUG(raw_string_ostream SS(OrigI); I->print(SS); OrigI = SS.str(););
    DEBUG(dbgs() << "IC: Visiting: " << OrigI << '\n');

    if (Instruction *Result = visit(*I)) {
      PassPrediction::PassPeeper(__FILE__, 494); // if
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        DEBUG(dbgs() << "IC: Old = " << *I << '\n'
                     << "    New = " << *Result << '\n');

        if (I->getDebugLoc()) {
          PassPrediction::PassPeeper(__FILE__, 495); // if
          Result->setDebugLoc(I->getDebugLoc());
        }
        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Move the name to the new instruction first.
        Result->takeName(I);

        // Push the new instruction and any users onto the worklist.
        Worklist.AddUsersToWorkList(*Result);
        Worklist.Add(Result);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        BasicBlock::iterator InsertPos = I->getIterator();

        // If we replace a PHI with something that isn't a PHI, fix up the
        // insertion point.
        if (!isa<PHINode>(Result) && isa<PHINode>(InsertPos)) {
          PassPrediction::PassPeeper(__FILE__, 496); // if
          InsertPos = InstParent->getFirstInsertionPt();
        }

        InstParent->getInstList().insert(InsertPos, Result);

        eraseInstFromFunction(*I);
      } else {
        DEBUG(dbgs() << "IC: Mod = " << OrigI << '\n'
                     << "    New = " << *I << '\n');

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I, &TLI)) {
          PassPrediction::PassPeeper(__FILE__, 497); // if
          eraseInstFromFunction(*I);
        } else {
          PassPrediction::PassPeeper(__FILE__, 498); // else
          Worklist.AddUsersToWorkList(*I);
          Worklist.Add(I);
        }
      }
      MadeIRChange = true;
    }
  }

  Worklist.Zap();
  return MadeIRChange;
}

/// Walk the function in depth-first order, adding all reachable code to the
/// worklist.
///
/// This has a couple of tricks to make the code faster and more powerful.  In
/// particular, we constant fold and DCE instructions as we go, to avoid adding
/// them to the worklist (this significantly speeds up instcombine on code where
/// many instructions are dead or constant).  Additionally, if we find a branch
/// whose condition is a known constant, we only visit the reachable successors.
///
static bool AddReachableCodeToWorklist(BasicBlock *BB, const DataLayout &DL,
                                       SmallPtrSetImpl<BasicBlock *> &Visited,
                                       InstCombineWorklist &ICWorklist,
                                       const TargetLibraryInfo *TLI) {
  bool MadeIRChange = false;
  SmallVector<BasicBlock *, 256> Worklist;
  Worklist.push_back(BB);

  SmallVector<Instruction *, 128> InstrsForInstCombineWorklist;
  DenseMap<Constant *, Constant *> FoldedConstants;

  do {
    PassPrediction::PassPeeper(__FILE__, 499); // do-while
    BB = Worklist.pop_back_val();

    // We have now visited this block!  If we've already been here, ignore it.
    if (!Visited.insert(BB).second) {
      PassPrediction::PassPeeper(__FILE__, 500); // if
      continue;
    }

    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E;) {
      PassPrediction::PassPeeper(__FILE__, 501); // for
      Instruction *Inst = &*BBI++;

      // DCE instruction if trivially dead.
      if (isInstructionTriviallyDead(Inst, TLI)) {
        PassPrediction::PassPeeper(__FILE__, 502); // if
        ++NumDeadInst;
        DEBUG(dbgs() << "IC: DCE: " << *Inst << '\n');
        Inst->eraseFromParent();
        MadeIRChange = true;
        continue;
      }

      // ConstantProp instruction if trivially constant.
      if (!Inst->use_empty() &&
          (Inst->getNumOperands() == 0 || isa<Constant>(Inst->getOperand(0)))) {
        PassPrediction::PassPeeper(__FILE__, 503); // if
        if (Constant *C = ConstantFoldInstruction(Inst, DL, TLI)) {
          DEBUG(dbgs() << "IC: ConstFold to: " << *C << " from: " << *Inst
                       << '\n');
          Inst->replaceAllUsesWith(C);
          ++NumConstProp;
          if (isInstructionTriviallyDead(Inst, TLI)) {
            PassPrediction::PassPeeper(__FILE__, 504); // if
            Inst->eraseFromParent();
          }
          MadeIRChange = true;
          continue;
        }
      }

      // See if we can constant fold its operands.
      for (Use &U : Inst->operands()) {
        PassPrediction::PassPeeper(__FILE__, 505); // for-range
        if (!isa<ConstantVector>(U) && !isa<ConstantExpr>(U)) {
          PassPrediction::PassPeeper(__FILE__, 506); // if
          continue;
        }

        auto *C = cast<Constant>(U);
        Constant *&FoldRes = FoldedConstants[C];
        if (!FoldRes) {
          PassPrediction::PassPeeper(__FILE__, 507); // if
          FoldRes = ConstantFoldConstant(C, DL, TLI);
        }
        if (!FoldRes) {
          PassPrediction::PassPeeper(__FILE__, 508); // if
          FoldRes = C;
        }

        if (FoldRes != C) {
          DEBUG(dbgs() << "IC: ConstFold operand of: " << *Inst
                       << "\n    Old = " << *C << "\n    New = " << *FoldRes
                       << '\n');
          U = FoldRes;
          MadeIRChange = true;
        }
      }

      // Skip processing debug intrinsics in InstCombine. Processing these call
      // instructions consumes non-trivial amount of time and provides no value
      // for the optimization.
      if (!isa<DbgInfoIntrinsic>(Inst)) {
        PassPrediction::PassPeeper(__FILE__, 509); // if
        InstrsForInstCombineWorklist.push_back(Inst);
      }
    }

    // Recursively visit successors.  If this is a branch or switch on a
    // constant, only visit the reachable successor.
    TerminatorInst *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      PassPrediction::PassPeeper(__FILE__, 510); // if
      if (BI->isConditional() && isa<ConstantInt>(BI->getCondition())) {
        PassPrediction::PassPeeper(__FILE__, 511); // if
        bool CondVal = cast<ConstantInt>(BI->getCondition())->getZExtValue();
        BasicBlock *ReachableBB = BI->getSuccessor(!CondVal);
        Worklist.push_back(ReachableBB);
        continue;
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      PassPrediction::PassPeeper(__FILE__, 512); // if
      if (ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition())) {
        PassPrediction::PassPeeper(__FILE__, 513); // if
        Worklist.push_back(SI->findCaseValue(Cond)->getCaseSuccessor());
        continue;
      }
    }

    for (BasicBlock *SuccBB : TI->successors()) {
      PassPrediction::PassPeeper(__FILE__, 514); // for-range
      Worklist.push_back(SuccBB);
    }
  } while (!Worklist.empty());

  // Once we've found all of the instructions to add to instcombine's worklist,
  // add them in reverse order.  This way instcombine will visit from the top
  // of the function down.  This jives well with the way that it adds all uses
  // of instructions to the worklist after doing a transformation, thus avoiding
  // some N^2 behavior in pathological cases.
  ICWorklist.AddInitialGroup(InstrsForInstCombineWorklist);

  return MadeIRChange;
}

/// \brief Populate the IC worklist from a function, and prune any dead basic
/// blocks discovered in the process.
///
/// This also does basic constant propagation and other forward fixing to make
/// the combiner itself run much faster.
static bool prepareICWorklistFromFunction(Function &F, const DataLayout &DL,
                                          TargetLibraryInfo *TLI,
                                          InstCombineWorklist &ICWorklist) {
  bool MadeIRChange = false;

  // Do a depth-first traversal of the function, populate the worklist with
  // the reachable instructions.  Ignore blocks that are not reachable.  Keep
  // track of which blocks we visit.
  SmallPtrSet<BasicBlock *, 32> Visited;
  MadeIRChange |=
      AddReachableCodeToWorklist(&F.front(), DL, Visited, ICWorklist, TLI);

  // Do a quick scan over the function.  If we find any blocks that are
  // unreachable, remove any instructions inside of them.  This prevents
  // the instcombine code from having to deal with some bad special cases.
  for (BasicBlock &BB : F) {
    PassPrediction::PassPeeper(__FILE__, 515); // for-range
    if (Visited.count(&BB)) {
      PassPrediction::PassPeeper(__FILE__, 516); // if
      continue;
    }

    unsigned NumDeadInstInBB = removeAllNonTerminatorAndEHPadInstructions(&BB);
    MadeIRChange |= NumDeadInstInBB > 0;
    NumDeadInst += NumDeadInstInBB;
  }

  return MadeIRChange;
}

static bool combineInstructionsOverFunction(
    Function &F, InstCombineWorklist &Worklist, AliasAnalysis *AA,
    AssumptionCache &AC, TargetLibraryInfo &TLI, DominatorTree &DT,
    bool ExpensiveCombines = true, LoopInfo *LI = nullptr) {
  auto &DL = F.getParent()->getDataLayout();
  ExpensiveCombines |= EnableExpensiveCombines;

  /// Builder - This is an IRBuilder that automatically inserts new
  /// instructions into the worklist when they are created.
  IRBuilder<TargetFolder, IRBuilderCallbackInserter> Builder(
      F.getContext(), TargetFolder(DL),
      IRBuilderCallbackInserter([&Worklist, &AC](Instruction *I) {
        Worklist.Add(I);

        using namespace llvm::PatternMatch;
        if (match(I, m_Intrinsic<Intrinsic::assume>())) {
          PassPrediction::PassPeeper(__FILE__, 517); // if
          AC.registerAssumption(cast<CallInst>(I));
        }
      }));

  // Lower dbg.declare intrinsics otherwise their value may be clobbered
  // by instcombiner.
  bool MadeIRChange = LowerDbgDeclare(F);

  // Iterate while there is work to do.
  int Iteration = 0;
  for (;;) {
    PassPrediction::PassPeeper(__FILE__, 518); // for
    ++Iteration;
    DEBUG(dbgs() << "\n\nINSTCOMBINE ITERATION #" << Iteration << " on "
                 << F.getName() << "\n");

    MadeIRChange |= prepareICWorklistFromFunction(F, DL, &TLI, Worklist);

    InstCombiner IC(Worklist, Builder, F.optForMinSize(), ExpensiveCombines, AA,
                    AC, TLI, DT, DL, LI);
    IC.MaxArraySizeForCombine = MaxArraySize;

    if (!IC.run()) {
      PassPrediction::PassPeeper(__FILE__, 519); // if
      break;
    }
  }

  return MadeIRChange || Iteration > 1;
}

PreservedAnalyses InstCombinePass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);

  auto *LI = AM.getCachedResult<LoopAnalysis>(F);

  // FIXME: The AliasAnalysis is not yet supported in the new pass manager
  if (!combineInstructionsOverFunction(F, Worklist, nullptr, AC, TLI, DT,
                                       ExpensiveCombines, LI)) {
    // No changes, all analyses are preserved.
    PassPrediction::PassPeeper(__FILE__, 520); // if
    return PreservedAnalyses::all();
  }

  // Mark all the analyses that instcombine updates as preserved.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<AAManager>();
  PA.preserve<GlobalsAA>();
  return PA;
}

void InstructionCombiningPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
}

bool InstructionCombiningPass::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    PassPrediction::PassPeeper(__FILE__, 521); // if
    return false;
  }

  // Required analyses.
  auto AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  // Optional analyses.
  auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;

  return combineInstructionsOverFunction(F, Worklist, AA, AC, TLI, DT,
                                         ExpensiveCombines, LI);
}

char InstructionCombiningPass::ID = 0;
INITIALIZE_PASS_BEGIN(InstructionCombiningPass, "instcombine",
                      "Combine redundant instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_END(InstructionCombiningPass, "instcombine",
                    "Combine redundant instructions", false, false)

// Initialization Routines
void llvm::initializeInstCombine(PassRegistry &Registry) {
  initializeInstructionCombiningPassPass(Registry);
}

void LLVMInitializeInstCombine(LLVMPassRegistryRef R) {
  initializeInstructionCombiningPassPass(*unwrap(R));
}

FunctionPass *llvm::createInstructionCombiningPass(bool ExpensiveCombines) {
  return new InstructionCombiningPass(ExpensiveCombines);
}
