#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===- GVNSink.cpp - sink expressions into successors -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file GVNSink.cpp
/// This pass attempts to sink instructions into successors, reducing static
/// instruction count and enabling if-conversion.
///
/// We use a variant of global value numbering to decide what can be sunk.
/// Consider:
///
/// [ %a1 = add i32 %b, 1  ]   [ %c1 = add i32 %d, 1  ]
/// [ %a2 = xor i32 %a1, 1 ]   [ %c2 = xor i32 %c1, 1 ]
///                  \           /
///            [ %e = phi i32 %a2, %c2 ]
///            [ add i32 %e, 4         ]
///
///
/// GVN would number %a1 and %c1 differently because they compute different
/// results - the VN of an instruction is a function of its opcode and the
/// transitive closure of its operands. This is the key property for hoisting
/// and CSE.
///
/// What we want when sinking however is for a numbering that is a function of
/// the *uses* of an instruction, which allows us to answer the question "if I
/// replace %a1 with %c1, will it contribute in an equivalent way to all
/// successive instructions?". The PostValueTable class in GVN provides this
/// mapping.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/GVNExpression.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <unordered_set>
using namespace llvm;

#define DEBUG_TYPE "gvn-sink"

STATISTIC(NumRemoved, "Number of instructions removed");

namespace llvm {
namespace GVNExpression {

LLVM_DUMP_METHOD void Expression::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

} // namespace GVNExpression
} // namespace llvm

namespace {

static bool isMemoryInst(const Instruction *I) {
  return isa<LoadInst>(I) || isa<StoreInst>(I) ||
         (isa<InvokeInst>(I) && !cast<InvokeInst>(I)->doesNotAccessMemory()) ||
         (isa<CallInst>(I) && !cast<CallInst>(I)->doesNotAccessMemory());
}

/// Iterates through instructions in a set of blocks in reverse order from the
/// first non-terminator. For example (assume all blocks have size n):
///   LockstepReverseIterator I([B1, B2, B3]);
///   *I-- = [B1[n], B2[n], B3[n]];
///   *I-- = [B1[n-1], B2[n-1], B3[n-1]];
///   *I-- = [B1[n-2], B2[n-2], B3[n-2]];
///   ...
///
/// It continues until all blocks have been exhausted. Use \c getActiveBlocks()
/// to
/// determine which blocks are still going and the order they appear in the
/// list returned by operator*.
class LockstepReverseIterator {
  ArrayRef<BasicBlock *> Blocks;
  SmallPtrSet<BasicBlock *, 4> ActiveBlocks;
  SmallVector<Instruction *, 4> Insts;
  bool Fail;

public:
  LockstepReverseIterator(ArrayRef<BasicBlock *> Blocks) : Blocks(Blocks) {
    reset();
  }

  void reset() {
    Fail = false;
    ActiveBlocks.clear();
    for (BasicBlock *BB : Blocks) {
      PassPrediction::PassPeeper(__FILE__, 1128); // for-range
      ActiveBlocks.insert(BB);
    }
    Insts.clear();
    for (BasicBlock *BB : Blocks) {
      PassPrediction::PassPeeper(__FILE__, 1129); // for-range
      if (BB->size() <= 1) {
        // Block wasn't big enough - only contained a terminator.
        PassPrediction::PassPeeper(__FILE__, 1130); // if
        ActiveBlocks.erase(BB);
        continue;
      }
      Insts.push_back(BB->getTerminator()->getPrevNode());
    }
    if (Insts.empty()) {
      PassPrediction::PassPeeper(__FILE__, 1131); // if
      Fail = true;
    }
  }

  bool isValid() const { return !Fail; }
  ArrayRef<Instruction *> operator*() const { return Insts; }
  SmallPtrSet<BasicBlock *, 4> &getActiveBlocks() { return ActiveBlocks; }

  void restrictToBlocks(SmallPtrSetImpl<BasicBlock *> &Blocks) {
    for (auto II = Insts.begin(); II != Insts.end();) {
      PassPrediction::PassPeeper(__FILE__, 1132); // for
      if (std::find(Blocks.begin(), Blocks.end(), (*II)->getParent()) ==
          Blocks.end()) {
        PassPrediction::PassPeeper(__FILE__, 1133); // if
        ActiveBlocks.erase((*II)->getParent());
        II = Insts.erase(II);
      } else {
        PassPrediction::PassPeeper(__FILE__, 1134); // else
        ++II;
      }
    }
  }

  void operator--() {
    if (Fail) {
      PassPrediction::PassPeeper(__FILE__, 1135); // if
      return;
    }
    SmallVector<Instruction *, 4> NewInsts;
    for (auto *Inst : Insts) {
      PassPrediction::PassPeeper(__FILE__, 1136); // for-range
      if (Inst == &Inst->getParent()->front()) {
        PassPrediction::PassPeeper(__FILE__, 1137); // if
        ActiveBlocks.erase(Inst->getParent());
      } else {
        PassPrediction::PassPeeper(__FILE__, 1138); // else
        NewInsts.push_back(Inst->getPrevNode());
      }
    }
    if (NewInsts.empty()) {
      PassPrediction::PassPeeper(__FILE__, 1139); // if
      Fail = true;
      return;
    }
    Insts = NewInsts;
  }
};

//===----------------------------------------------------------------------===//

/// Candidate solution for sinking. There may be different ways to
/// sink instructions, differing in the number of instructions sunk,
/// the number of predecessors sunk from and the number of PHIs
/// required.
struct SinkingInstructionCandidate {
  unsigned NumBlocks;
  unsigned NumInstructions;
  unsigned NumPHIs;
  unsigned NumMemoryInsts;
  int Cost = -1;
  SmallVector<BasicBlock *, 4> Blocks;

  void calculateCost(unsigned NumOrigPHIs, unsigned NumOrigBlocks) {
    unsigned NumExtraPHIs = NumPHIs - NumOrigPHIs;
    unsigned SplitEdgeCost = (NumOrigBlocks > NumBlocks) ? 2 : 0;
    Cost = (NumInstructions * (NumBlocks - 1)) -
           (NumExtraPHIs *
            NumExtraPHIs) // PHIs are expensive, so make sure they're worth it.
           - SplitEdgeCost;
  }
  bool operator>(const SinkingInstructionCandidate &Other) const {
    return Cost > Other.Cost;
  }
};

#ifndef NDEBUG
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const SinkingInstructionCandidate &C) {
  OS << "<Candidate Cost=" << C.Cost << " #Blocks=" << C.NumBlocks
     << " #Insts=" << C.NumInstructions << " #PHIs=" << C.NumPHIs << ">";
  return OS;
}
#endif

//===----------------------------------------------------------------------===//

/// Describes a PHI node that may or may not exist. These track the PHIs
/// that must be created if we sunk a sequence of instructions. It provides
/// a hash function for efficient equality comparisons.
class ModelledPHI {
  SmallVector<Value *, 4> Values;
  SmallVector<BasicBlock *, 4> Blocks;

public:
  ModelledPHI() {}
  ModelledPHI(const PHINode *PN) {
    for (unsigned I = 0, E = PN->getNumIncomingValues(); I != E; ++I) {
      PassPrediction::PassPeeper(__FILE__, 1140); // for
      Blocks.push_back(PN->getIncomingBlock(I));
    }
    std::sort(Blocks.begin(), Blocks.end());

    // This assumes the PHI is already well-formed and there aren't conflicting
    // incoming values for the same block.
    for (auto *B : Blocks) {
      PassPrediction::PassPeeper(__FILE__, 1141); // for-range
      Values.push_back(PN->getIncomingValueForBlock(B));
    }
  }
  /// Create a dummy ModelledPHI that will compare unequal to any other
  /// ModelledPHI without the same ID. \note This is specifically for
  /// DenseMapInfo - do not use this!
  static ModelledPHI createDummy(size_t ID) {
    ModelledPHI M;
    M.Values.push_back(reinterpret_cast<Value *>(ID));
    return M;
  }

  /// Create a PHI from an array of incoming values and incoming blocks.
  template <typename VArray, typename BArray>
  ModelledPHI(const VArray &V, const BArray &B) {
    std::copy(V.begin(), V.end(), std::back_inserter(Values));
    std::copy(B.begin(), B.end(), std::back_inserter(Blocks));
  }

  /// Create a PHI from [I[OpNum] for I in Insts].
  template <typename BArray>
  ModelledPHI(ArrayRef<Instruction *> Insts, unsigned OpNum, const BArray &B) {
    std::copy(B.begin(), B.end(), std::back_inserter(Blocks));
    for (auto *I : Insts) {
      PassPrediction::PassPeeper(__FILE__, 1142); // for-range
      Values.push_back(I->getOperand(OpNum));
    }
  }

  /// Restrict the PHI's contents down to only \c NewBlocks.
  /// \c NewBlocks must be a subset of \c this->Blocks.
  void restrictToBlocks(const SmallPtrSetImpl<BasicBlock *> &NewBlocks) {
    auto BI = Blocks.begin();
    auto VI = Values.begin();
    while (BI != Blocks.end()) {
      assert(VI != Values.end());
      if (std::find(NewBlocks.begin(), NewBlocks.end(), *BI) ==
          NewBlocks.end()) {
        PassPrediction::PassPeeper(__FILE__, 1143); // if
        BI = Blocks.erase(BI);
        VI = Values.erase(VI);
      } else {
        PassPrediction::PassPeeper(__FILE__, 1144); // else
        ++BI;
        ++VI;
      }
    }
    assert(Blocks.size() == NewBlocks.size());
  }

  ArrayRef<Value *> getValues() const { return Values; }

  bool areAllIncomingValuesSame() const {
    return all_of(Values, [&](Value *V) { return V == Values[0]; });
  }
  bool areAllIncomingValuesSameType() const {
    return all_of(
        Values, [&](Value *V) { return V->getType() == Values[0]->getType(); });
  }
  bool areAnyIncomingValuesConstant() const {
    return any_of(Values, [&](Value *V) { return isa<Constant>(V); });
  }
  // Hash functor
  unsigned hash() const {
    return (unsigned)hash_combine_range(Values.begin(), Values.end());
  }
  bool operator==(const ModelledPHI &Other) const {
    return Values == Other.Values && Blocks == Other.Blocks;
  }
};

template <typename ModelledPHI> struct DenseMapInfo {
  static inline ModelledPHI &getEmptyKey() {
    static ModelledPHI Dummy = ModelledPHI::createDummy(0);
    return Dummy;
  }
  static inline ModelledPHI &getTombstoneKey() {
    static ModelledPHI Dummy = ModelledPHI::createDummy(1);
    return Dummy;
  }
  static unsigned getHashValue(const ModelledPHI &V) { return V.hash(); }
  static bool isEqual(const ModelledPHI &LHS, const ModelledPHI &RHS) {
    return LHS == RHS;
  }
};

typedef DenseSet<ModelledPHI, DenseMapInfo<ModelledPHI>> ModelledPHISet;

//===----------------------------------------------------------------------===//
//                             ValueTable
//===----------------------------------------------------------------------===//
// This is a value number table where the value number is a function of the
// *uses* of a value, rather than its operands. Thus, if VN(A) == VN(B) we know
// that the program would be equivalent if we replaced A with PHI(A, B).
//===----------------------------------------------------------------------===//

/// A GVN expression describing how an instruction is used. The operands
/// field of BasicExpression is used to store uses, not operands.
///
/// This class also contains fields for discriminators used when determining
/// equivalence of instructions with sideeffects.
class InstructionUseExpr : public GVNExpression::BasicExpression {
  unsigned MemoryUseOrder = -1;
  bool Volatile = false;

public:
  InstructionUseExpr(Instruction *I, ArrayRecycler<Value *> &R,
                     BumpPtrAllocator &A)
      : GVNExpression::BasicExpression(I->getNumUses()) {
    allocateOperands(R, A);
    setOpcode(I->getOpcode());
    setType(I->getType());

    for (auto &U : I->uses()) {
      PassPrediction::PassPeeper(__FILE__, 1145); // for-range
      op_push_back(U.getUser());
    }
    std::sort(op_begin(), op_end());
  }
  void setMemoryUseOrder(unsigned MUO) { MemoryUseOrder = MUO; }
  void setVolatile(bool V) { Volatile = V; }

  virtual hash_code getHashValue() const {
    return hash_combine(GVNExpression::BasicExpression::getHashValue(),
                        MemoryUseOrder, Volatile);
  }

  template <typename Function> hash_code getHashValue(Function MapFn) {
    hash_code H =
        hash_combine(getOpcode(), getType(), MemoryUseOrder, Volatile);
    for (auto *V : operands()) {
      PassPrediction::PassPeeper(__FILE__, 1146); // for-range
      H = hash_combine(H, MapFn(V));
    }
    return H;
  }
};

class ValueTable {
  DenseMap<Value *, uint32_t> ValueNumbering;
  DenseMap<GVNExpression::Expression *, uint32_t> ExpressionNumbering;
  DenseMap<size_t, uint32_t> HashNumbering;
  BumpPtrAllocator Allocator;
  ArrayRecycler<Value *> Recycler;
  uint32_t nextValueNumber;

  /// Create an expression for I based on its opcode and its uses. If I
  /// touches or reads memory, the expression is also based upon its memory
  /// order - see \c getMemoryUseOrder().
  InstructionUseExpr *createExpr(Instruction *I) {
    InstructionUseExpr *E =
        new (Allocator) InstructionUseExpr(I, Recycler, Allocator);
    if (isMemoryInst(I)) {
      PassPrediction::PassPeeper(__FILE__, 1147); // if
      E->setMemoryUseOrder(getMemoryUseOrder(I));
    }

    if (CmpInst *C = dyn_cast<CmpInst>(I)) {
      PassPrediction::PassPeeper(__FILE__, 1148); // if
      CmpInst::Predicate Predicate = C->getPredicate();
      E->setOpcode((C->getOpcode() << 8) | Predicate);
    }
    return E;
  }

  /// Helper to compute the value number for a memory instruction
  /// (LoadInst/StoreInst), including checking the memory ordering and
  /// volatility.
  template <class Inst> InstructionUseExpr *createMemoryExpr(Inst *I) {
    if (isStrongerThanUnordered(I->getOrdering()) || I->isAtomic()) {
      PassPrediction::PassPeeper(__FILE__, 1149); // if
      return nullptr;
    }
    InstructionUseExpr *E = createExpr(I);
    E->setVolatile(I->isVolatile());
    return E;
  }

public:
  /// Returns the value number for the specified value, assigning
  /// it a new number if it did not have one before.
  uint32_t lookupOrAdd(Value *V) {
    auto VI = ValueNumbering.find(V);
    if (VI != ValueNumbering.end()) {
      PassPrediction::PassPeeper(__FILE__, 1150); // if
      return VI->second;
    }

    if (!isa<Instruction>(V)) {
      PassPrediction::PassPeeper(__FILE__, 1151); // if
      ValueNumbering[V] = nextValueNumber;
      return nextValueNumber++;
    }

    Instruction *I = cast<Instruction>(V);
    InstructionUseExpr *exp = nullptr;
    switch (I->getOpcode()) {
    case Instruction::Load:
      PassPrediction::PassPeeper(__FILE__, 1152); // case

      exp = createMemoryExpr(cast<LoadInst>(I));
      PassPrediction::PassPeeper(__FILE__, 1153); // break
      break;
    case Instruction::Store:
      PassPrediction::PassPeeper(__FILE__, 1154); // case

      exp = createMemoryExpr(cast<StoreInst>(I));
      PassPrediction::PassPeeper(__FILE__, 1155); // break
      break;
    case Instruction::Call:
      PassPrediction::PassPeeper(__FILE__, 1156); // case

    case Instruction::Invoke:
      PassPrediction::PassPeeper(__FILE__, 1157); // case

    case Instruction::Add:
      PassPrediction::PassPeeper(__FILE__, 1158); // case

    case Instruction::FAdd:
      PassPrediction::PassPeeper(__FILE__, 1159); // case

    case Instruction::Sub:
      PassPrediction::PassPeeper(__FILE__, 1160); // case

    case Instruction::FSub:
      PassPrediction::PassPeeper(__FILE__, 1161); // case

    case Instruction::Mul:
      PassPrediction::PassPeeper(__FILE__, 1162); // case

    case Instruction::FMul:
      PassPrediction::PassPeeper(__FILE__, 1163); // case

    case Instruction::UDiv:
      PassPrediction::PassPeeper(__FILE__, 1164); // case

    case Instruction::SDiv:
      PassPrediction::PassPeeper(__FILE__, 1165); // case

    case Instruction::FDiv:
      PassPrediction::PassPeeper(__FILE__, 1166); // case

    case Instruction::URem:
      PassPrediction::PassPeeper(__FILE__, 1167); // case

    case Instruction::SRem:
      PassPrediction::PassPeeper(__FILE__, 1168); // case

    case Instruction::FRem:
      PassPrediction::PassPeeper(__FILE__, 1169); // case

    case Instruction::Shl:
      PassPrediction::PassPeeper(__FILE__, 1170); // case

    case Instruction::LShr:
      PassPrediction::PassPeeper(__FILE__, 1171); // case

    case Instruction::AShr:
      PassPrediction::PassPeeper(__FILE__, 1172); // case

    case Instruction::And:
      PassPrediction::PassPeeper(__FILE__, 1173); // case

    case Instruction::Or:
      PassPrediction::PassPeeper(__FILE__, 1174); // case

    case Instruction::Xor:
      PassPrediction::PassPeeper(__FILE__, 1175); // case

    case Instruction::ICmp:
      PassPrediction::PassPeeper(__FILE__, 1176); // case

    case Instruction::FCmp:
      PassPrediction::PassPeeper(__FILE__, 1177); // case

    case Instruction::Trunc:
      PassPrediction::PassPeeper(__FILE__, 1178); // case

    case Instruction::ZExt:
      PassPrediction::PassPeeper(__FILE__, 1179); // case

    case Instruction::SExt:
      PassPrediction::PassPeeper(__FILE__, 1180); // case

    case Instruction::FPToUI:
      PassPrediction::PassPeeper(__FILE__, 1181); // case

    case Instruction::FPToSI:
      PassPrediction::PassPeeper(__FILE__, 1182); // case

    case Instruction::UIToFP:
      PassPrediction::PassPeeper(__FILE__, 1183); // case

    case Instruction::SIToFP:
      PassPrediction::PassPeeper(__FILE__, 1184); // case

    case Instruction::FPTrunc:
      PassPrediction::PassPeeper(__FILE__, 1185); // case

    case Instruction::FPExt:
      PassPrediction::PassPeeper(__FILE__, 1186); // case

    case Instruction::PtrToInt:
      PassPrediction::PassPeeper(__FILE__, 1187); // case

    case Instruction::IntToPtr:
      PassPrediction::PassPeeper(__FILE__, 1188); // case

    case Instruction::BitCast:
      PassPrediction::PassPeeper(__FILE__, 1189); // case

    case Instruction::Select:
      PassPrediction::PassPeeper(__FILE__, 1190); // case

    case Instruction::ExtractElement:
      PassPrediction::PassPeeper(__FILE__, 1191); // case

    case Instruction::InsertElement:
      PassPrediction::PassPeeper(__FILE__, 1192); // case

    case Instruction::ShuffleVector:
      PassPrediction::PassPeeper(__FILE__, 1193); // case

    case Instruction::InsertValue:
      PassPrediction::PassPeeper(__FILE__, 1194); // case

    case Instruction::GetElementPtr:
      PassPrediction::PassPeeper(__FILE__, 1195); // case

      exp = createExpr(I);
      PassPrediction::PassPeeper(__FILE__, 1196); // break
      break;
    default:
      PassPrediction::PassPeeper(__FILE__, 1197); // break
      break;
    }

    if (!exp) {
      PassPrediction::PassPeeper(__FILE__, 1198); // if
      ValueNumbering[V] = nextValueNumber;
      return nextValueNumber++;
    }

    uint32_t e = ExpressionNumbering[exp];
    if (!e) {
      PassPrediction::PassPeeper(__FILE__, 1199); // if
      hash_code H = exp->getHashValue([=](Value *V) { return lookupOrAdd(V); });
      auto I = HashNumbering.find(H);
      if (I != HashNumbering.end()) {
        PassPrediction::PassPeeper(__FILE__, 1200); // if
        e = I->second;
      } else {
        PassPrediction::PassPeeper(__FILE__, 1201); // else
        e = nextValueNumber++;
        HashNumbering[H] = e;
        ExpressionNumbering[exp] = e;
      }
    }
    ValueNumbering[V] = e;
    return e;
  }

  /// Returns the value number of the specified value. Fails if the value has
  /// not yet been numbered.
  uint32_t lookup(Value *V) const {
    auto VI = ValueNumbering.find(V);
    assert(VI != ValueNumbering.end() && "Value not numbered?");
    return VI->second;
  }

  /// Removes all value numberings and resets the value table.
  void clear() {
    ValueNumbering.clear();
    ExpressionNumbering.clear();
    HashNumbering.clear();
    Recycler.clear(Allocator);
    nextValueNumber = 1;
  }

  ValueTable() : nextValueNumber(1) {}

  /// \c Inst uses or touches memory. Return an ID describing the memory state
  /// at \c Inst such that if getMemoryUseOrder(I1) == getMemoryUseOrder(I2),
  /// the exact same memory operations happen after I1 and I2.
  ///
  /// This is a very hard problem in general, so we use domain-specific
  /// knowledge that we only ever check for equivalence between blocks sharing a
  /// single immediate successor that is common, and when determining if I1 ==
  /// I2 we will have already determined that next(I1) == next(I2). This
  /// inductive property allows us to simply return the value number of the next
  /// instruction that defines memory.
  uint32_t getMemoryUseOrder(Instruction *Inst) {
    auto *BB = Inst->getParent();
    for (auto I = std::next(Inst->getIterator()), E = BB->end();
         I != E && !I->isTerminator(); ++I) {
      PassPrediction::PassPeeper(__FILE__, 1202); // for
      if (!isMemoryInst(&*I)) {
        PassPrediction::PassPeeper(__FILE__, 1203); // if
        continue;
      }
      if (isa<LoadInst>(&*I)) {
        PassPrediction::PassPeeper(__FILE__, 1204); // if
        continue;
      }
      CallInst *CI = dyn_cast<CallInst>(&*I);
      if (CI && CI->onlyReadsMemory()) {
        PassPrediction::PassPeeper(__FILE__, 1205); // if
        continue;
      }
      InvokeInst *II = dyn_cast<InvokeInst>(&*I);
      if (II && II->onlyReadsMemory()) {
        PassPrediction::PassPeeper(__FILE__, 1206); // if
        continue;
      }
      return lookupOrAdd(&*I);
    }
    return 0;
  }
};

//===----------------------------------------------------------------------===//

class GVNSink {
public:
  GVNSink() : VN() {}
  bool run(Function &F) {
    DEBUG(dbgs() << "GVNSink: running on function @" << F.getName() << "\n");

    unsigned NumSunk = 0;
    ReversePostOrderTraversal<Function *> RPOT(&F);
    for (auto *N : RPOT) {
      PassPrediction::PassPeeper(__FILE__, 1207); // for-range
      NumSunk += sinkBB(N);
    }

    return NumSunk > 0;
  }

private:
  ValueTable VN;

  bool isInstructionBlacklisted(Instruction *I) {
    // These instructions may change or break semantics if moved.
    if (isa<PHINode>(I) || I->isEHPad() || isa<AllocaInst>(I) ||
        I->getType()->isTokenTy()) {
      PassPrediction::PassPeeper(__FILE__, 1208); // if
      return true;
    }
    return false;
  }

  /// The main heuristic function. Analyze the set of instructions pointed to by
  /// LRI and return a candidate solution if these instructions can be sunk, or
  /// None otherwise.
  Optional<SinkingInstructionCandidate> analyzeInstructionForSinking(
      LockstepReverseIterator &LRI, unsigned &InstNum, unsigned &MemoryInstNum,
      ModelledPHISet &NeededPHIs, SmallPtrSetImpl<Value *> &PHIContents);

  /// Create a ModelledPHI for each PHI in BB, adding to PHIs.
  void analyzeInitialPHIs(BasicBlock *BB, ModelledPHISet &PHIs,
                          SmallPtrSetImpl<Value *> &PHIContents) {
    for (auto &I : *BB) {
      PassPrediction::PassPeeper(__FILE__, 1209); // for-range
      auto *PN = dyn_cast<PHINode>(&I);
      if (!PN) {
        PassPrediction::PassPeeper(__FILE__, 1210); // if
        return;
      }

      auto MPHI = ModelledPHI(PN);
      PHIs.insert(MPHI);
      for (auto *V : MPHI.getValues()) {
        PassPrediction::PassPeeper(__FILE__, 1211); // for-range
        PHIContents.insert(V);
      }
    }
  }

  /// The main instruction sinking driver. Set up state and try and sink
  /// instructions into BBEnd from its predecessors.
  unsigned sinkBB(BasicBlock *BBEnd);

  /// Perform the actual mechanics of sinking an instruction from Blocks into
  /// BBEnd, which is their only successor.
  void sinkLastInstruction(ArrayRef<BasicBlock *> Blocks, BasicBlock *BBEnd);

  /// Remove PHIs that all have the same incoming value.
  void foldPointlessPHINodes(BasicBlock *BB) {
    auto I = BB->begin();
    while (PHINode *PN = dyn_cast<PHINode>(I++)) {
      PassPrediction::PassPeeper(__FILE__, 1212); // while
      if (!all_of(PN->incoming_values(), [&](const Value *V) {
            return V == PN->getIncomingValue(0);
          })) {
        PassPrediction::PassPeeper(__FILE__, 1213); // if
        continue;
      }
      if (PN->getIncomingValue(0) != PN) {
        PassPrediction::PassPeeper(__FILE__, 1214); // if
        PN->replaceAllUsesWith(PN->getIncomingValue(0));
      } else {
        PassPrediction::PassPeeper(__FILE__, 1215); // else
        PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
      }
      PN->eraseFromParent();
    }
  }
};

Optional<SinkingInstructionCandidate> GVNSink::analyzeInstructionForSinking(
    LockstepReverseIterator &LRI, unsigned &InstNum, unsigned &MemoryInstNum,
    ModelledPHISet &NeededPHIs, SmallPtrSetImpl<Value *> &PHIContents) {
  auto Insts = *LRI;
  DEBUG(dbgs() << " -- Analyzing instruction set: [\n"; for (auto *I
                                                             : Insts) {
    I->dump();
  } dbgs() << " ]\n";);

  DenseMap<uint32_t, unsigned> VNums;
  for (auto *I : Insts) {
    PassPrediction::PassPeeper(__FILE__, 1216); // for-range
    uint32_t N = VN.lookupOrAdd(I);
    DEBUG(dbgs() << " VN=" << utohexstr(N) << " for" << *I << "\n");
    if (N == ~0U) {
      PassPrediction::PassPeeper(__FILE__, 1217); // if
      return None;
    }
    VNums[N]++;
  }
  unsigned VNumToSink =
      std::max_element(VNums.begin(), VNums.end(),
                       [](const std::pair<uint32_t, unsigned> &I,
                          const std::pair<uint32_t, unsigned> &J) {
                         return I.second < J.second;
                       })
          ->first;

  if (VNums[VNumToSink] == 1) {
    // Can't sink anything!
    PassPrediction::PassPeeper(__FILE__, 1218); // if
    return None;
  }

  // Now restrict the number of incoming blocks down to only those with
  // VNumToSink.
  auto &ActivePreds = LRI.getActiveBlocks();
  unsigned InitialActivePredSize = ActivePreds.size();
  SmallVector<Instruction *, 4> NewInsts;
  for (auto *I : Insts) {
    PassPrediction::PassPeeper(__FILE__, 1219); // for-range
    if (VN.lookup(I) != VNumToSink) {
      PassPrediction::PassPeeper(__FILE__, 1220); // if
      ActivePreds.erase(I->getParent());
    } else {
      PassPrediction::PassPeeper(__FILE__, 1221); // else
      NewInsts.push_back(I);
    }
  }
  for (auto *I : NewInsts) {
    PassPrediction::PassPeeper(__FILE__, 1222); // for-range
    if (isInstructionBlacklisted(I)) {
      PassPrediction::PassPeeper(__FILE__, 1223); // if
      return None;
    }
  }

  // If we've restricted the incoming blocks, restrict all needed PHIs also
  // to that set.
  bool RecomputePHIContents = false;
  if (ActivePreds.size() != InitialActivePredSize) {
    PassPrediction::PassPeeper(__FILE__, 1224); // if
    ModelledPHISet NewNeededPHIs;
    for (auto P : NeededPHIs) {
      PassPrediction::PassPeeper(__FILE__, 1225); // for-range
      P.restrictToBlocks(ActivePreds);
      NewNeededPHIs.insert(P);
    }
    NeededPHIs = NewNeededPHIs;
    LRI.restrictToBlocks(ActivePreds);
    RecomputePHIContents = true;
  }

  // The sunk instruction's results.
  ModelledPHI NewPHI(NewInsts, ActivePreds);

  // Does sinking this instruction render previous PHIs redundant?
  if (NeededPHIs.find(NewPHI) != NeededPHIs.end()) {
    PassPrediction::PassPeeper(__FILE__, 1226); // if
    NeededPHIs.erase(NewPHI);
    RecomputePHIContents = true;
  }

  if (RecomputePHIContents) {
    // The needed PHIs have changed, so recompute the set of all needed
    // values.
    PassPrediction::PassPeeper(__FILE__, 1227); // if
    PHIContents.clear();
    for (auto &PHI : NeededPHIs) {
      PassPrediction::PassPeeper(__FILE__, 1228); // for-range
      PHIContents.insert(PHI.getValues().begin(), PHI.getValues().end());
    }
  }

  // Is this instruction required by a later PHI that doesn't match this PHI?
  // if so, we can't sink this instruction.
  for (auto *V : NewPHI.getValues()) {
    PassPrediction::PassPeeper(__FILE__, 1229); // for-range
    if (PHIContents.count(V)) {
      // V exists in this PHI, but the whole PHI is different to NewPHI
      // (else it would have been removed earlier). We cannot continue
      // because this isn't representable.
      PassPrediction::PassPeeper(__FILE__, 1230); // if
      return None;
    }
  }

  // Which operands need PHIs?
  // FIXME: If any of these fail, we should partition up the candidates to
  // try and continue making progress.
  Instruction *I0 = NewInsts[0];
  for (unsigned OpNum = 0, E = I0->getNumOperands(); OpNum != E; ++OpNum) {
    PassPrediction::PassPeeper(__FILE__, 1231); // for
    ModelledPHI PHI(NewInsts, OpNum, ActivePreds);
    if (PHI.areAllIncomingValuesSame()) {
      PassPrediction::PassPeeper(__FILE__, 1232); // if
      continue;
    }
    if (!canReplaceOperandWithVariable(I0, OpNum)) {
      // We can 't create a PHI from this instruction!
      PassPrediction::PassPeeper(__FILE__, 1233); // if
      return None;
    }
    if (NeededPHIs.count(PHI)) {
      PassPrediction::PassPeeper(__FILE__, 1234); // if
      continue;
    }
    if (!PHI.areAllIncomingValuesSameType()) {
      PassPrediction::PassPeeper(__FILE__, 1235); // if
      return None;
    }
    // Don't create indirect calls! The called value is the final operand.
    if ((isa<CallInst>(I0) || isa<InvokeInst>(I0)) && OpNum == E - 1 &&
        PHI.areAnyIncomingValuesConstant()) {
      PassPrediction::PassPeeper(__FILE__, 1236); // if
      return None;
    }

    NeededPHIs.reserve(NeededPHIs.size());
    NeededPHIs.insert(PHI);
    PHIContents.insert(PHI.getValues().begin(), PHI.getValues().end());
  }

  if (isMemoryInst(NewInsts[0])) {
    PassPrediction::PassPeeper(__FILE__, 1237); // if
    ++MemoryInstNum;
  }

  SinkingInstructionCandidate Cand;
  Cand.NumInstructions = ++InstNum;
  Cand.NumMemoryInsts = MemoryInstNum;
  Cand.NumBlocks = ActivePreds.size();
  Cand.NumPHIs = NeededPHIs.size();
  for (auto *C : ActivePreds) {
    PassPrediction::PassPeeper(__FILE__, 1238); // for-range
    Cand.Blocks.push_back(C);
  }

  return Cand;
}

unsigned GVNSink::sinkBB(BasicBlock *BBEnd) {
  DEBUG(dbgs() << "GVNSink: running on basic block ";
        BBEnd->printAsOperand(dbgs()); dbgs() << "\n");
  SmallVector<BasicBlock *, 4> Preds;
  for (auto *B : predecessors(BBEnd)) {
    PassPrediction::PassPeeper(__FILE__, 1239); // for-range
    auto *T = B->getTerminator();
    if (isa<BranchInst>(T) || isa<SwitchInst>(T)) {
      PassPrediction::PassPeeper(__FILE__, 1240); // if
      Preds.push_back(B);
    } else {
      PassPrediction::PassPeeper(__FILE__, 1241); // else
      return 0;
    }
  }
  if (Preds.size() < 2) {
    PassPrediction::PassPeeper(__FILE__, 1242); // if
    return 0;
  }
  std::sort(Preds.begin(), Preds.end());

  unsigned NumOrigPreds = Preds.size();
  // We can only sink instructions through unconditional branches.
  for (auto I = Preds.begin(); I != Preds.end();) {
    PassPrediction::PassPeeper(__FILE__, 1243); // for
    if ((*I)->getTerminator()->getNumSuccessors() != 1) {
      PassPrediction::PassPeeper(__FILE__, 1244); // if
      I = Preds.erase(I);
    } else {
      PassPrediction::PassPeeper(__FILE__, 1245); // else
      ++I;
    }
  }

  LockstepReverseIterator LRI(Preds);
  SmallVector<SinkingInstructionCandidate, 4> Candidates;
  unsigned InstNum = 0, MemoryInstNum = 0;
  ModelledPHISet NeededPHIs;
  SmallPtrSet<Value *, 4> PHIContents;
  analyzeInitialPHIs(BBEnd, NeededPHIs, PHIContents);
  unsigned NumOrigPHIs = NeededPHIs.size();

  while (LRI.isValid()) {
    PassPrediction::PassPeeper(__FILE__, 1246); // while
    auto Cand = analyzeInstructionForSinking(LRI, InstNum, MemoryInstNum,
                                             NeededPHIs, PHIContents);
    if (!Cand) {
      PassPrediction::PassPeeper(__FILE__, 1247); // if
      break;
    }
    Cand->calculateCost(NumOrigPHIs, Preds.size());
    Candidates.emplace_back(*Cand);
    --LRI;
  }

  std::stable_sort(Candidates.begin(), Candidates.end(),
                   [](const SinkingInstructionCandidate &A,
                      const SinkingInstructionCandidate &B) { return A > B; });
  DEBUG(dbgs() << " -- Sinking candidates:\n";
        for (auto &C
             : Candidates) { dbgs() << "  " << C << "\n"; });

  // Pick the top candidate, as long it is positive!
  if (Candidates.empty() || Candidates.front().Cost <= 0) {
    PassPrediction::PassPeeper(__FILE__, 1248); // if
    return 0;
  }
  auto C = Candidates.front();

  DEBUG(dbgs() << " -- Sinking: " << C << "\n");
  BasicBlock *InsertBB = BBEnd;
  if (C.Blocks.size() < NumOrigPreds) {
    DEBUG(dbgs() << " -- Splitting edge to "; BBEnd->printAsOperand(dbgs());
          dbgs() << "\n");
    InsertBB = SplitBlockPredecessors(BBEnd, C.Blocks, ".gvnsink.split");
    if (!InsertBB) {
      DEBUG(dbgs() << " -- FAILED to split edge!\n");
      // Edge couldn't be split.
      return 0;
    }
  }

  for (unsigned I = 0; I < C.NumInstructions; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1249); // for
    sinkLastInstruction(C.Blocks, InsertBB);
  }

  return C.NumInstructions;
}

void GVNSink::sinkLastInstruction(ArrayRef<BasicBlock *> Blocks,
                                  BasicBlock *BBEnd) {
  SmallVector<Instruction *, 4> Insts;
  for (BasicBlock *BB : Blocks) {
    PassPrediction::PassPeeper(__FILE__, 1250); // for-range
    Insts.push_back(BB->getTerminator()->getPrevNode());
  }
  Instruction *I0 = Insts.front();

  SmallVector<Value *, 4> NewOperands;
  for (unsigned O = 0, E = I0->getNumOperands(); O != E; ++O) {
    PassPrediction::PassPeeper(__FILE__, 1251); // for
    bool NeedPHI = any_of(Insts, [&I0, O](const Instruction *I) {
      return I->getOperand(O) != I0->getOperand(O);
    });
    if (!NeedPHI) {
      PassPrediction::PassPeeper(__FILE__, 1252); // if
      NewOperands.push_back(I0->getOperand(O));
      continue;
    }

    // Create a new PHI in the successor block and populate it.
    auto *Op = I0->getOperand(O);
    assert(!Op->getType()->isTokenTy() && "Can't PHI tokens!");
    auto *PN = PHINode::Create(Op->getType(), Insts.size(),
                               Op->getName() + ".sink", &BBEnd->front());
    for (auto *I : Insts) {
      PassPrediction::PassPeeper(__FILE__, 1253); // for-range
      PN->addIncoming(I->getOperand(O), I->getParent());
    }
    NewOperands.push_back(PN);
  }

  // Arbitrarily use I0 as the new "common" instruction; remap its operands
  // and move it to the start of the successor block.
  for (unsigned O = 0, E = I0->getNumOperands(); O != E; ++O) {
    PassPrediction::PassPeeper(__FILE__, 1254); // for
    I0->getOperandUse(O).set(NewOperands[O]);
  }
  I0->moveBefore(&*BBEnd->getFirstInsertionPt());

  // Update metadata and IR flags.
  for (auto *I : Insts) {
    PassPrediction::PassPeeper(__FILE__, 1255); // for-range
    if (I != I0) {
      PassPrediction::PassPeeper(__FILE__, 1256); // if
      combineMetadataForCSE(I0, I);
      I0->andIRFlags(I);
    }
  }

  for (auto *I : Insts) {
    PassPrediction::PassPeeper(__FILE__, 1257); // for-range
    if (I != I0) {
      PassPrediction::PassPeeper(__FILE__, 1258); // if
      I->replaceAllUsesWith(I0);
    }
  }
  foldPointlessPHINodes(BBEnd);

  // Finally nuke all instructions apart from the common instruction.
  for (auto *I : Insts) {
    PassPrediction::PassPeeper(__FILE__, 1259); // for-range
    if (I != I0) {
      PassPrediction::PassPeeper(__FILE__, 1260); // if
      I->eraseFromParent();
    }
  }

  NumRemoved += Insts.size() - 1;
}

////////////////////////////////////////////////////////////////////////////////
// Pass machinery / boilerplate

class GVNSinkLegacyPass : public FunctionPass {
public:
  static char ID;

  GVNSinkLegacyPass() : FunctionPass(ID) {
    initializeGVNSinkLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F)) {
      PassPrediction::PassPeeper(__FILE__, 1261); // if
      return false;
    }
    GVNSink G;
    return G.run(F);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
} // namespace

PreservedAnalyses GVNSinkPass::run(Function &F, FunctionAnalysisManager &AM) {
  GVNSink G;
  if (!G.run(F)) {
    PassPrediction::PassPeeper(__FILE__, 1262); // if
    return PreservedAnalyses::all();
  }

  PreservedAnalyses PA;
  PA.preserve<GlobalsAA>();
  return PA;
}

char GVNSinkLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(GVNSinkLegacyPass, "gvn-sink",
                      "Early GVN sinking of Expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(GVNSinkLegacyPass, "gvn-sink",
                    "Early GVN sinking of Expressions", false, false)

FunctionPass *llvm::createGVNSinkPass() { return new GVNSinkLegacyPass(); }
