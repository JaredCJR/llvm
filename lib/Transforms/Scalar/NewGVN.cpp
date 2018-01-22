#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===---- NewGVN.cpp - Global Value Numbering Pass --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the new LLVM's Global Value Numbering pass.
/// GVN partitions values computed by a function into congruence classes.
/// Values ending up in the same congruence class are guaranteed to be the same
/// for every execution of the program. In that respect, congruency is a
/// compile-time approximation of equivalence of values at runtime.
/// The algorithm implemented here uses a sparse formulation and it's based
/// on the ideas described in the paper:
/// "A Sparse Algorithm for Predicated Global Value Numbering" from
/// Karthik Gargi.
///
/// A brief overview of the algorithm: The algorithm is essentially the same as
/// the standard RPO value numbering algorithm (a good reference is the paper
/// "SCC based value numbering" by L. Taylor Simpson) with one major difference:
/// The RPO algorithm proceeds, on every iteration, to process every reachable
/// block and every instruction in that block.  This is because the standard RPO
/// algorithm does not track what things have the same value number, it only
/// tracks what the value number of a given operation is (the mapping is
/// operation -> value number).  Thus, when a value number of an operation
/// changes, it must reprocess everything to ensure all uses of a value number
/// get updated properly.  In constrast, the sparse algorithm we use *also*
/// tracks what operations have a given value number (IE it also tracks the
/// reverse mapping from value number -> operations with that value number), so
/// that it only needs to reprocess the instructions that are affected when
/// something's value number changes.  The vast majority of complexity and code
/// in this file is devoted to tracking what value numbers could change for what
/// instructions when various things happen.  The rest of the algorithm is
/// devoted to performing symbolic evaluation, forward propagation, and
/// simplification of operations based on the value numbers deduced so far
///
/// In order to make the GVN mostly-complete, we use a technique derived from
/// "Detection of Redundant Expressions: A Complete and Polynomial-time
/// Algorithm in SSA" by R.R. Pai.  The source of incompleteness in most SSA
/// based GVN algorithms is related to their inability to detect equivalence
/// between phi of ops (IE phi(a+b, c+d)) and op of phis (phi(a,c) + phi(b, d)).
/// We resolve this issue by generating the equivalent "phi of ops" form for
/// each op of phis we see, in a way that only takes polynomial time to resolve.
///
/// We also do not perform elimination by using any published algorithm.  All
/// published algorithms are O(Instructions). Instead, we use a technique that
/// is O(number of operations with the same value number), enabling us to skip
/// trying to eliminate things that have unique value numbers.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVNExpression.h"
#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PredicateInfo.h"
#include "llvm/Transforms/Utils/VNCoercion.h"
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>
using namespace llvm;
using namespace PatternMatch;
using namespace llvm::GVNExpression;
using namespace llvm::VNCoercion;
#define DEBUG_TYPE "newgvn"

STATISTIC(NumGVNInstrDeleted, "Number of instructions deleted");
STATISTIC(NumGVNBlocksDeleted, "Number of blocks deleted");
STATISTIC(NumGVNOpsSimplified, "Number of Expressions simplified");
STATISTIC(NumGVNPhisAllSame, "Number of PHIs whos arguments are all the same");
STATISTIC(NumGVNMaxIterations,
          "Maximum Number of iterations it took to converge GVN");
STATISTIC(NumGVNLeaderChanges, "Number of leader changes");
STATISTIC(NumGVNSortedLeaderChanges, "Number of sorted leader changes");
STATISTIC(NumGVNAvoidedSortedLeaderChanges,
          "Number of avoided sorted leader changes");
STATISTIC(NumGVNDeadStores, "Number of redundant/dead stores eliminated");
STATISTIC(NumGVNPHIOfOpsCreated, "Number of PHI of ops created");
STATISTIC(NumGVNPHIOfOpsEliminations,
          "Number of things eliminated using PHI of ops");
DEBUG_COUNTER(VNCounter, "newgvn-vn",
              "Controls which instructions are value numbered")
DEBUG_COUNTER(PHIOfOpsCounter, "newgvn-phi",
              "Controls which instructions we create phi of ops for")
// Currently store defining access refinement is too slow due to basicaa being
// egregiously slow.  This flag lets us keep it working while we work on this
// issue.
static cl::opt<bool> EnableStoreRefinement("enable-store-refinement",
                                           cl::init(false), cl::Hidden);

//===----------------------------------------------------------------------===//
//                                GVN Pass
//===----------------------------------------------------------------------===//

// Anchor methods.
namespace llvm {
namespace GVNExpression {
Expression::~Expression() = default;
BasicExpression::~BasicExpression() = default;
CallExpression::~CallExpression() = default;
LoadExpression::~LoadExpression() = default;
StoreExpression::~StoreExpression() = default;
AggregateValueExpression::~AggregateValueExpression() = default;
PHIExpression::~PHIExpression() = default;
} // namespace GVNExpression
} // namespace llvm

// Tarjan's SCC finding algorithm with Nuutila's improvements
// SCCIterator is actually fairly complex for the simple thing we want.
// It also wants to hand us SCC's that are unrelated to the phi node we ask
// about, and have us process them there or risk redoing work.
// Graph traits over a filter iterator also doesn't work that well here.
// This SCC finder is specialized to walk use-def chains, and only follows
// instructions,
// not generic values (arguments, etc).
struct TarjanSCC {

  TarjanSCC() : Components(1) {}

  void Start(const Instruction *Start) {
    if (Root.lookup(Start) == 0) {
      PassPrediction::PassPeeper(3019); // if
      FindSCC(Start);
    }
  }

  const SmallPtrSetImpl<const Value *> &getComponentFor(const Value *V) const {
    unsigned ComponentID = ValueToComponent.lookup(V);

    assert(ComponentID > 0 &&
           "Asking for a component for a value we never processed");
    return Components[ComponentID];
  }

private:
  void FindSCC(const Instruction *I) {
    Root[I] = ++DFSNum;
    // Store the DFS Number we had before it possibly gets incremented.
    unsigned int OurDFS = DFSNum;
    for (auto &Op : I->operands()) {
      PassPrediction::PassPeeper(3020); // for-range
      if (auto *InstOp = dyn_cast<Instruction>(Op)) {
        PassPrediction::PassPeeper(3021); // if
        if (Root.lookup(Op) == 0) {
          PassPrediction::PassPeeper(3022); // if
          FindSCC(InstOp);
        }
        if (!InComponent.count(Op)) {
          PassPrediction::PassPeeper(3023); // if
          Root[I] = std::min(Root.lookup(I), Root.lookup(Op));
        }
      }
    }
    // See if we really were the root of a component, by seeing if we still have
    // our DFSNumber.  If we do, we are the root of the component, and we have
    // completed a component. If we do not, we are not the root of a component,
    // and belong on the component stack.
    if (Root.lookup(I) == OurDFS) {
      PassPrediction::PassPeeper(3024); // if
      unsigned ComponentID = Components.size();
      Components.resize(Components.size() + 1);
      auto &Component = Components.back();
      Component.insert(I);
      DEBUG(dbgs() << "Component root is " << *I << "\n");
      InComponent.insert(I);
      ValueToComponent[I] = ComponentID;
      // Pop a component off the stack and label it.
      while (!Stack.empty() && Root.lookup(Stack.back()) >= OurDFS) {
        PassPrediction::PassPeeper(3026); // while
        auto *Member = Stack.back();
        DEBUG(dbgs() << "Component member is " << *Member << "\n");
        Component.insert(Member);
        InComponent.insert(Member);
        ValueToComponent[Member] = ComponentID;
        Stack.pop_back();
      }
    } else {
      // Part of a component, push to stack
      PassPrediction::PassPeeper(3025); // else
      Stack.push_back(I);
    }
  }
  unsigned int DFSNum = 1;
  SmallPtrSet<const Value *, 8> InComponent;
  DenseMap<const Value *, unsigned int> Root;
  SmallVector<const Value *, 8> Stack;
  // Store the components as vector of ptr sets, because we need the topo order
  // of SCC's, but not individual member order
  SmallVector<SmallPtrSet<const Value *, 8>, 8> Components;
  DenseMap<const Value *, unsigned> ValueToComponent;
};
// Congruence classes represent the set of expressions/instructions
// that are all the same *during some scope in the function*.
// That is, because of the way we perform equality propagation, and
// because of memory value numbering, it is not correct to assume
// you can willy-nilly replace any member with any other at any
// point in the function.
//
// For any Value in the Member set, it is valid to replace any dominated member
// with that Value.
//
// Every congruence class has a leader, and the leader is used to symbolize
// instructions in a canonical way (IE every operand of an instruction that is a
// member of the same congruence class will always be replaced with leader
// during symbolization).  To simplify symbolization, we keep the leader as a
// constant if class can be proved to be a constant value.  Otherwise, the
// leader is the member of the value set with the smallest DFS number.  Each
// congruence class also has a defining expression, though the expression may be
// null.  If it exists, it can be used for forward propagation and reassociation
// of values.

// For memory, we also track a representative MemoryAccess, and a set of memory
// members for MemoryPhis (which have no real instructions). Note that for
// memory, it seems tempting to try to split the memory members into a
// MemoryCongruenceClass or something.  Unfortunately, this does not work
// easily.  The value numbering of a given memory expression depends on the
// leader of the memory congruence class, and the leader of memory congruence
// class depends on the value numbering of a given memory expression.  This
// leads to wasted propagation, and in some cases, missed optimization.  For
// example: If we had value numbered two stores together before, but now do not,
// we move them to a new value congruence class.  This in turn will move at one
// of the memorydefs to a new memory congruence class.  Which in turn, affects
// the value numbering of the stores we just value numbered (because the memory
// congruence class is part of the value number).  So while theoretically
// possible to split them up, it turns out to be *incredibly* complicated to get
// it to work right, because of the interdependency.  While structurally
// slightly messier, it is algorithmically much simpler and faster to do what we
// do here, and track them both at once in the same class.
// Note: The default iterators for this class iterate over values
class CongruenceClass {
public:
  using MemberType = Value;
  using MemberSet = SmallPtrSet<MemberType *, 4>;
  using MemoryMemberType = MemoryPhi;
  using MemoryMemberSet = SmallPtrSet<const MemoryMemberType *, 2>;

  explicit CongruenceClass(unsigned ID) : ID(ID) {}
  CongruenceClass(unsigned ID, Value *Leader, const Expression *E)
      : ID(ID), RepLeader(Leader), DefiningExpr(E) {}
  unsigned getID() const { return ID; }
  // True if this class has no members left.  This is mainly used for assertion
  // purposes, and for skipping empty classes.
  bool isDead() const {
    // If it's both dead from a value perspective, and dead from a memory
    // perspective, it's really dead.
    return empty() && memory_empty();
  }
  // Leader functions
  Value *getLeader() const { return RepLeader; }
  void setLeader(Value *Leader) { RepLeader = Leader; }
  const std::pair<Value *, unsigned int> &getNextLeader() const {
    return NextLeader;
  }
  void resetNextLeader() { NextLeader = {nullptr, ~0}; }

  void addPossibleNextLeader(std::pair<Value *, unsigned int> LeaderPair) {
    if (LeaderPair.second < NextLeader.second) {
      PassPrediction::PassPeeper(3027); // if
      NextLeader = LeaderPair;
    }
  }

  Value *getStoredValue() const { return RepStoredValue; }
  void setStoredValue(Value *Leader) { RepStoredValue = Leader; }
  const MemoryAccess *getMemoryLeader() const { return RepMemoryAccess; }
  void setMemoryLeader(const MemoryAccess *Leader) { RepMemoryAccess = Leader; }

  // Forward propagation info
  const Expression *getDefiningExpr() const { return DefiningExpr; }

  // Value member set
  bool empty() const { return Members.empty(); }
  unsigned size() const { return Members.size(); }
  MemberSet::const_iterator begin() const { return Members.begin(); }
  MemberSet::const_iterator end() const { return Members.end(); }
  void insert(MemberType *M) { Members.insert(M); }
  void erase(MemberType *M) { Members.erase(M); }
  void swap(MemberSet &Other) { Members.swap(Other); }

  // Memory member set
  bool memory_empty() const { return MemoryMembers.empty(); }
  unsigned memory_size() const { return MemoryMembers.size(); }
  MemoryMemberSet::const_iterator memory_begin() const {
    return MemoryMembers.begin();
  }
  MemoryMemberSet::const_iterator memory_end() const {
    return MemoryMembers.end();
  }
  iterator_range<MemoryMemberSet::const_iterator> memory() const {
    return make_range(memory_begin(), memory_end());
  }
  void memory_insert(const MemoryMemberType *M) { MemoryMembers.insert(M); }
  void memory_erase(const MemoryMemberType *M) { MemoryMembers.erase(M); }

  // Store count
  unsigned getStoreCount() const { return StoreCount; }
  void incStoreCount() { ++StoreCount; }
  void decStoreCount() {
    assert(StoreCount != 0 && "Store count went negative");
    --StoreCount;
  }

  // True if this class has no memory members.
  bool definesNoMemory() const { return StoreCount == 0 && memory_empty(); }

  // Return true if two congruence classes are equivalent to each other.  This
  // means
  // that every field but the ID number and the dead field are equivalent.
  bool isEquivalentTo(const CongruenceClass *Other) const {
    if (!Other) {
      PassPrediction::PassPeeper(3028); // if
      return false;
    }
    if (this == Other) {
      PassPrediction::PassPeeper(3029); // if
      return true;
    }

    if (std::tie(StoreCount, RepLeader, RepStoredValue, RepMemoryAccess) !=
        std::tie(Other->StoreCount, Other->RepLeader, Other->RepStoredValue,
                 Other->RepMemoryAccess)) {
      PassPrediction::PassPeeper(3030); // if
      return false;
    }
    if (DefiningExpr != Other->DefiningExpr) {
      PassPrediction::PassPeeper(3031); // if
      if (!DefiningExpr || !Other->DefiningExpr ||
          *DefiningExpr != *Other->DefiningExpr) {
        PassPrediction::PassPeeper(3032); // if
        return false;
      }
    }
    // We need some ordered set
    std::set<Value *> AMembers(Members.begin(), Members.end());
    std::set<Value *> BMembers(Members.begin(), Members.end());
    return AMembers == BMembers;
  }

private:
  unsigned ID;
  // Representative leader.
  Value *RepLeader = nullptr;
  // The most dominating leader after our current leader, because the member set
  // is not sorted and is expensive to keep sorted all the time.
  std::pair<Value *, unsigned int> NextLeader = {nullptr, ~0U};
  // If this is represented by a store, the value of the store.
  Value *RepStoredValue = nullptr;
  // If this class contains MemoryDefs or MemoryPhis, this is the leading memory
  // access.
  const MemoryAccess *RepMemoryAccess = nullptr;
  // Defining Expression.
  const Expression *DefiningExpr = nullptr;
  // Actual members of this class.
  MemberSet Members;
  // This is the set of MemoryPhis that exist in the class. MemoryDefs and
  // MemoryUses have real instructions representing them, so we only need to
  // track MemoryPhis here.
  MemoryMemberSet MemoryMembers;
  // Number of stores in this congruence class.
  // This is used so we can detect store equivalence changes properly.
  int StoreCount = 0;
};

namespace llvm {
struct ExactEqualsExpression {
  const Expression &E;
  explicit ExactEqualsExpression(const Expression &E) : E(E) {}
  hash_code getComputedHash() const { return E.getComputedHash(); }
  bool operator==(const Expression &Other) const {
    return E.exactlyEquals(Other);
  }
};

template <> struct DenseMapInfo<const Expression *> {
  static const Expression *getEmptyKey() {
    auto Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<const Expression *>::NumLowBitsAvailable;
    return reinterpret_cast<const Expression *>(Val);
  }
  static const Expression *getTombstoneKey() {
    auto Val = static_cast<uintptr_t>(~1U);
    Val <<= PointerLikeTypeTraits<const Expression *>::NumLowBitsAvailable;
    return reinterpret_cast<const Expression *>(Val);
  }
  static unsigned getHashValue(const Expression *E) {
    return E->getComputedHash();
  }
  static unsigned getHashValue(const ExactEqualsExpression &E) {
    return E.getComputedHash();
  }
  static bool isEqual(const ExactEqualsExpression &LHS, const Expression *RHS) {
    if (RHS == getTombstoneKey() || RHS == getEmptyKey()) {
      PassPrediction::PassPeeper(3033); // if
      return false;
    }
    return LHS == *RHS;
  }

  static bool isEqual(const Expression *LHS, const Expression *RHS) {
    if (LHS == RHS) {
      PassPrediction::PassPeeper(3034); // if
      return true;
    }
    if (LHS == getTombstoneKey() || RHS == getTombstoneKey() ||
        LHS == getEmptyKey() || RHS == getEmptyKey()) {
      PassPrediction::PassPeeper(3035); // if
      return false;
    }
    // Compare hashes before equality.  This is *not* what the hashtable does,
    // since it is computing it modulo the number of buckets, whereas we are
    // using the full hash keyspace.  Since the hashes are precomputed, this
    // check is *much* faster than equality.
    if (LHS->getComputedHash() != RHS->getComputedHash()) {
      PassPrediction::PassPeeper(3036); // if
      return false;
    }
    return *LHS == *RHS;
  }
};
} // end namespace llvm

namespace {
class NewGVN {
  Function &F;
  DominatorTree *DT;
  const TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  MemorySSA *MSSA;
  MemorySSAWalker *MSSAWalker;
  const DataLayout &DL;
  std::unique_ptr<PredicateInfo> PredInfo;

  // These are the only two things the create* functions should have
  // side-effects on due to allocating memory.
  mutable BumpPtrAllocator ExpressionAllocator;
  mutable ArrayRecycler<Value *> ArgRecycler;
  mutable TarjanSCC SCCFinder;
  const SimplifyQuery SQ;

  // Number of function arguments, used by ranking
  unsigned int NumFuncArgs;

  // RPOOrdering of basic blocks
  DenseMap<const DomTreeNode *, unsigned> RPOOrdering;

  // Congruence class info.

  // This class is called INITIAL in the paper. It is the class everything
  // startsout in, and represents any value. Being an optimistic analysis,
  // anything in the TOP class has the value TOP, which is indeterminate and
  // equivalent to everything.
  CongruenceClass *TOPClass;
  std::vector<CongruenceClass *> CongruenceClasses;
  unsigned NextCongruenceNum;

  // Value Mappings.
  DenseMap<Value *, CongruenceClass *> ValueToClass;
  DenseMap<Value *, const Expression *> ValueToExpression;
  // Value PHI handling, used to make equivalence between phi(op, op) and
  // op(phi, phi).
  // These mappings just store various data that would normally be part of the
  // IR.
  DenseSet<const Instruction *> PHINodeUses;
  // Map a temporary instruction we created to a parent block.
  DenseMap<const Value *, BasicBlock *> TempToBlock;
  // Map between the temporary phis we created and the real instructions they
  // are known equivalent to.
  DenseMap<const Value *, PHINode *> RealToTemp;
  // In order to know when we should re-process instructions that have
  // phi-of-ops, we track the set of expressions that they needed as
  // leaders. When we discover new leaders for those expressions, we process the
  // associated phi-of-op instructions again in case they have changed.  The
  // other way they may change is if they had leaders, and those leaders
  // disappear.  However, at the point they have leaders, there are uses of the
  // relevant operands in the created phi node, and so they will get reprocessed
  // through the normal user marking we perform.
  mutable DenseMap<const Value *, SmallPtrSet<Value *, 2>> AdditionalUsers;
  DenseMap<const Expression *, SmallPtrSet<Instruction *, 2>>
      ExpressionToPhiOfOps;
  // Map from basic block to the temporary operations we created
  DenseMap<const BasicBlock *, SmallVector<PHINode *, 8>> PHIOfOpsPHIs;
  // Map from temporary operation to MemoryAccess.
  DenseMap<const Instruction *, MemoryUseOrDef *> TempToMemory;
  // Set of all temporary instructions we created.
  DenseSet<Instruction *> AllTempInstructions;

  // Mapping from predicate info we used to the instructions we used it with.
  // In order to correctly ensure propagation, we must keep track of what
  // comparisons we used, so that when the values of the comparisons change, we
  // propagate the information to the places we used the comparison.
  mutable DenseMap<const Value *, SmallPtrSet<Instruction *, 2>>
      PredicateToUsers;
  // the same reasoning as PredicateToUsers.  When we skip MemoryAccesses for
  // stores, we no longer can rely solely on the def-use chains of MemorySSA.
  mutable DenseMap<const MemoryAccess *, SmallPtrSet<MemoryAccess *, 2>>
      MemoryToUsers;

  // A table storing which memorydefs/phis represent a memory state provably
  // equivalent to another memory state.
  // We could use the congruence class machinery, but the MemoryAccess's are
  // abstract memory states, so they can only ever be equivalent to each other,
  // and not to constants, etc.
  DenseMap<const MemoryAccess *, CongruenceClass *> MemoryAccessToClass;

  // We could, if we wanted, build MemoryPhiExpressions and
  // MemoryVariableExpressions, etc, and value number them the same way we value
  // number phi expressions.  For the moment, this seems like overkill.  They
  // can only exist in one of three states: they can be TOP (equal to
  // everything), Equivalent to something else, or unique.  Because we do not
  // create expressions for them, we need to simulate leader change not just
  // when they change class, but when they change state.  Note: We can do the
  // same thing for phis, and avoid having phi expressions if we wanted, We
  // should eventually unify in one direction or the other, so this is a little
  // bit of an experiment in which turns out easier to maintain.
  enum MemoryPhiState { MPS_Invalid, MPS_TOP, MPS_Equivalent, MPS_Unique };
  DenseMap<const MemoryPhi *, MemoryPhiState> MemoryPhiState;

  enum InstCycleState { ICS_Unknown, ICS_CycleFree, ICS_Cycle };
  mutable DenseMap<const Instruction *, InstCycleState> InstCycleState;
  // Expression to class mapping.
  using ExpressionClassMap = DenseMap<const Expression *, CongruenceClass *>;
  ExpressionClassMap ExpressionToClass;

  // We have a single expression that represents currently DeadExpressions.
  // For dead expressions we can prove will stay dead, we mark them with
  // DFS number zero.  However, it's possible in the case of phi nodes
  // for us to assume/prove all arguments are dead during fixpointing.
  // We use DeadExpression for that case.
  DeadExpression *SingletonDeadExpression = nullptr;

  // Which values have changed as a result of leader changes.
  SmallPtrSet<Value *, 8> LeaderChanges;

  // Reachability info.
  using BlockEdge = BasicBlockEdge;
  DenseSet<BlockEdge> ReachableEdges;
  SmallPtrSet<const BasicBlock *, 8> ReachableBlocks;

  // This is a bitvector because, on larger functions, we may have
  // thousands of touched instructions at once (entire blocks,
  // instructions with hundreds of uses, etc).  Even with optimization
  // for when we mark whole blocks as touched, when this was a
  // SmallPtrSet or DenseSet, for some functions, we spent >20% of all
  // the time in GVN just managing this list.  The bitvector, on the
  // other hand, efficiently supports test/set/clear of both
  // individual and ranges, as well as "find next element" This
  // enables us to use it as a worklist with essentially 0 cost.
  BitVector TouchedInstructions;

  DenseMap<const BasicBlock *, std::pair<unsigned, unsigned>> BlockInstRange;

#ifndef NDEBUG
  // Debugging for how many times each block and instruction got processed.
  DenseMap<const Value *, unsigned> ProcessedCount;
#endif

  // DFS info.
  // This contains a mapping from Instructions to DFS numbers.
  // The numbering starts at 1. An instruction with DFS number zero
  // means that the instruction is dead.
  DenseMap<const Value *, unsigned> InstrDFS;

  // This contains the mapping DFS numbers to instructions.
  SmallVector<Value *, 32> DFSToInstr;

  // Deletion info.
  SmallPtrSet<Instruction *, 8> InstructionsToErase;

public:
  NewGVN(Function &F, DominatorTree *DT, AssumptionCache *AC,
         TargetLibraryInfo *TLI, AliasAnalysis *AA, MemorySSA *MSSA,
         const DataLayout &DL)
      : F(F), DT(DT), TLI(TLI), AA(AA), MSSA(MSSA), DL(DL),
        PredInfo(make_unique<PredicateInfo>(F, *DT, *AC)), SQ(DL, TLI, DT, AC) {
  }
  bool runGVN();

private:
  // Expression handling.
  const Expression *createExpression(Instruction *) const;
  const Expression *createBinaryExpression(unsigned, Type *, Value *, Value *,
                                           Instruction *) const;
  PHIExpression *createPHIExpression(Instruction *, bool &HasBackEdge,
                                     bool &OriginalOpsConstant) const;
  const DeadExpression *createDeadExpression() const;
  const VariableExpression *createVariableExpression(Value *) const;
  const ConstantExpression *createConstantExpression(Constant *) const;
  const Expression *createVariableOrConstant(Value *V) const;
  const UnknownExpression *createUnknownExpression(Instruction *) const;
  const StoreExpression *createStoreExpression(StoreInst *,
                                               const MemoryAccess *) const;
  LoadExpression *createLoadExpression(Type *, Value *, LoadInst *,
                                       const MemoryAccess *) const;
  const CallExpression *createCallExpression(CallInst *,
                                             const MemoryAccess *) const;
  const AggregateValueExpression *
  createAggregateValueExpression(Instruction *) const;
  bool setBasicExpressionInfo(Instruction *, BasicExpression *) const;

  // Congruence class handling.
  CongruenceClass *createCongruenceClass(Value *Leader, const Expression *E) {
    auto *result = new CongruenceClass(NextCongruenceNum++, Leader, E);
    CongruenceClasses.emplace_back(result);
    return result;
  }

  CongruenceClass *createMemoryClass(MemoryAccess *MA) {
    auto *CC = createCongruenceClass(nullptr, nullptr);
    CC->setMemoryLeader(MA);
    return CC;
  }
  CongruenceClass *ensureLeaderOfMemoryClass(MemoryAccess *MA) {
    auto *CC = getMemoryClass(MA);
    if (CC->getMemoryLeader() != MA) {
      PassPrediction::PassPeeper(3037); // if
      CC = createMemoryClass(MA);
    }
    return CC;
  }

  CongruenceClass *createSingletonCongruenceClass(Value *Member) {
    CongruenceClass *CClass = createCongruenceClass(Member, nullptr);
    CClass->insert(Member);
    ValueToClass[Member] = CClass;
    return CClass;
  }
  void initializeCongruenceClasses(Function &F);
  const Expression *makePossiblePhiOfOps(Instruction *,
                                         SmallPtrSetImpl<Value *> &);
  void addPhiOfOps(PHINode *Op, BasicBlock *BB, Instruction *ExistingValue);

  // Value number an Instruction or MemoryPhi.
  void valueNumberMemoryPhi(MemoryPhi *);
  void valueNumberInstruction(Instruction *);

  // Symbolic evaluation.
  const Expression *checkSimplificationResults(Expression *, Instruction *,
                                               Value *) const;
  const Expression *performSymbolicEvaluation(Value *,
                                              SmallPtrSetImpl<Value *> &) const;
  const Expression *performSymbolicLoadCoercion(Type *, Value *, LoadInst *,
                                                Instruction *,
                                                MemoryAccess *) const;
  const Expression *performSymbolicLoadEvaluation(Instruction *) const;
  const Expression *performSymbolicStoreEvaluation(Instruction *) const;
  const Expression *performSymbolicCallEvaluation(Instruction *) const;
  const Expression *performSymbolicPHIEvaluation(Instruction *) const;
  const Expression *performSymbolicAggrValueEvaluation(Instruction *) const;
  const Expression *performSymbolicCmpEvaluation(Instruction *) const;
  const Expression *performSymbolicPredicateInfoEvaluation(Instruction *) const;

  // Congruence finding.
  bool someEquivalentDominates(const Instruction *, const Instruction *) const;
  Value *lookupOperandLeader(Value *) const;
  void performCongruenceFinding(Instruction *, const Expression *);
  void moveValueToNewCongruenceClass(Instruction *, const Expression *,
                                     CongruenceClass *, CongruenceClass *);
  void moveMemoryToNewCongruenceClass(Instruction *, MemoryAccess *,
                                      CongruenceClass *, CongruenceClass *);
  Value *getNextValueLeader(CongruenceClass *) const;
  const MemoryAccess *getNextMemoryLeader(CongruenceClass *) const;
  bool setMemoryClass(const MemoryAccess *From, CongruenceClass *To);
  CongruenceClass *getMemoryClass(const MemoryAccess *MA) const;
  const MemoryAccess *lookupMemoryLeader(const MemoryAccess *) const;
  bool isMemoryAccessTOP(const MemoryAccess *) const;

  // Ranking
  unsigned int getRank(const Value *) const;
  bool shouldSwapOperands(const Value *, const Value *) const;

  // Reachability handling.
  void updateReachableEdge(BasicBlock *, BasicBlock *);
  void processOutgoingEdges(TerminatorInst *, BasicBlock *);
  Value *findConditionEquivalence(Value *) const;

  // Elimination.
  struct ValueDFS;
  void convertClassToDFSOrdered(const CongruenceClass &,
                                SmallVectorImpl<ValueDFS> &,
                                DenseMap<const Value *, unsigned int> &,
                                SmallPtrSetImpl<Instruction *> &) const;
  void convertClassToLoadsAndStores(const CongruenceClass &,
                                    SmallVectorImpl<ValueDFS> &) const;

  bool eliminateInstructions(Function &);
  void replaceInstruction(Instruction *, Value *);
  void markInstructionForDeletion(Instruction *);
  void deleteInstructionsInBlock(BasicBlock *);
  Value *findPhiOfOpsLeader(const Expression *E, const BasicBlock *BB) const;

  // New instruction creation.
  void handleNewInstruction(Instruction *){};

  // Various instruction touch utilities
  template <typename Map, typename KeyType, typename Func>
  void for_each_found(Map &, const KeyType &, Func);
  template <typename Map, typename KeyType>
  void touchAndErase(Map &, const KeyType &);
  void markUsersTouched(Value *);
  void markMemoryUsersTouched(const MemoryAccess *);
  void markMemoryDefTouched(const MemoryAccess *);
  void markPredicateUsersTouched(Instruction *);
  void markValueLeaderChangeTouched(CongruenceClass *CC);
  void markMemoryLeaderChangeTouched(CongruenceClass *CC);
  void markPhiOfOpsChanged(const Expression *E);
  void addPredicateUsers(const PredicateBase *, Instruction *) const;
  void addMemoryUsers(const MemoryAccess *To, MemoryAccess *U) const;
  void addAdditionalUsers(Value *To, Value *User) const;

  // Main loop of value numbering
  void iterateTouchedInstructions();

  // Utilities.
  void cleanupTables();
  std::pair<unsigned, unsigned> assignDFSNumbers(BasicBlock *, unsigned);
  void updateProcessedCount(const Value *V);
  void verifyMemoryCongruency() const;
  void verifyIterationSettled(Function &F);
  void verifyStoreExpressions() const;
  bool singleReachablePHIPath(SmallPtrSet<const MemoryAccess *, 8> &,
                              const MemoryAccess *, const MemoryAccess *) const;
  BasicBlock *getBlockForValue(Value *V) const;
  void deleteExpression(const Expression *E) const;
  MemoryUseOrDef *getMemoryAccess(const Instruction *) const;
  MemoryAccess *getDefiningAccess(const MemoryAccess *) const;
  MemoryPhi *getMemoryAccess(const BasicBlock *) const;
  template <class T, class Range> T *getMinDFSOfRange(const Range &) const;
  unsigned InstrToDFSNum(const Value *V) const {
    assert(isa<Instruction>(V) && "This should not be used for MemoryAccesses");
    return InstrDFS.lookup(V);
  }

  unsigned InstrToDFSNum(const MemoryAccess *MA) const {
    return MemoryToDFSNum(MA);
  }
  Value *InstrFromDFSNum(unsigned DFSNum) { return DFSToInstr[DFSNum]; }
  // Given a MemoryAccess, return the relevant instruction DFS number.  Note:
  // This deliberately takes a value so it can be used with Use's, which will
  // auto-convert to Value's but not to MemoryAccess's.
  unsigned MemoryToDFSNum(const Value *MA) const {
    assert(isa<MemoryAccess>(MA) &&
           "This should not be used with instructions");
    return isa<MemoryUseOrDef>(MA)
               ? InstrToDFSNum(cast<MemoryUseOrDef>(MA)->getMemoryInst())
               : InstrDFS.lookup(MA);
  }
  bool isCycleFree(const Instruction *) const;
  bool isBackedge(BasicBlock *From, BasicBlock *To) const;
  // Debug counter info.  When verifying, we have to reset the value numbering
  // debug counter to the same state it started in to get the same results.
  std::pair<int, int> StartingVNCounter;
};
} // end anonymous namespace

template <typename T>
static bool equalsLoadStoreHelper(const T &LHS, const Expression &RHS) {
  if (!isa<LoadExpression>(RHS) && !isa<StoreExpression>(RHS)) {
    PassPrediction::PassPeeper(3044); // if
    return false;
  }
  return LHS.MemoryExpression::equals(RHS);
}

bool LoadExpression::equals(const Expression &Other) const {
  return equalsLoadStoreHelper(*this, Other);
}

bool StoreExpression::equals(const Expression &Other) const {
  if (!equalsLoadStoreHelper(*this, Other)) {
    PassPrediction::PassPeeper(3045); // if
    return false;
  }
  // Make sure that store vs store includes the value operand.
  if (const auto *S = dyn_cast<StoreExpression>(&Other)) {
    PassPrediction::PassPeeper(3046); // if
    if (getStoredValue() != S->getStoredValue()) {
      PassPrediction::PassPeeper(3047); // if
      return false;
    }
  }
  return true;
}

// Determine if the edge From->To is a backedge
bool NewGVN::isBackedge(BasicBlock *From, BasicBlock *To) const {
  if (From == To) {
    PassPrediction::PassPeeper(3048); // if
    return true;
  }
  auto *FromDTN = DT->getNode(From);
  auto *ToDTN = DT->getNode(To);
  return RPOOrdering.lookup(FromDTN) >= RPOOrdering.lookup(ToDTN);
}

#ifndef NDEBUG
static std::string getBlockName(const BasicBlock *B) {
  return DOTGraphTraits<const Function *>::getSimpleNodeLabel(B, nullptr);
}
#endif

// Get a MemoryAccess for an instruction, fake or real.
MemoryUseOrDef *NewGVN::getMemoryAccess(const Instruction *I) const {
  auto *Result = MSSA->getMemoryAccess(I);
  return Result ? Result : TempToMemory.lookup(I);
}

// Get a MemoryPhi for a basic block. These are all real.
MemoryPhi *NewGVN::getMemoryAccess(const BasicBlock *BB) const {
  return MSSA->getMemoryAccess(BB);
}

// Get the basic block from an instruction/memory value.
BasicBlock *NewGVN::getBlockForValue(Value *V) const {
  if (auto *I = dyn_cast<Instruction>(V)) {
    PassPrediction::PassPeeper(3049); // if
    auto *Parent = I->getParent();
    if (Parent) {
      PassPrediction::PassPeeper(3050); // if
      return Parent;
    }
    Parent = TempToBlock.lookup(V);
    assert(Parent && "Every fake instruction should have a block");
    return Parent;
  }

  auto *MP = dyn_cast<MemoryPhi>(V);
  assert(MP && "Should have been an instruction or a MemoryPhi");
  return MP->getBlock();
}

// Delete a definitely dead expression, so it can be reused by the expression
// allocator.  Some of these are not in creation functions, so we have to accept
// const versions.
void NewGVN::deleteExpression(const Expression *E) const {
  assert(isa<BasicExpression>(E));
  auto *BE = cast<BasicExpression>(E);
  const_cast<BasicExpression *>(BE)->deallocateOperands(ArgRecycler);
  ExpressionAllocator.Deallocate(E);
}
PHIExpression *NewGVN::createPHIExpression(Instruction *I, bool &HasBackedge,
                                           bool &OriginalOpsConstant) const {
  BasicBlock *PHIBlock = getBlockForValue(I);
  auto *PN = cast<PHINode>(I);
  auto *E =
      new (ExpressionAllocator) PHIExpression(PN->getNumOperands(), PHIBlock);

  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());

  // NewGVN assumes the operands of a PHI node are in a consistent order across
  // PHIs. LLVM doesn't seem to always guarantee this. While we need to fix
  // this in LLVM at some point we don't want GVN to find wrong congruences.
  // Therefore, here we sort uses in predecessor order.
  // We're sorting the values by pointer. In theory this might be cause of
  // non-determinism, but here we don't rely on the ordering for anything
  // significant, e.g. we don't create new instructions based on it so we're
  // fine.
  SmallVector<const Use *, 4> PHIOperands;
  for (const Use &U : PN->operands()) {
    PassPrediction::PassPeeper(3051); // for-range
    PHIOperands.push_back(&U);
  }
  std::sort(PHIOperands.begin(), PHIOperands.end(),
            [&](const Use *U1, const Use *U2) {
              return PN->getIncomingBlock(*U1) < PN->getIncomingBlock(*U2);
            });

  // Filter out unreachable phi operands.
  auto Filtered = make_filter_range(PHIOperands, [&](const Use *U) {
    if (*U == PN) {
      PassPrediction::PassPeeper(3052); // if
      return false;
    }
    if (!ReachableEdges.count({PN->getIncomingBlock(*U), PHIBlock})) {
      PassPrediction::PassPeeper(3053); // if
      return false;
    }
    // Things in TOPClass are equivalent to everything.
    if (ValueToClass.lookup(*U) == TOPClass) {
      PassPrediction::PassPeeper(3054); // if
      return false;
    }
    return lookupOperandLeader(*U) != PN;
  });
  std::transform(Filtered.begin(), Filtered.end(), op_inserter(E),
                 [&](const Use *U) -> Value * {
                   auto *BB = PN->getIncomingBlock(*U);
                   HasBackedge = HasBackedge || isBackedge(BB, PHIBlock);
                   OriginalOpsConstant =
                       OriginalOpsConstant && isa<Constant>(*U);
                   return lookupOperandLeader(*U);
                 });
  return E;
}

// Set basic expression info (Arguments, type, opcode) for Expression
// E from Instruction I in block B.
bool NewGVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E) const {
  bool AllConstant = true;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    PassPrediction::PassPeeper(3055); // if
    E->setType(GEP->getSourceElementType());
  } else {
    PassPrediction::PassPeeper(3056); // else
    E->setType(I->getType());
  }
  E->setOpcode(I->getOpcode());
  E->allocateOperands(ArgRecycler, ExpressionAllocator);

  // Transform the operand array into an operand leader array, and keep track of
  // whether all members are constant.
  std::transform(I->op_begin(), I->op_end(), op_inserter(E), [&](Value *O) {
    auto Operand = lookupOperandLeader(O);
    AllConstant = AllConstant && isa<Constant>(Operand);
    return Operand;
  });

  return AllConstant;
}

const Expression *NewGVN::createBinaryExpression(unsigned Opcode, Type *T,
                                                 Value *Arg1, Value *Arg2,
                                                 Instruction *I) const {
  auto *E = new (ExpressionAllocator) BasicExpression(2);

  E->setType(T);
  E->setOpcode(Opcode);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  if (Instruction::isCommutative(Opcode)) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    PassPrediction::PassPeeper(3057); // if
    if (shouldSwapOperands(Arg1, Arg2)) {
      PassPrediction::PassPeeper(3058); // if
      std::swap(Arg1, Arg2);
    }
  }
  E->op_push_back(lookupOperandLeader(Arg1));
  E->op_push_back(lookupOperandLeader(Arg2));

  Value *V = SimplifyBinOp(Opcode, E->getOperand(0), E->getOperand(1), SQ);
  if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V)) {
    PassPrediction::PassPeeper(3059); // if
    return SimplifiedE;
  }
  return E;
}

// Take a Value returned by simplification of Expression E/Instruction
// I, and see if it resulted in a simpler expression. If so, return
// that expression.
// TODO: Once finished, this should not take an Instruction, we only
// use it for printing.
const Expression *NewGVN::checkSimplificationResults(Expression *E,
                                                     Instruction *I,
                                                     Value *V) const {
  if (!V) {
    PassPrediction::PassPeeper(3060); // if
    return nullptr;
  }
  if (auto *C = dyn_cast<Constant>(V)) {
    PassPrediction::PassPeeper(3061); // if
    if (I) {
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " constant " << *C << "\n");
    }
    NumGVNOpsSimplified++;
    assert(isa<BasicExpression>(E) &&
           "We should always have had a basic expression here");
    deleteExpression(E);
    return createConstantExpression(C);
  } else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    PassPrediction::PassPeeper(3062); // if
    if (I) {
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " variable " << *V << "\n");
    }
    deleteExpression(E);
    return createVariableExpression(V);
  }

  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && CC->getDefiningExpr()) {
    // If we simplified to something else, we need to communicate
    // that we're users of the value we simplified to.
    PassPrediction::PassPeeper(3063); // if
    if (I != V) {
      // Don't add temporary instructions to the user lists.
      PassPrediction::PassPeeper(3064); // if
      if (!AllTempInstructions.count(I)) {
        PassPrediction::PassPeeper(3065); // if
        addAdditionalUsers(V, I);
      }
    }

    if (I) {
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " expression " << *CC->getDefiningExpr() << "\n");
    }
    NumGVNOpsSimplified++;
    deleteExpression(E);
    return CC->getDefiningExpr();
  }
  return nullptr;
}

const Expression *NewGVN::createExpression(Instruction *I) const {
  auto *E = new (ExpressionAllocator) BasicExpression(I->getNumOperands());

  bool AllConstant = setBasicExpressionInfo(I, E);

  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (shouldSwapOperands(E->getOperand(0), E->getOperand(1))) {
      PassPrediction::PassPeeper(3066); // if
      E->swapOperands(0, 1);
    }
  }

  // Perform simplificaiton
  // TODO: Right now we only check to see if we get a constant result.
  // We may get a less than constant, but still better, result for
  // some operations.
  // IE
  //  add 0, x -> x
  //  and x, x -> x
  // We should handle this by simply rewriting the expression.
  if (auto *CI = dyn_cast<CmpInst>(I)) {
    // Sort the operand value numbers so x<y and y>x get the same value
    // number.
    PassPrediction::PassPeeper(3067); // if
    CmpInst::Predicate Predicate = CI->getPredicate();
    if (shouldSwapOperands(E->getOperand(0), E->getOperand(1))) {
      PassPrediction::PassPeeper(3068); // if
      E->swapOperands(0, 1);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    // TODO: 25% of our time is spent in SimplifyCmpInst with pointer operands
    assert(I->getOperand(0)->getType() == I->getOperand(1)->getType() &&
           "Wrong types on cmp instruction");
    assert((E->getOperand(0)->getType() == I->getOperand(0)->getType() &&
            E->getOperand(1)->getType() == I->getOperand(1)->getType()));
    Value *V =
        SimplifyCmpInst(Predicate, E->getOperand(0), E->getOperand(1), SQ);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V)) {
      PassPrediction::PassPeeper(3069); // if
      return SimplifiedE;
    }
  } else if (isa<SelectInst>(I)) {
    PassPrediction::PassPeeper(3070); // if
    if (isa<Constant>(E->getOperand(0)) ||
        E->getOperand(0) == E->getOperand(1)) {
      assert(E->getOperand(1)->getType() == I->getOperand(1)->getType() &&
             E->getOperand(2)->getType() == I->getOperand(2)->getType());
      Value *V = SimplifySelectInst(E->getOperand(0), E->getOperand(1),
                                    E->getOperand(2), SQ);
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V)) {
        PassPrediction::PassPeeper(3071); // if
        return SimplifiedE;
      }
    }
  } else if (I->isBinaryOp()) {
    PassPrediction::PassPeeper(3072); // if
    Value *V =
        SimplifyBinOp(E->getOpcode(), E->getOperand(0), E->getOperand(1), SQ);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V)) {
      PassPrediction::PassPeeper(3073); // if
      return SimplifiedE;
    }
  } else if (auto *BI = dyn_cast<BitCastInst>(I)) {
    PassPrediction::PassPeeper(3074); // if
    Value *V =
        SimplifyCastInst(BI->getOpcode(), BI->getOperand(0), BI->getType(), SQ);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V)) {
      PassPrediction::PassPeeper(3075); // if
      return SimplifiedE;
    }
  } else if (isa<GetElementPtrInst>(I)) {
    PassPrediction::PassPeeper(3076); // if
    Value *V = SimplifyGEPInst(
        E->getType(), ArrayRef<Value *>(E->op_begin(), E->op_end()), SQ);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V)) {
      PassPrediction::PassPeeper(3077); // if
      return SimplifiedE;
    }
  } else if (AllConstant) {
    // We don't bother trying to simplify unless all of the operands
    // were constant.
    // TODO: There are a lot of Simplify*'s we could call here, if we
    // wanted to.  The original motivating case for this code was a
    // zext i1 false to i8, which we don't have an interface to
    // simplify (IE there is no SimplifyZExt).

    PassPrediction::PassPeeper(3078); // if
    SmallVector<Constant *, 8> C;
    for (Value *Arg : E->operands()) {
      PassPrediction::PassPeeper(3079); // for-range
      C.emplace_back(cast<Constant>(Arg));
    }

    if (Value *V = ConstantFoldInstOperands(I, C, DL, TLI)) {
      PassPrediction::PassPeeper(3080); // if
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V)) {
        PassPrediction::PassPeeper(3081); // if
        return SimplifiedE;
      }
    }
  }
  return E;
}

const AggregateValueExpression *
NewGVN::createAggregateValueExpression(Instruction *I) const {
  if (auto *II = dyn_cast<InsertValueInst>(I)) {
    PassPrediction::PassPeeper(3082); // if
    auto *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), II->getNumIndices());
    setBasicExpressionInfo(I, E);
    E->allocateIntOperands(ExpressionAllocator);
    std::copy(II->idx_begin(), II->idx_end(), int_op_inserter(E));
    return E;
  } else if (auto *EI = dyn_cast<ExtractValueInst>(I)) {
    PassPrediction::PassPeeper(3083); // if
    auto *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), EI->getNumIndices());
    setBasicExpressionInfo(EI, E);
    E->allocateIntOperands(ExpressionAllocator);
    std::copy(EI->idx_begin(), EI->idx_end(), int_op_inserter(E));
    return E;
  }
  llvm_unreachable("Unhandled type of aggregate value operation");
}

const DeadExpression *NewGVN::createDeadExpression() const {
  // DeadExpression has no arguments and all DeadExpression's are the same,
  // so we only need one of them.
  return SingletonDeadExpression;
}

const VariableExpression *NewGVN::createVariableExpression(Value *V) const {
  auto *E = new (ExpressionAllocator) VariableExpression(V);
  E->setOpcode(V->getValueID());
  return E;
}

const Expression *NewGVN::createVariableOrConstant(Value *V) const {
  if (auto *C = dyn_cast<Constant>(V)) {
    PassPrediction::PassPeeper(3084); // if
    return createConstantExpression(C);
  }
  return createVariableExpression(V);
}

const ConstantExpression *NewGVN::createConstantExpression(Constant *C) const {
  auto *E = new (ExpressionAllocator) ConstantExpression(C);
  E->setOpcode(C->getValueID());
  return E;
}

const UnknownExpression *NewGVN::createUnknownExpression(Instruction *I) const {
  auto *E = new (ExpressionAllocator) UnknownExpression(I);
  E->setOpcode(I->getOpcode());
  return E;
}

const CallExpression *
NewGVN::createCallExpression(CallInst *CI, const MemoryAccess *MA) const {
  // FIXME: Add operand bundles for calls.
  auto *E =
      new (ExpressionAllocator) CallExpression(CI->getNumOperands(), CI, MA);
  setBasicExpressionInfo(CI, E);
  return E;
}

// Return true if some equivalent of instruction Inst dominates instruction U.
bool NewGVN::someEquivalentDominates(const Instruction *Inst,
                                     const Instruction *U) const {
  auto *CC = ValueToClass.lookup(Inst);
  // This must be an instruction because we are only called from phi nodes
  // in the case that the value it needs to check against is an instruction.

  // The most likely candiates for dominance are the leader and the next leader.
  // The leader or nextleader will dominate in all cases where there is an
  // equivalent that is higher up in the dom tree.
  // We can't *only* check them, however, because the
  // dominator tree could have an infinite number of non-dominating siblings
  // with instructions that are in the right congruence class.
  //       A
  // B C D E F G
  // |
  // H
  // Instruction U could be in H,  with equivalents in every other sibling.
  // Depending on the rpo order picked, the leader could be the equivalent in
  // any of these siblings.
  if (!CC) {
    PassPrediction::PassPeeper(3085); // if
    return false;
  }
  if (DT->dominates(cast<Instruction>(CC->getLeader()), U)) {
    PassPrediction::PassPeeper(3086); // if
    return true;
  }
  if (CC->getNextLeader().first &&
      DT->dominates(cast<Instruction>(CC->getNextLeader().first), U)) {
    PassPrediction::PassPeeper(3087); // if
    return true;
  }
  return llvm::any_of(*CC, [&](const Value *Member) {
    return Member != CC->getLeader() &&
           DT->dominates(cast<Instruction>(Member), U);
  });
}

// See if we have a congruence class and leader for this operand, and if so,
// return it. Otherwise, return the operand itself.
Value *NewGVN::lookupOperandLeader(Value *V) const {
  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC) {
    // Everything in TOP is represented by undef, as it can be any value.
    // We do have to make sure we get the type right though, so we can't set the
    // RepLeader to undef.
    PassPrediction::PassPeeper(3088); // if
    if (CC == TOPClass) {
      PassPrediction::PassPeeper(3089); // if
      return UndefValue::get(V->getType());
    }
    return CC->getStoredValue() ? CC->getStoredValue() : CC->getLeader();
  }

  return V;
}

const MemoryAccess *NewGVN::lookupMemoryLeader(const MemoryAccess *MA) const {
  auto *CC = getMemoryClass(MA);
  assert(CC->getMemoryLeader() &&
         "Every MemoryAccess should be mapped to a congruence class with a "
         "representative memory access");
  return CC->getMemoryLeader();
}

// Return true if the MemoryAccess is really equivalent to everything. This is
// equivalent to the lattice value "TOP" in most lattices.  This is the initial
// state of all MemoryAccesses.
bool NewGVN::isMemoryAccessTOP(const MemoryAccess *MA) const {
  return getMemoryClass(MA) == TOPClass;
}

LoadExpression *NewGVN::createLoadExpression(Type *LoadType, Value *PointerOp,
                                             LoadInst *LI,
                                             const MemoryAccess *MA) const {
  auto *E =
      new (ExpressionAllocator) LoadExpression(1, LI, lookupMemoryLeader(MA));
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(LoadType);

  // Give store and loads same opcode so they value number together.
  E->setOpcode(0);
  E->op_push_back(PointerOp);
  if (LI) {
    PassPrediction::PassPeeper(3090); // if
    E->setAlignment(LI->getAlignment());
  }

  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load).
  return E;
}

const StoreExpression *
NewGVN::createStoreExpression(StoreInst *SI, const MemoryAccess *MA) const {
  auto *StoredValueLeader = lookupOperandLeader(SI->getValueOperand());
  auto *E = new (ExpressionAllocator)
      StoreExpression(SI->getNumOperands(), SI, StoredValueLeader, MA);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(SI->getValueOperand()->getType());

  // Give store and loads same opcode so they value number together.
  E->setOpcode(0);
  E->op_push_back(lookupOperandLeader(SI->getPointerOperand()));

  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load).
  return E;
}

const Expression *NewGVN::performSymbolicStoreEvaluation(Instruction *I) const {
  // Unlike loads, we never try to eliminate stores, so we do not check if they
  // are simple and avoid value numbering them.
  auto *SI = cast<StoreInst>(I);
  auto *StoreAccess = getMemoryAccess(SI);
  // Get the expression, if any, for the RHS of the MemoryDef.
  const MemoryAccess *StoreRHS = StoreAccess->getDefiningAccess();
  if (EnableStoreRefinement) {
    PassPrediction::PassPeeper(3091); // if
    StoreRHS = MSSAWalker->getClobberingMemoryAccess(StoreAccess);
  }
  // If we bypassed the use-def chains, make sure we add a use.
  if (StoreRHS != StoreAccess->getDefiningAccess()) {
    PassPrediction::PassPeeper(3092); // if
    addMemoryUsers(StoreRHS, StoreAccess);
  }
  StoreRHS = lookupMemoryLeader(StoreRHS);
  // If we are defined by ourselves, use the live on entry def.
  if (StoreRHS == StoreAccess) {
    PassPrediction::PassPeeper(3093); // if
    StoreRHS = MSSA->getLiveOnEntryDef();
  }

  if (SI->isSimple()) {
    // See if we are defined by a previous store expression, it already has a
    // value, and it's the same value as our current store. FIXME: Right now, we
    // only do this for simple stores, we should expand to cover memcpys, etc.
    PassPrediction::PassPeeper(3094); // if
    const auto *LastStore = createStoreExpression(SI, StoreRHS);
    const auto *LastCC = ExpressionToClass.lookup(LastStore);
    // We really want to check whether the expression we matched was a store. No
    // easy way to do that. However, we can check that the class we found has a
    // store, which, assuming the value numbering state is not corrupt, is
    // sufficient, because we must also be equivalent to that store's expression
    // for it to be in the same class as the load.
    if (LastCC && LastCC->getStoredValue() == LastStore->getStoredValue()) {
      PassPrediction::PassPeeper(3095); // if
      return LastStore;
    }
    // Also check if our value operand is defined by a load of the same memory
    // location, and the memory state is the same as it was then (otherwise, it
    // could have been overwritten later. See test32 in
    // transforms/DeadStoreElimination/simple.ll).
    if (auto *LI = dyn_cast<LoadInst>(LastStore->getStoredValue())) {
      PassPrediction::PassPeeper(3096); // if
      if ((lookupOperandLeader(LI->getPointerOperand()) ==
           LastStore->getOperand(0)) &&
          (lookupMemoryLeader(getMemoryAccess(LI)->getDefiningAccess()) ==
           StoreRHS)) {
        PassPrediction::PassPeeper(3097); // if
        return LastStore;
      }
    }
    deleteExpression(LastStore);
  }

  // If the store is not equivalent to anything, value number it as a store that
  // produces a unique memory state (instead of using it's MemoryUse, we use
  // it's MemoryDef).
  return createStoreExpression(SI, StoreAccess);
}

// See if we can extract the value of a loaded pointer from a load, a store, or
// a memory instruction.
const Expression *
NewGVN::performSymbolicLoadCoercion(Type *LoadType, Value *LoadPtr,
                                    LoadInst *LI, Instruction *DepInst,
                                    MemoryAccess *DefiningAccess) const {
  assert((!LI || LI->isSimple()) && "Not a simple load");
  if (auto *DepSI = dyn_cast<StoreInst>(DepInst)) {
    // Can't forward from non-atomic to atomic without violating memory model.
    // Also don't need to coerce if they are the same type, we will just
    // propogate..
    PassPrediction::PassPeeper(3098); // if
    if (LI->isAtomic() > DepSI->isAtomic() ||
        LoadType == DepSI->getValueOperand()->getType()) {
      PassPrediction::PassPeeper(3099); // if
      return nullptr;
    }
    int Offset = analyzeLoadFromClobberingStore(LoadType, LoadPtr, DepSI, DL);
    if (Offset >= 0) {
      PassPrediction::PassPeeper(3100); // if
      if (auto *C = dyn_cast<Constant>(
              lookupOperandLeader(DepSI->getValueOperand()))) {
        DEBUG(dbgs() << "Coercing load from store " << *DepSI << " to constant "
                     << *C << "\n");
        return createConstantExpression(
            getConstantStoreValueForLoad(C, Offset, LoadType, DL));
      }
    }

  } else if (LoadInst *DepLI = dyn_cast<LoadInst>(DepInst)) {
    // Can't forward from non-atomic to atomic without violating memory model.
    PassPrediction::PassPeeper(3101); // if
    if (LI->isAtomic() > DepLI->isAtomic()) {
      PassPrediction::PassPeeper(3102); // if
      return nullptr;
    }
    int Offset = analyzeLoadFromClobberingLoad(LoadType, LoadPtr, DepLI, DL);
    if (Offset >= 0) {
      // We can coerce a constant load into a load
      PassPrediction::PassPeeper(3103); // if
      if (auto *C = dyn_cast<Constant>(lookupOperandLeader(DepLI))) {
        PassPrediction::PassPeeper(3104); // if
        if (auto *PossibleConstant =
                getConstantLoadValueForLoad(C, Offset, LoadType, DL)) {
          DEBUG(dbgs() << "Coercing load from load " << *LI << " to constant "
                       << *PossibleConstant << "\n");
          return createConstantExpression(PossibleConstant);
        }
      }
    }

  } else if (MemIntrinsic *DepMI = dyn_cast<MemIntrinsic>(DepInst)) {
    PassPrediction::PassPeeper(3105); // if
    int Offset = analyzeLoadFromClobberingMemInst(LoadType, LoadPtr, DepMI, DL);
    if (Offset >= 0) {
      PassPrediction::PassPeeper(3106); // if
      if (auto *PossibleConstant =
              getConstantMemInstValueForLoad(DepMI, Offset, LoadType, DL)) {
        DEBUG(dbgs() << "Coercing load from meminst " << *DepMI
                     << " to constant " << *PossibleConstant << "\n");
        return createConstantExpression(PossibleConstant);
      }
    }
  }

  // All of the below are only true if the loaded pointer is produced
  // by the dependent instruction.
  if (LoadPtr != lookupOperandLeader(DepInst) &&
      !AA->isMustAlias(LoadPtr, DepInst)) {
    PassPrediction::PassPeeper(3107); // if
    return nullptr;
  }
  // If this load really doesn't depend on anything, then we must be loading an
  // undef value.  This can happen when loading for a fresh allocation with no
  // intervening stores, for example.  Note that this is only true in the case
  // that the result of the allocation is pointer equal to the load ptr.
  if (isa<AllocaInst>(DepInst) || isMallocLikeFn(DepInst, TLI)) {
    PassPrediction::PassPeeper(3108); // if
    return createConstantExpression(UndefValue::get(LoadType));
  }
  // If this load occurs either right after a lifetime begin,
  // then the loaded value is undefined.
  else if (auto *II = dyn_cast<IntrinsicInst>(DepInst)) {
    PassPrediction::PassPeeper(3109); // if
    if (II->getIntrinsicID() == Intrinsic::lifetime_start) {
      PassPrediction::PassPeeper(3110); // if
      return createConstantExpression(UndefValue::get(LoadType));
    }
  }
  // If this load follows a calloc (which zero initializes memory),
  // then the loaded value is zero
  else if (isCallocLikeFn(DepInst, TLI)) {
    PassPrediction::PassPeeper(3111); // if
    return createConstantExpression(Constant::getNullValue(LoadType));
  }

  return nullptr;
}

const Expression *NewGVN::performSymbolicLoadEvaluation(Instruction *I) const {
  auto *LI = cast<LoadInst>(I);

  // We can eliminate in favor of non-simple loads, but we won't be able to
  // eliminate the loads themselves.
  if (!LI->isSimple()) {
    PassPrediction::PassPeeper(3112); // if
    return nullptr;
  }

  Value *LoadAddressLeader = lookupOperandLeader(LI->getPointerOperand());
  // Load of undef is undef.
  if (isa<UndefValue>(LoadAddressLeader)) {
    PassPrediction::PassPeeper(3113); // if
    return createConstantExpression(UndefValue::get(LI->getType()));
  }
  MemoryAccess *OriginalAccess = getMemoryAccess(I);
  MemoryAccess *DefiningAccess =
      MSSAWalker->getClobberingMemoryAccess(OriginalAccess);

  if (!MSSA->isLiveOnEntryDef(DefiningAccess)) {
    PassPrediction::PassPeeper(3114); // if
    if (auto *MD = dyn_cast<MemoryDef>(DefiningAccess)) {
      PassPrediction::PassPeeper(3115); // if
      Instruction *DefiningInst = MD->getMemoryInst();
      // If the defining instruction is not reachable, replace with undef.
      if (!ReachableBlocks.count(DefiningInst->getParent())) {
        PassPrediction::PassPeeper(3116); // if
        return createConstantExpression(UndefValue::get(LI->getType()));
      }
      // This will handle stores and memory insts.  We only do if it the
      // defining access has a different type, or it is a pointer produced by
      // certain memory operations that cause the memory to have a fixed value
      // (IE things like calloc).
      if (const auto *CoercionResult =
              performSymbolicLoadCoercion(LI->getType(), LoadAddressLeader, LI,
                                          DefiningInst, DefiningAccess)) {
        PassPrediction::PassPeeper(3117); // if
        return CoercionResult;
      }
    }
  }

  const Expression *E = createLoadExpression(LI->getType(), LoadAddressLeader,
                                             LI, DefiningAccess);
  return E;
}

const Expression *
NewGVN::performSymbolicPredicateInfoEvaluation(Instruction *I) const {
  auto *PI = PredInfo->getPredicateInfoFor(I);
  if (!PI) {
    PassPrediction::PassPeeper(3118); // if
    return nullptr;
  }

  DEBUG(dbgs() << "Found predicate info from instruction !\n");

  auto *PWC = dyn_cast<PredicateWithCondition>(PI);
  if (!PWC) {
    PassPrediction::PassPeeper(3119); // if
    return nullptr;
  }

  auto *CopyOf = I->getOperand(0);
  auto *Cond = PWC->Condition;

  // If this a copy of the condition, it must be either true or false depending
  // on the predicate info type and edge
  if (CopyOf == Cond) {
    // We should not need to add predicate users because the predicate info is
    // already a use of this operand.
    PassPrediction::PassPeeper(3120); // if
    if (isa<PredicateAssume>(PI)) {
      PassPrediction::PassPeeper(3121); // if
      return createConstantExpression(ConstantInt::getTrue(Cond->getType()));
    }
    if (auto *PBranch = dyn_cast<PredicateBranch>(PI)) {
      PassPrediction::PassPeeper(3122); // if
      if (PBranch->TrueEdge) {
        PassPrediction::PassPeeper(3123); // if
        return createConstantExpression(ConstantInt::getTrue(Cond->getType()));
      }
      return createConstantExpression(ConstantInt::getFalse(Cond->getType()));
    }
    if (auto *PSwitch = dyn_cast<PredicateSwitch>(PI)) {
      PassPrediction::PassPeeper(3124); // if
      return createConstantExpression(cast<Constant>(PSwitch->CaseValue));
    }
  }

  // Not a copy of the condition, so see what the predicates tell us about this
  // value.  First, though, we check to make sure the value is actually a copy
  // of one of the condition operands. It's possible, in certain cases, for it
  // to be a copy of a predicateinfo copy. In particular, if two branch
  // operations use the same condition, and one branch dominates the other, we
  // will end up with a copy of a copy.  This is currently a small deficiency in
  // predicateinfo.  What will end up happening here is that we will value
  // number both copies the same anyway.

  // Everything below relies on the condition being a comparison.
  auto *Cmp = dyn_cast<CmpInst>(Cond);
  if (!Cmp) {
    PassPrediction::PassPeeper(3125); // if
    return nullptr;
  }

  if (CopyOf != Cmp->getOperand(0) && CopyOf != Cmp->getOperand(1)) {
    DEBUG(dbgs() << "Copy is not of any condition operands!\n");
    return nullptr;
  }
  Value *FirstOp = lookupOperandLeader(Cmp->getOperand(0));
  Value *SecondOp = lookupOperandLeader(Cmp->getOperand(1));
  bool SwappedOps = false;
  // Sort the ops
  if (shouldSwapOperands(FirstOp, SecondOp)) {
    PassPrediction::PassPeeper(3126); // if
    std::swap(FirstOp, SecondOp);
    SwappedOps = true;
  }
  CmpInst::Predicate Predicate =
      SwappedOps ? Cmp->getSwappedPredicate() : Cmp->getPredicate();

  if (isa<PredicateAssume>(PI)) {
    // If the comparison is true when the operands are equal, then we know the
    // operands are equal, because assumes must always be true.
    PassPrediction::PassPeeper(3127); // if
    if (CmpInst::isTrueWhenEqual(Predicate)) {
      PassPrediction::PassPeeper(3128); // if
      addPredicateUsers(PI, I);
      addAdditionalUsers(Cmp->getOperand(0), I);
      return createVariableOrConstant(FirstOp);
    }
  }
  if (const auto *PBranch = dyn_cast<PredicateBranch>(PI)) {
    // If we are *not* a copy of the comparison, we may equal to the other
    // operand when the predicate implies something about equality of
    // operations.  In particular, if the comparison is true/false when the
    // operands are equal, and we are on the right edge, we know this operation
    // is equal to something.
    PassPrediction::PassPeeper(3129); // if
    if ((PBranch->TrueEdge && Predicate == CmpInst::ICMP_EQ) ||
        (!PBranch->TrueEdge && Predicate == CmpInst::ICMP_NE)) {
      PassPrediction::PassPeeper(3130); // if
      addPredicateUsers(PI, I);
      addAdditionalUsers(Cmp->getOperand(0), I);
      return createVariableOrConstant(FirstOp);
    }
    // Handle the special case of floating point.
    if (((PBranch->TrueEdge && Predicate == CmpInst::FCMP_OEQ) ||
         (!PBranch->TrueEdge && Predicate == CmpInst::FCMP_UNE)) &&
        isa<ConstantFP>(FirstOp) && !cast<ConstantFP>(FirstOp)->isZero()) {
      PassPrediction::PassPeeper(3131); // if
      addPredicateUsers(PI, I);
      addAdditionalUsers(Cmp->getOperand(0), I);
      return createConstantExpression(cast<Constant>(FirstOp));
    }
  }
  return nullptr;
}

// Evaluate read only and pure calls, and create an expression result.
const Expression *NewGVN::performSymbolicCallEvaluation(Instruction *I) const {
  auto *CI = cast<CallInst>(I);
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    // Instrinsics with the returned attribute are copies of arguments.
    PassPrediction::PassPeeper(3132); // if
    if (auto *ReturnedValue = II->getReturnedArgOperand()) {
      PassPrediction::PassPeeper(3133); // if
      if (II->getIntrinsicID() == Intrinsic::ssa_copy) {
        PassPrediction::PassPeeper(3134); // if
        if (const auto *Result = performSymbolicPredicateInfoEvaluation(I)) {
          PassPrediction::PassPeeper(3135); // if
          return Result;
        }
      }
      return createVariableOrConstant(ReturnedValue);
    }
  }
  if (AA->doesNotAccessMemory(CI)) {
    PassPrediction::PassPeeper(3136); // if
    return createCallExpression(CI, TOPClass->getMemoryLeader());
  } else if (AA->onlyReadsMemory(CI)) {
    PassPrediction::PassPeeper(3137); // if
    MemoryAccess *DefiningAccess = MSSAWalker->getClobberingMemoryAccess(CI);
    return createCallExpression(CI, DefiningAccess);
  }
  return nullptr;
}

// Retrieve the memory class for a given MemoryAccess.
CongruenceClass *NewGVN::getMemoryClass(const MemoryAccess *MA) const {

  auto *Result = MemoryAccessToClass.lookup(MA);
  assert(Result && "Should have found memory class");
  return Result;
}

// Update the MemoryAccess equivalence table to say that From is equal to To,
// and return true if this is different from what already existed in the table.
bool NewGVN::setMemoryClass(const MemoryAccess *From,
                            CongruenceClass *NewClass) {
  assert(NewClass &&
         "Every MemoryAccess should be getting mapped to a non-null class");
  DEBUG(dbgs() << "Setting " << *From);
  DEBUG(dbgs() << " equivalent to congruence class ");
  DEBUG(dbgs() << NewClass->getID() << " with current MemoryAccess leader ");
  DEBUG(dbgs() << *NewClass->getMemoryLeader() << "\n");

  auto LookupResult = MemoryAccessToClass.find(From);
  bool Changed = false;
  // If it's already in the table, see if the value changed.
  if (LookupResult != MemoryAccessToClass.end()) {
    PassPrediction::PassPeeper(3138); // if
    auto *OldClass = LookupResult->second;
    if (OldClass != NewClass) {
      // If this is a phi, we have to handle memory member updates.
      PassPrediction::PassPeeper(3139); // if
      if (auto *MP = dyn_cast<MemoryPhi>(From)) {
        PassPrediction::PassPeeper(3140); // if
        OldClass->memory_erase(MP);
        NewClass->memory_insert(MP);
        // This may have killed the class if it had no non-memory members
        if (OldClass->getMemoryLeader() == From) {
          PassPrediction::PassPeeper(3141); // if
          if (OldClass->definesNoMemory()) {
            PassPrediction::PassPeeper(3142); // if
            OldClass->setMemoryLeader(nullptr);
          } else {
            PassPrediction::PassPeeper(3143); // else
            OldClass->setMemoryLeader(getNextMemoryLeader(OldClass));
            DEBUG(dbgs() << "Memory class leader change for class "
                         << OldClass->getID() << " to "
                         << *OldClass->getMemoryLeader()
                         << " due to removal of a memory member " << *From
                         << "\n");
            markMemoryLeaderChangeTouched(OldClass);
          }
        }
      }
      // It wasn't equivalent before, and now it is.
      LookupResult->second = NewClass;
      Changed = true;
    }
  }

  return Changed;
}

// Determine if a instruction is cycle-free.  That means the values in the
// instruction don't depend on any expressions that can change value as a result
// of the instruction.  For example, a non-cycle free instruction would be v =
// phi(0, v+1).
bool NewGVN::isCycleFree(const Instruction *I) const {
  // In order to compute cycle-freeness, we do SCC finding on the instruction,
  // and see what kind of SCC it ends up in.  If it is a singleton, it is
  // cycle-free.  If it is not in a singleton, it is only cycle free if the
  // other members are all phi nodes (as they do not compute anything, they are
  // copies).
  auto ICS = InstCycleState.lookup(I);
  if (ICS == ICS_Unknown) {
    PassPrediction::PassPeeper(3144); // if
    SCCFinder.Start(I);
    auto &SCC = SCCFinder.getComponentFor(I);
    // It's cycle free if it's size 1 or or the SCC is *only* phi nodes.
    if (SCC.size() == 1) {
      PassPrediction::PassPeeper(3145); // if
      InstCycleState.insert({I, ICS_CycleFree});
    } else {
      PassPrediction::PassPeeper(3146); // else
      bool AllPhis =
          llvm::all_of(SCC, [](const Value *V) { return isa<PHINode>(V); });
      ICS = AllPhis ? ICS_CycleFree : ICS_Cycle;
      for (auto *Member : SCC) {
        PassPrediction::PassPeeper(3147); // for-range
        if (auto *MemberPhi = dyn_cast<PHINode>(Member)) {
          PassPrediction::PassPeeper(3148); // if
          InstCycleState.insert({MemberPhi, ICS});
        }
      }
    }
  }
  if (ICS == ICS_Cycle) {
    PassPrediction::PassPeeper(3149); // if
    return false;
  }
  return true;
}

// Evaluate PHI nodes symbolically, and create an expression result.
const Expression *NewGVN::performSymbolicPHIEvaluation(Instruction *I) const {
  // True if one of the incoming phi edges is a backedge.
  bool HasBackedge = false;
  // All constant tracks the state of whether all the *original* phi operands
  // This is really shorthand for "this phi cannot cycle due to forward
  // change in value of the phi is guaranteed not to later change the value of
  // the phi. IE it can't be v = phi(undef, v+1)
  bool AllConstant = true;
  auto *E =
      cast<PHIExpression>(createPHIExpression(I, HasBackedge, AllConstant));
  // We match the semantics of SimplifyPhiNode from InstructionSimplify here.
  // See if all arguments are the same.
  // We track if any were undef because they need special handling.
  bool HasUndef = false;
  auto Filtered = make_filter_range(E->operands(), [&](Value *Arg) {
    if (isa<UndefValue>(Arg)) {
      PassPrediction::PassPeeper(3150); // if
      HasUndef = true;
      return false;
    }
    return true;
  });
  // If we are left with no operands, it's dead.
  if (Filtered.begin() == Filtered.end()) {
    // If it has undef at this point, it means there are no-non-undef arguments,
    // and thus, the value of the phi node must be undef.
    PassPrediction::PassPeeper(3151); // if
    if (HasUndef) {
      DEBUG(dbgs() << "PHI Node " << *I
                   << " has no non-undef arguments, valuing it as undef\n");
      return createConstantExpression(UndefValue::get(I->getType()));
    }

    DEBUG(dbgs() << "No arguments of PHI node " << *I << " are live\n");
    deleteExpression(E);
    return createDeadExpression();
  }
  unsigned NumOps = 0;
  Value *AllSameValue = *(Filtered.begin());
  ++Filtered.begin();
  // Can't use std::equal here, sadly, because filter.begin moves.
  if (llvm::all_of(Filtered, [&](Value *Arg) {
        ++NumOps;
        return Arg == AllSameValue;
      })) {
    // In LLVM's non-standard representation of phi nodes, it's possible to have
    // phi nodes with cycles (IE dependent on other phis that are .... dependent
    // on the original phi node), especially in weird CFG's where some arguments
    // are unreachable, or uninitialized along certain paths.  This can cause
    // infinite loops during evaluation. We work around this by not trying to
    // really evaluate them independently, but instead using a variable
    // expression to say if one is equivalent to the other.
    // We also special case undef, so that if we have an undef, we can't use the
    // common value unless it dominates the phi block.
    PassPrediction::PassPeeper(3152); // if
    if (HasUndef) {
      // If we have undef and at least one other value, this is really a
      // multivalued phi, and we need to know if it's cycle free in order to
      // evaluate whether we can ignore the undef.  The other parts of this are
      // just shortcuts.  If there is no backedge, or all operands are
      // constants, or all operands are ignored but the undef, it also must be
      // cycle free.
      PassPrediction::PassPeeper(3153); // if
      if (!AllConstant && HasBackedge && NumOps > 0 &&
          !isa<UndefValue>(AllSameValue) && !isCycleFree(I)) {
        PassPrediction::PassPeeper(3154); // if
        return E;
      }

      // Only have to check for instructions
      if (auto *AllSameInst = dyn_cast<Instruction>(AllSameValue)) {
        PassPrediction::PassPeeper(3155); // if
        if (!someEquivalentDominates(AllSameInst, I)) {
          PassPrediction::PassPeeper(3156); // if
          return E;
        }
      }
    }
    // Can't simplify to something that comes later in the iteration.
    // Otherwise, when and if it changes congruence class, we will never catch
    // up. We will always be a class behind it.
    if (isa<Instruction>(AllSameValue) &&
        InstrToDFSNum(AllSameValue) > InstrToDFSNum(I)) {
      PassPrediction::PassPeeper(3157); // if
      return E;
    }
    NumGVNPhisAllSame++;
    DEBUG(dbgs() << "Simplified PHI node " << *I << " to " << *AllSameValue
                 << "\n");
    deleteExpression(E);
    return createVariableOrConstant(AllSameValue);
  }
  return E;
}

const Expression *
NewGVN::performSymbolicAggrValueEvaluation(Instruction *I) const {
  if (auto *EI = dyn_cast<ExtractValueInst>(I)) {
    PassPrediction::PassPeeper(3158); // if
    auto *II = dyn_cast<IntrinsicInst>(EI->getAggregateOperand());
    if (II && EI->getNumIndices() == 1 && *EI->idx_begin() == 0) {
      PassPrediction::PassPeeper(3159); // if
      unsigned Opcode = 0;
      // EI might be an extract from one of our recognised intrinsics. If it
      // is we'll synthesize a semantically equivalent expression instead on
      // an extract value expression.
      switch (II->getIntrinsicID()) {
      case Intrinsic::sadd_with_overflow:
        PassPrediction::PassPeeper(3160); // case

      case Intrinsic::uadd_with_overflow:
        PassPrediction::PassPeeper(3161); // case

        Opcode = Instruction::Add;
        PassPrediction::PassPeeper(3162); // break
        break;
      case Intrinsic::ssub_with_overflow:
        PassPrediction::PassPeeper(3163); // case

      case Intrinsic::usub_with_overflow:
        PassPrediction::PassPeeper(3164); // case

        Opcode = Instruction::Sub;
        PassPrediction::PassPeeper(3165); // break
        break;
      case Intrinsic::smul_with_overflow:
        PassPrediction::PassPeeper(3166); // case

      case Intrinsic::umul_with_overflow:
        PassPrediction::PassPeeper(3167); // case

        Opcode = Instruction::Mul;
        PassPrediction::PassPeeper(3168); // break
        break;
      default:
        PassPrediction::PassPeeper(3169); // break
        break;
      }

      if (Opcode != 0) {
        // Intrinsic recognized. Grab its args to finish building the
        // expression.
        assert(II->getNumArgOperands() == 2 &&
               "Expect two args for recognised intrinsics.");
        return createBinaryExpression(Opcode, EI->getType(),
                                      II->getArgOperand(0),
                                      II->getArgOperand(1), I);
      }
    }
  }

  return createAggregateValueExpression(I);
}
const Expression *NewGVN::performSymbolicCmpEvaluation(Instruction *I) const {
  auto *CI = dyn_cast<CmpInst>(I);
  // See if our operands are equal to those of a previous predicate, and if so,
  // if it implies true or false.
  auto Op0 = lookupOperandLeader(CI->getOperand(0));
  auto Op1 = lookupOperandLeader(CI->getOperand(1));
  auto OurPredicate = CI->getPredicate();
  if (shouldSwapOperands(Op0, Op1)) {
    PassPrediction::PassPeeper(3170); // if
    std::swap(Op0, Op1);
    OurPredicate = CI->getSwappedPredicate();
  }

  // Avoid processing the same info twice
  const PredicateBase *LastPredInfo = nullptr;
  // See if we know something about the comparison itself, like it is the target
  // of an assume.
  auto *CmpPI = PredInfo->getPredicateInfoFor(I);
  if (dyn_cast_or_null<PredicateAssume>(CmpPI)) {
    PassPrediction::PassPeeper(3171); // if
    return createConstantExpression(ConstantInt::getTrue(CI->getType()));
  }

  if (Op0 == Op1) {
    // This condition does not depend on predicates, no need to add users
    PassPrediction::PassPeeper(3172); // if
    if (CI->isTrueWhenEqual()) {
      PassPrediction::PassPeeper(3173); // if
      return createConstantExpression(ConstantInt::getTrue(CI->getType()));
    } else if (CI->isFalseWhenEqual()) {
      PassPrediction::PassPeeper(3174); // if
      return createConstantExpression(ConstantInt::getFalse(CI->getType()));
    }
  }

  // NOTE: Because we are comparing both operands here and below, and using
  // previous comparisons, we rely on fact that predicateinfo knows to mark
  // comparisons that use renamed operands as users of the earlier comparisons.
  // It is *not* enough to just mark predicateinfo renamed operands as users of
  // the earlier comparisons, because the *other* operand may have changed in a
  // previous iteration.
  // Example:
  // icmp slt %a, %b
  // %b.0 = ssa.copy(%b)
  // false branch:
  // icmp slt %c, %b.0

  // %c and %a may start out equal, and thus, the code below will say the second
  // %icmp is false.  c may become equal to something else, and in that case the
  // %second icmp *must* be reexamined, but would not if only the renamed
  // %operands are considered users of the icmp.

  // *Currently* we only check one level of comparisons back, and only mark one
  // level back as touched when changes appen .  If you modify this code to look
  // back farther through comparisons, you *must* mark the appropriate
  // comparisons as users in PredicateInfo.cpp, or you will cause bugs.  See if
  // we know something just from the operands themselves

  // See if our operands have predicate info, so that we may be able to derive
  // something from a previous comparison.
  for (const auto &Op : CI->operands()) {
    PassPrediction::PassPeeper(3175); // for-range
    auto *PI = PredInfo->getPredicateInfoFor(Op);
    if (const auto *PBranch = dyn_cast_or_null<PredicateBranch>(PI)) {
      PassPrediction::PassPeeper(3176); // if
      if (PI == LastPredInfo) {
        PassPrediction::PassPeeper(3177); // if
        continue;
      }
      LastPredInfo = PI;

      // TODO: Along the false edge, we may know more things too, like icmp of
      // same operands is false.
      // TODO: We only handle actual comparison conditions below, not and/or.
      auto *BranchCond = dyn_cast<CmpInst>(PBranch->Condition);
      if (!BranchCond) {
        PassPrediction::PassPeeper(3178); // if
        continue;
      }
      auto *BranchOp0 = lookupOperandLeader(BranchCond->getOperand(0));
      auto *BranchOp1 = lookupOperandLeader(BranchCond->getOperand(1));
      auto BranchPredicate = BranchCond->getPredicate();
      if (shouldSwapOperands(BranchOp0, BranchOp1)) {
        PassPrediction::PassPeeper(3179); // if
        std::swap(BranchOp0, BranchOp1);
        BranchPredicate = BranchCond->getSwappedPredicate();
      }
      if (BranchOp0 == Op0 && BranchOp1 == Op1) {
        PassPrediction::PassPeeper(3180); // if
        if (PBranch->TrueEdge) {
          // If we know the previous predicate is true and we are in the true
          // edge then we may be implied true or false.
          PassPrediction::PassPeeper(3181); // if
          if (CmpInst::isImpliedTrueByMatchingCmp(BranchPredicate,
                                                  OurPredicate)) {
            PassPrediction::PassPeeper(3183); // if
            addPredicateUsers(PI, I);
            return createConstantExpression(
                ConstantInt::getTrue(CI->getType()));
          }

          if (CmpInst::isImpliedFalseByMatchingCmp(BranchPredicate,
                                                   OurPredicate)) {
            PassPrediction::PassPeeper(3184); // if
            addPredicateUsers(PI, I);
            return createConstantExpression(
                ConstantInt::getFalse(CI->getType()));
          }

        } else {
          // Just handle the ne and eq cases, where if we have the same
          // operands, we may know something.
          PassPrediction::PassPeeper(3182); // else
          if (BranchPredicate == OurPredicate) {
            PassPrediction::PassPeeper(3185); // if
            addPredicateUsers(PI, I);
            // Same predicate, same ops,we know it was false, so this is false.
            return createConstantExpression(
                ConstantInt::getFalse(CI->getType()));
          } else if (BranchPredicate ==
                     CmpInst::getInversePredicate(OurPredicate)) {
            PassPrediction::PassPeeper(3186); // if
            addPredicateUsers(PI, I);
            // Inverse predicate, we know the other was false, so this is true.
            return createConstantExpression(
                ConstantInt::getTrue(CI->getType()));
          }
        }
      }
    }
  }
  // Create expression will take care of simplifyCmpInst
  return createExpression(I);
}

// Return true if V is a value that will always be available (IE can
// be placed anywhere) in the function.  We don't do globals here
// because they are often worse to put in place.
// TODO: Separate cost from availability
static bool alwaysAvailable(Value *V) {
  return isa<Constant>(V) || isa<Argument>(V);
}

// Substitute and symbolize the value before value numbering.
const Expression *
NewGVN::performSymbolicEvaluation(Value *V,
                                  SmallPtrSetImpl<Value *> &Visited) const {
  const Expression *E = nullptr;
  if (auto *C = dyn_cast<Constant>(V)) {
    PassPrediction::PassPeeper(3187); // if
    E = createConstantExpression(C);
  } else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    PassPrediction::PassPeeper(3188); // if
    E = createVariableExpression(V);
  } else {
    // TODO: memory intrinsics.
    // TODO: Some day, we should do the forward propagation and reassociation
    // parts of the algorithm.
    PassPrediction::PassPeeper(3189); // else
    auto *I = cast<Instruction>(V);
    switch (I->getOpcode()) {
    case Instruction::ExtractValue:
      PassPrediction::PassPeeper(3190); // case

    case Instruction::InsertValue:
      PassPrediction::PassPeeper(3191); // case

      E = performSymbolicAggrValueEvaluation(I);
      PassPrediction::PassPeeper(3192); // break
      break;
    case Instruction::PHI:
      PassPrediction::PassPeeper(3193); // case

      E = performSymbolicPHIEvaluation(I);
      PassPrediction::PassPeeper(3194); // break
      break;
    case Instruction::Call:
      PassPrediction::PassPeeper(3195); // case

      E = performSymbolicCallEvaluation(I);
      PassPrediction::PassPeeper(3196); // break
      break;
    case Instruction::Store:
      PassPrediction::PassPeeper(3197); // case

      E = performSymbolicStoreEvaluation(I);
      PassPrediction::PassPeeper(3198); // break
      break;
    case Instruction::Load:
      PassPrediction::PassPeeper(3199); // case

      E = performSymbolicLoadEvaluation(I);
      PassPrediction::PassPeeper(3200); // break
      break;
    case Instruction::BitCast:
      PassPrediction::PassPeeper(3201); // case
      { E = createExpression(I); }
      PassPrediction::PassPeeper(3202); // break
      break;
    case Instruction::ICmp:
      PassPrediction::PassPeeper(3203); // case

    case Instruction::FCmp:
      PassPrediction::PassPeeper(3204); // case
      { E = performSymbolicCmpEvaluation(I); }
      PassPrediction::PassPeeper(3205); // break
      break;
    case Instruction::Add:
      PassPrediction::PassPeeper(3206); // case

    case Instruction::FAdd:
      PassPrediction::PassPeeper(3207); // case

    case Instruction::Sub:
      PassPrediction::PassPeeper(3208); // case

    case Instruction::FSub:
      PassPrediction::PassPeeper(3209); // case

    case Instruction::Mul:
      PassPrediction::PassPeeper(3210); // case

    case Instruction::FMul:
      PassPrediction::PassPeeper(3211); // case

    case Instruction::UDiv:
      PassPrediction::PassPeeper(3212); // case

    case Instruction::SDiv:
      PassPrediction::PassPeeper(3213); // case

    case Instruction::FDiv:
      PassPrediction::PassPeeper(3214); // case

    case Instruction::URem:
      PassPrediction::PassPeeper(3215); // case

    case Instruction::SRem:
      PassPrediction::PassPeeper(3216); // case

    case Instruction::FRem:
      PassPrediction::PassPeeper(3217); // case

    case Instruction::Shl:
      PassPrediction::PassPeeper(3218); // case

    case Instruction::LShr:
      PassPrediction::PassPeeper(3219); // case

    case Instruction::AShr:
      PassPrediction::PassPeeper(3220); // case

    case Instruction::And:
      PassPrediction::PassPeeper(3221); // case

    case Instruction::Or:
      PassPrediction::PassPeeper(3222); // case

    case Instruction::Xor:
      PassPrediction::PassPeeper(3223); // case

    case Instruction::Trunc:
      PassPrediction::PassPeeper(3224); // case

    case Instruction::ZExt:
      PassPrediction::PassPeeper(3225); // case

    case Instruction::SExt:
      PassPrediction::PassPeeper(3226); // case

    case Instruction::FPToUI:
      PassPrediction::PassPeeper(3227); // case

    case Instruction::FPToSI:
      PassPrediction::PassPeeper(3228); // case

    case Instruction::UIToFP:
      PassPrediction::PassPeeper(3229); // case

    case Instruction::SIToFP:
      PassPrediction::PassPeeper(3230); // case

    case Instruction::FPTrunc:
      PassPrediction::PassPeeper(3231); // case

    case Instruction::FPExt:
      PassPrediction::PassPeeper(3232); // case

    case Instruction::PtrToInt:
      PassPrediction::PassPeeper(3233); // case

    case Instruction::IntToPtr:
      PassPrediction::PassPeeper(3234); // case

    case Instruction::Select:
      PassPrediction::PassPeeper(3235); // case

    case Instruction::ExtractElement:
      PassPrediction::PassPeeper(3236); // case

    case Instruction::InsertElement:
      PassPrediction::PassPeeper(3237); // case

    case Instruction::ShuffleVector:
      PassPrediction::PassPeeper(3238); // case

    case Instruction::GetElementPtr:
      PassPrediction::PassPeeper(3239); // case

      E = createExpression(I);
      PassPrediction::PassPeeper(3240); // break
      break;
    default:
      return nullptr;
    }
  }
  return E;
}

// Look up a container in a map, and then call a function for each thing in the
// found container.
template <typename Map, typename KeyType, typename Func>
void NewGVN::for_each_found(Map &M, const KeyType &Key, Func F) {
  const auto Result = M.find_as(Key);
  if (Result != M.end()) {
    PassPrediction::PassPeeper(3038); // if
    for (typename Map::mapped_type::value_type Mapped : Result->second) {
      PassPrediction::PassPeeper(3039); // for-range
      F(Mapped);
    }
  }
}

// Look up a container of values/instructions in a map, and touch all the
// instructions in the container.  Then erase value from the map.
template <typename Map, typename KeyType>
void NewGVN::touchAndErase(Map &M, const KeyType &Key) {
  const auto Result = M.find_as(Key);
  if (Result != M.end()) {
    PassPrediction::PassPeeper(3040); // if
    for (const typename Map::mapped_type::value_type Mapped : Result->second) {
      PassPrediction::PassPeeper(3041); // for-range
      TouchedInstructions.set(InstrToDFSNum(Mapped));
    }
    M.erase(Result);
  }
}

void NewGVN::addAdditionalUsers(Value *To, Value *User) const {
  assert(User && To != User);
  if (isa<Instruction>(To)) {
    PassPrediction::PassPeeper(3241); // if
    AdditionalUsers[To].insert(User);
  }
}

void NewGVN::markUsersTouched(Value *V) {
  // Now mark the users as touched.
  for (auto *User : V->users()) {
    assert(isa<Instruction>(User) && "Use of value not within an instruction?");
    TouchedInstructions.set(InstrToDFSNum(User));
  }
  touchAndErase(AdditionalUsers, V);
}

void NewGVN::addMemoryUsers(const MemoryAccess *To, MemoryAccess *U) const {
  DEBUG(dbgs() << "Adding memory user " << *U << " to " << *To << "\n");
  MemoryToUsers[To].insert(U);
}

void NewGVN::markMemoryDefTouched(const MemoryAccess *MA) {
  TouchedInstructions.set(MemoryToDFSNum(MA));
}

void NewGVN::markMemoryUsersTouched(const MemoryAccess *MA) {
  if (isa<MemoryUse>(MA)) {
    PassPrediction::PassPeeper(3242); // if
    return;
  }
  for (auto U : MA->users()) {
    PassPrediction::PassPeeper(3243); // for-range
    TouchedInstructions.set(MemoryToDFSNum(U));
  }
  touchAndErase(MemoryToUsers, MA);
}

// Add I to the set of users of a given predicate.
void NewGVN::addPredicateUsers(const PredicateBase *PB, Instruction *I) const {
  // Don't add temporary instructions to the user lists.
  if (AllTempInstructions.count(I)) {
    PassPrediction::PassPeeper(3244); // if
    return;
  }

  if (auto *PBranch = dyn_cast<PredicateBranch>(PB)) {
    PassPrediction::PassPeeper(3245); // if
    PredicateToUsers[PBranch->Condition].insert(I);
  } else if (auto *PAssume = dyn_cast<PredicateBranch>(PB)) {
    PassPrediction::PassPeeper(3246); // if
    PredicateToUsers[PAssume->Condition].insert(I);
  }
}

// Touch all the predicates that depend on this instruction.
void NewGVN::markPredicateUsersTouched(Instruction *I) {
  touchAndErase(PredicateToUsers, I);
}

// Mark users affected by a memory leader change.
void NewGVN::markMemoryLeaderChangeTouched(CongruenceClass *CC) {
  for (auto M : CC->memory()) {
    PassPrediction::PassPeeper(3247); // for-range
    markMemoryDefTouched(M);
  }
}

// Touch the instructions that need to be updated after a congruence class has a
// leader change, and mark changed values.
void NewGVN::markValueLeaderChangeTouched(CongruenceClass *CC) {
  for (auto M : *CC) {
    PassPrediction::PassPeeper(3248); // for-range
    if (auto *I = dyn_cast<Instruction>(M)) {
      PassPrediction::PassPeeper(3249); // if
      TouchedInstructions.set(InstrToDFSNum(I));
    }
    LeaderChanges.insert(M);
  }
}

// Give a range of things that have instruction DFS numbers, this will return
// the member of the range with the smallest dfs number.
template <class T, class Range>
T *NewGVN::getMinDFSOfRange(const Range &R) const {
  std::pair<T *, unsigned> MinDFS = {nullptr, ~0U};
  for (const auto X : R) {
    PassPrediction::PassPeeper(3042); // for-range
    auto DFSNum = InstrToDFSNum(X);
    if (DFSNum < MinDFS.second) {
      PassPrediction::PassPeeper(3043); // if
      MinDFS = {X, DFSNum};
    }
  }
  return MinDFS.first;
}

// This function returns the MemoryAccess that should be the next leader of
// congruence class CC, under the assumption that the current leader is going to
// disappear.
const MemoryAccess *NewGVN::getNextMemoryLeader(CongruenceClass *CC) const {
  // TODO: If this ends up to slow, we can maintain a next memory leader like we
  // do for regular leaders.
  // Make sure there will be a leader to find
  assert(!CC->definesNoMemory() && "Can't get next leader if there is none");
  if (CC->getStoreCount() > 0) {
    PassPrediction::PassPeeper(3250); // if
    if (auto *NL = dyn_cast_or_null<StoreInst>(CC->getNextLeader().first)) {
      PassPrediction::PassPeeper(3251); // if
      return getMemoryAccess(NL);
    }
    // Find the store with the minimum DFS number.
    auto *V = getMinDFSOfRange<Value>(make_filter_range(
        *CC, [&](const Value *V) { return isa<StoreInst>(V); }));
    return getMemoryAccess(cast<StoreInst>(V));
  }
  assert(CC->getStoreCount() == 0);

  // Given our assertion, hitting this part must mean
  // !OldClass->memory_empty()
  if (CC->memory_size() == 1) {
    PassPrediction::PassPeeper(3252); // if
    return *CC->memory_begin();
  }
  return getMinDFSOfRange<const MemoryPhi>(CC->memory());
}

// This function returns the next value leader of a congruence class, under the
// assumption that the current leader is going away.  This should end up being
// the next most dominating member.
Value *NewGVN::getNextValueLeader(CongruenceClass *CC) const {
  // We don't need to sort members if there is only 1, and we don't care about
  // sorting the TOP class because everything either gets out of it or is
  // unreachable.

  if (CC->size() == 1 || CC == TOPClass) {
    PassPrediction::PassPeeper(3253); // if
    return *(CC->begin());
  } else if (CC->getNextLeader().first) {
    PassPrediction::PassPeeper(3254); // if
    ++NumGVNAvoidedSortedLeaderChanges;
    return CC->getNextLeader().first;
  } else {
    PassPrediction::PassPeeper(3255); // else
    ++NumGVNSortedLeaderChanges;
    // NOTE: If this ends up to slow, we can maintain a dual structure for
    // member testing/insertion, or keep things mostly sorted, and sort only
    // here, or use SparseBitVector or ....
    return getMinDFSOfRange<Value>(*CC);
  }
}

// Move a MemoryAccess, currently in OldClass, to NewClass, including updates to
// the memory members, etc for the move.
//
// The invariants of this function are:
//
// - I must be moving to NewClass from OldClass
// - The StoreCount of OldClass and NewClass is expected to have been updated
//   for I already if it is is a store.
// - The OldClass memory leader has not been updated yet if I was the leader.
void NewGVN::moveMemoryToNewCongruenceClass(Instruction *I,
                                            MemoryAccess *InstMA,
                                            CongruenceClass *OldClass,
                                            CongruenceClass *NewClass) {
  // If the leader is I, and we had a represenative MemoryAccess, it should
  // be the MemoryAccess of OldClass.
  assert((!InstMA || !OldClass->getMemoryLeader() ||
          OldClass->getLeader() != I ||
          MemoryAccessToClass.lookup(OldClass->getMemoryLeader()) ==
              MemoryAccessToClass.lookup(InstMA)) &&
         "Representative MemoryAccess mismatch");
  // First, see what happens to the new class
  if (!NewClass->getMemoryLeader()) {
    // Should be a new class, or a store becoming a leader of a new class.
    assert(NewClass->size() == 1 ||
           (isa<StoreInst>(I) && NewClass->getStoreCount() == 1));
    NewClass->setMemoryLeader(InstMA);
    // Mark it touched if we didn't just create a singleton
    DEBUG(dbgs() << "Memory class leader change for class " << NewClass->getID()
                 << " due to new memory instruction becoming leader\n");
    markMemoryLeaderChangeTouched(NewClass);
  }
  setMemoryClass(InstMA, NewClass);
  // Now, fixup the old class if necessary
  if (OldClass->getMemoryLeader() == InstMA) {
    PassPrediction::PassPeeper(3256); // if
    if (!OldClass->definesNoMemory()) {
      PassPrediction::PassPeeper(3257); // if
      OldClass->setMemoryLeader(getNextMemoryLeader(OldClass));
      DEBUG(dbgs() << "Memory class leader change for class "
                   << OldClass->getID() << " to "
                   << *OldClass->getMemoryLeader()
                   << " due to removal of old leader " << *InstMA << "\n");
      markMemoryLeaderChangeTouched(OldClass);
    } else {
      PassPrediction::PassPeeper(3258); // else
      OldClass->setMemoryLeader(nullptr);
    }
  }
}

// Move a value, currently in OldClass, to be part of NewClass
// Update OldClass and NewClass for the move (including changing leaders, etc).
void NewGVN::moveValueToNewCongruenceClass(Instruction *I, const Expression *E,
                                           CongruenceClass *OldClass,
                                           CongruenceClass *NewClass) {
  if (I == OldClass->getNextLeader().first) {
    PassPrediction::PassPeeper(3259); // if
    OldClass->resetNextLeader();
  }

  OldClass->erase(I);
  NewClass->insert(I);

  if (NewClass->getLeader() != I) {
    PassPrediction::PassPeeper(3260); // if
    NewClass->addPossibleNextLeader({I, InstrToDFSNum(I)});
  }
  // Handle our special casing of stores.
  if (auto *SI = dyn_cast<StoreInst>(I)) {
    PassPrediction::PassPeeper(3261); // if
    OldClass->decStoreCount();
    // Okay, so when do we want to make a store a leader of a class?
    // If we have a store defined by an earlier load, we want the earlier load
    // to lead the class.
    // If we have a store defined by something else, we want the store to lead
    // the class so everything else gets the "something else" as a value.
    // If we have a store as the single member of the class, we want the store
    // as the leader
    if (NewClass->getStoreCount() == 0 && !NewClass->getStoredValue()) {
      // If it's a store expression we are using, it means we are not equivalent
      // to something earlier.
      PassPrediction::PassPeeper(3262); // if
      if (auto *SE = dyn_cast<StoreExpression>(E)) {
        PassPrediction::PassPeeper(3263); // if
        NewClass->setStoredValue(SE->getStoredValue());
        markValueLeaderChangeTouched(NewClass);
        // Shift the new class leader to be the store
        DEBUG(dbgs() << "Changing leader of congruence class "
                     << NewClass->getID() << " from " << *NewClass->getLeader()
                     << " to  " << *SI << " because store joined class\n");
        // If we changed the leader, we have to mark it changed because we don't
        // know what it will do to symbolic evaluation.
        NewClass->setLeader(SI);
      }
      // We rely on the code below handling the MemoryAccess change.
    }
    NewClass->incStoreCount();
  }
  // True if there is no memory instructions left in a class that had memory
  // instructions before.

  // If it's not a memory use, set the MemoryAccess equivalence
  auto *InstMA = dyn_cast_or_null<MemoryDef>(getMemoryAccess(I));
  if (InstMA) {
    PassPrediction::PassPeeper(3264); // if
    moveMemoryToNewCongruenceClass(I, InstMA, OldClass, NewClass);
  }
  ValueToClass[I] = NewClass;
  // See if we destroyed the class or need to swap leaders.
  if (OldClass->empty() && OldClass != TOPClass) {
    PassPrediction::PassPeeper(3265); // if
    if (OldClass->getDefiningExpr()) {
      DEBUG(dbgs() << "Erasing expression " << *OldClass->getDefiningExpr()
                   << " from table\n");
      // We erase it as an exact expression to make sure we don't just erase an
      // equivalent one.
      auto Iter = ExpressionToClass.find_as(
          ExactEqualsExpression(*OldClass->getDefiningExpr()));
      if (Iter != ExpressionToClass.end()) {
        PassPrediction::PassPeeper(3266); // if
        ExpressionToClass.erase(Iter);
      }
#ifdef EXPENSIVE_CHECKS
      assert(
          (*OldClass->getDefiningExpr() != *E || ExpressionToClass.lookup(E)) &&
          "We erased the expression we just inserted, which should not happen");
#endif
    }
  } else if (OldClass->getLeader() == I) {
    // When the leader changes, the value numbering of
    // everything may change due to symbolization changes, so we need to
    // reprocess.
    DEBUG(dbgs() << "Value class leader change for class " << OldClass->getID()
                 << "\n");
    ++NumGVNLeaderChanges;
    // Destroy the stored value if there are no more stores to represent it.
    // Note that this is basically clean up for the expression removal that
    // happens below.  If we remove stores from a class, we may leave it as a
    // class of equivalent memory phis.
    if (OldClass->getStoreCount() == 0) {
      PassPrediction::PassPeeper(3267); // if
      if (OldClass->getStoredValue()) {
        PassPrediction::PassPeeper(3268); // if
        OldClass->setStoredValue(nullptr);
      }
    }
    OldClass->setLeader(getNextValueLeader(OldClass));
    OldClass->resetNextLeader();
    markValueLeaderChangeTouched(OldClass);
  }
}

// For a given expression, mark the phi of ops instructions that could have
// changed as a result.
void NewGVN::markPhiOfOpsChanged(const Expression *E) {
  touchAndErase(ExpressionToPhiOfOps, ExactEqualsExpression(*E));
}

// Perform congruence finding on a given value numbering expression.
void NewGVN::performCongruenceFinding(Instruction *I, const Expression *E) {
  // This is guaranteed to return something, since it will at least find
  // TOP.

  CongruenceClass *IClass = ValueToClass.lookup(I);
  assert(IClass && "Should have found a IClass");
  // Dead classes should have been eliminated from the mapping.
  assert(!IClass->isDead() && "Found a dead class");

  CongruenceClass *EClass = nullptr;
  if (const auto *VE = dyn_cast<VariableExpression>(E)) {
    PassPrediction::PassPeeper(3269); // if
    EClass = ValueToClass.lookup(VE->getVariableValue());
  } else if (isa<DeadExpression>(E)) {
    PassPrediction::PassPeeper(3270); // if
    EClass = TOPClass;
  }
  if (!EClass) {
    PassPrediction::PassPeeper(3271); // if
    auto lookupResult = ExpressionToClass.insert({E, nullptr});

    // If it's not in the value table, create a new congruence class.
    if (lookupResult.second) {
      PassPrediction::PassPeeper(3272); // if
      CongruenceClass *NewClass = createCongruenceClass(nullptr, E);
      auto place = lookupResult.first;
      place->second = NewClass;

      // Constants and variables should always be made the leader.
      if (const auto *CE = dyn_cast<ConstantExpression>(E)) {
        PassPrediction::PassPeeper(3274); // if
        NewClass->setLeader(CE->getConstantValue());
      } else if (const auto *SE = dyn_cast<StoreExpression>(E)) {
        PassPrediction::PassPeeper(3275); // if
        StoreInst *SI = SE->getStoreInst();
        NewClass->setLeader(SI);
        NewClass->setStoredValue(SE->getStoredValue());
        // The RepMemoryAccess field will be filled in properly by the
        // moveValueToNewCongruenceClass call.
      } else {
        PassPrediction::PassPeeper(3276); // else
        NewClass->setLeader(I);
      }
      assert(!isa<VariableExpression>(E) &&
             "VariableExpression should have been handled already");

      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << *I
                   << " using expression " << *E << " at " << NewClass->getID()
                   << " and leader " << *(NewClass->getLeader()));
      if (NewClass->getStoredValue()) {
        DEBUG(dbgs() << " and stored value " << *(NewClass->getStoredValue()));
      }
      DEBUG(dbgs() << "\n");
    } else {
      PassPrediction::PassPeeper(3273); // else
      EClass = lookupResult.first->second;
      if (isa<ConstantExpression>(E)) {
        assert((isa<Constant>(EClass->getLeader()) ||
                (EClass->getStoredValue() &&
                 isa<Constant>(EClass->getStoredValue()))) &&
               "Any class with a constant expression should have a "
               "constant leader");
      }

      assert(EClass && "Somehow don't have an eclass");

      assert(!EClass->isDead() && "We accidentally looked up a dead class");
    }
  }
  bool ClassChanged = IClass != EClass;
  bool LeaderChanged = LeaderChanges.erase(I);
  if (ClassChanged || LeaderChanged) {
    DEBUG(dbgs() << "New class " << EClass->getID() << " for expression " << *E
                 << "\n");
    if (ClassChanged) {
      PassPrediction::PassPeeper(3277); // if
      moveValueToNewCongruenceClass(I, E, IClass, EClass);
      markPhiOfOpsChanged(E);
    }

    markUsersTouched(I);
    if (MemoryAccess *MA = getMemoryAccess(I)) {
      PassPrediction::PassPeeper(3278); // if
      markMemoryUsersTouched(MA);
    }
    if (auto *CI = dyn_cast<CmpInst>(I)) {
      PassPrediction::PassPeeper(3279); // if
      markPredicateUsersTouched(CI);
    }
  }
  // If we changed the class of the store, we want to ensure nothing finds the
  // old store expression.  In particular, loads do not compare against stored
  // value, so they will find old store expressions (and associated class
  // mappings) if we leave them in the table.
  if (ClassChanged && isa<StoreInst>(I)) {
    PassPrediction::PassPeeper(3280); // if
    auto *OldE = ValueToExpression.lookup(I);
    // It could just be that the old class died. We don't want to erase it if we
    // just moved classes.
    if (OldE && isa<StoreExpression>(OldE) && *E != *OldE) {
      // Erase this as an exact expression to ensure we don't erase expressions
      // equivalent to it.
      PassPrediction::PassPeeper(3281); // if
      auto Iter = ExpressionToClass.find_as(ExactEqualsExpression(*OldE));
      if (Iter != ExpressionToClass.end()) {
        PassPrediction::PassPeeper(3282); // if
        ExpressionToClass.erase(Iter);
      }
    }
  }
  ValueToExpression[I] = E;
}

// Process the fact that Edge (from, to) is reachable, including marking
// any newly reachable blocks and instructions for processing.
void NewGVN::updateReachableEdge(BasicBlock *From, BasicBlock *To) {
  // Check if the Edge was reachable before.
  if (ReachableEdges.insert({From, To}).second) {
    // If this block wasn't reachable before, all instructions are touched.
    PassPrediction::PassPeeper(3283); // if
    if (ReachableBlocks.insert(To).second) {
      DEBUG(dbgs() << "Block " << getBlockName(To) << " marked reachable\n");
      const auto &InstRange = BlockInstRange.lookup(To);
      TouchedInstructions.set(InstRange.first, InstRange.second);
    } else {
      DEBUG(dbgs() << "Block " << getBlockName(To)
                   << " was reachable, but new edge {" << getBlockName(From)
                   << "," << getBlockName(To) << "} to it found\n");

      // We've made an edge reachable to an existing block, which may
      // impact predicates. Otherwise, only mark the phi nodes as touched, as
      // they are the only thing that depend on new edges. Anything using their
      // values will get propagated to if necessary.
      if (MemoryAccess *MemPhi = getMemoryAccess(To)) {
        PassPrediction::PassPeeper(3284); // if
        TouchedInstructions.set(InstrToDFSNum(MemPhi));
      }

      auto BI = To->begin();
      while (isa<PHINode>(BI)) {
        PassPrediction::PassPeeper(3285); // while
        TouchedInstructions.set(InstrToDFSNum(&*BI));
        ++BI;
      }
      for_each_found(PHIOfOpsPHIs, To, [&](const PHINode *I) {
        TouchedInstructions.set(InstrToDFSNum(I));
      });
    }
  }
}

// Given a predicate condition (from a switch, cmp, or whatever) and a block,
// see if we know some constant value for it already.
Value *NewGVN::findConditionEquivalence(Value *Cond) const {
  auto Result = lookupOperandLeader(Cond);
  return isa<Constant>(Result) ? Result : nullptr;
}

// Process the outgoing edges of a block for reachability.
void NewGVN::processOutgoingEdges(TerminatorInst *TI, BasicBlock *B) {
  // Evaluate reachability of terminator instruction.
  BranchInst *BR;
  if ((BR = dyn_cast<BranchInst>(TI)) && BR->isConditional()) {
    PassPrediction::PassPeeper(3286); // if
    Value *Cond = BR->getCondition();
    Value *CondEvaluated = findConditionEquivalence(Cond);
    if (!CondEvaluated) {
      PassPrediction::PassPeeper(3287); // if
      if (auto *I = dyn_cast<Instruction>(Cond)) {
        PassPrediction::PassPeeper(3288); // if
        const Expression *E = createExpression(I);
        if (const auto *CE = dyn_cast<ConstantExpression>(E)) {
          PassPrediction::PassPeeper(3289); // if
          CondEvaluated = CE->getConstantValue();
        }
      } else if (isa<ConstantInt>(Cond)) {
        PassPrediction::PassPeeper(3290); // if
        CondEvaluated = Cond;
      }
    }
    ConstantInt *CI;
    BasicBlock *TrueSucc = BR->getSuccessor(0);
    BasicBlock *FalseSucc = BR->getSuccessor(1);
    if (CondEvaluated && (CI = dyn_cast<ConstantInt>(CondEvaluated))) {
      PassPrediction::PassPeeper(3291); // if
      if (CI->isOne()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to true\n");
        updateReachableEdge(B, TrueSucc);
      } else if (CI->isZero()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to false\n");
        updateReachableEdge(B, FalseSucc);
      }
    } else {
      PassPrediction::PassPeeper(3292); // else
      updateReachableEdge(B, TrueSucc);
      updateReachableEdge(B, FalseSucc);
    }
  } else if (auto *SI = dyn_cast<SwitchInst>(TI)) {
    // For switches, propagate the case values into the case
    // destinations.

    // Remember how many outgoing edges there are to every successor.
    PassPrediction::PassPeeper(3293); // if
    SmallDenseMap<BasicBlock *, unsigned, 16> SwitchEdges;

    Value *SwitchCond = SI->getCondition();
    Value *CondEvaluated = findConditionEquivalence(SwitchCond);
    // See if we were able to turn this switch statement into a constant.
    if (CondEvaluated && isa<ConstantInt>(CondEvaluated)) {
      PassPrediction::PassPeeper(3295); // if
      auto *CondVal = cast<ConstantInt>(CondEvaluated);
      // We should be able to get case value for this.
      auto Case = *SI->findCaseValue(CondVal);
      if (Case.getCaseSuccessor() == SI->getDefaultDest()) {
        // We proved the value is outside of the range of the case.
        // We can't do anything other than mark the default dest as reachable,
        // and go home.
        PassPrediction::PassPeeper(3297); // if
        updateReachableEdge(B, SI->getDefaultDest());
        return;
      }
      // Now get where it goes and mark it reachable.
      BasicBlock *TargetBlock = Case.getCaseSuccessor();
      updateReachableEdge(B, TargetBlock);
    } else {
      PassPrediction::PassPeeper(3296); // else
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        PassPrediction::PassPeeper(3298); // for
        BasicBlock *TargetBlock = SI->getSuccessor(i);
        ++SwitchEdges[TargetBlock];
        updateReachableEdge(B, TargetBlock);
      }
    }
  } else {
    // Otherwise this is either unconditional, or a type we have no
    // idea about. Just mark successors as reachable.
    PassPrediction::PassPeeper(3294); // else
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
      PassPrediction::PassPeeper(3299); // for
      BasicBlock *TargetBlock = TI->getSuccessor(i);
      updateReachableEdge(B, TargetBlock);
    }

    // This also may be a memory defining terminator, in which case, set it
    // equivalent only to itself.
    //
    auto *MA = getMemoryAccess(TI);
    if (MA && !isa<MemoryUse>(MA)) {
      PassPrediction::PassPeeper(3300); // if
      auto *CC = ensureLeaderOfMemoryClass(MA);
      if (setMemoryClass(MA, CC)) {
        PassPrediction::PassPeeper(3301); // if
        markMemoryUsersTouched(MA);
      }
    }
  }
}

void NewGVN::addPhiOfOps(PHINode *Op, BasicBlock *BB,
                         Instruction *ExistingValue) {
  InstrDFS[Op] = InstrToDFSNum(ExistingValue);
  AllTempInstructions.insert(Op);
  PHIOfOpsPHIs[BB].push_back(Op);
  TempToBlock[Op] = BB;
  RealToTemp[ExistingValue] = Op;
}

static bool okayForPHIOfOps(const Instruction *I) {
  return isa<BinaryOperator>(I) || isa<SelectInst>(I) || isa<CmpInst>(I) ||
         isa<LoadInst>(I);
}

// When we see an instruction that is an op of phis, generate the equivalent phi
// of ops form.
const Expression *
NewGVN::makePossiblePhiOfOps(Instruction *I,
                             SmallPtrSetImpl<Value *> &Visited) {
  if (!okayForPHIOfOps(I)) {
    PassPrediction::PassPeeper(3302); // if
    return nullptr;
  }

  if (!Visited.insert(I).second) {
    PassPrediction::PassPeeper(3303); // if
    return nullptr;
  }
  // For now, we require the instruction be cycle free because we don't
  // *always* create a phi of ops for instructions that could be done as phi
  // of ops, we only do it if we think it is useful.  If we did do it all the
  // time, we could remove the cycle free check.
  if (!isCycleFree(I)) {
    PassPrediction::PassPeeper(3304); // if
    return nullptr;
  }

  unsigned IDFSNum = InstrToDFSNum(I);
  SmallPtrSet<const Value *, 8> ProcessedPHIs;
  // TODO: We don't do phi translation on memory accesses because it's
  // complicated. For a load, we'd need to be able to simulate a new memoryuse,
  // which we don't have a good way of doing ATM.
  auto *MemAccess = getMemoryAccess(I);
  // If the memory operation is defined by a memory operation this block that
  // isn't a MemoryPhi, transforming the pointer backwards through a scalar phi
  // can't help, as it would still be killed by that memory operation.
  if (MemAccess && !isa<MemoryPhi>(MemAccess->getDefiningAccess()) &&
      MemAccess->getDefiningAccess()->getBlock() == I->getParent()) {
    PassPrediction::PassPeeper(3305); // if
    return nullptr;
  }

  // Convert op of phis to phi of ops
  for (auto &Op : I->operands()) {
    // TODO: We can't handle expressions that must be recursively translated
    // IE
    // a = phi (b, c)
    // f = use a
    // g = f + phi of something
    // To properly make a phi of ops for g, we'd have to properly translate and
    // use the instruction for f.  We should add this by splitting out the
    // instruction creation we do below.
    PassPrediction::PassPeeper(3306); // for-range
    if (isa<Instruction>(Op) && PHINodeUses.count(cast<Instruction>(Op))) {
      PassPrediction::PassPeeper(3307); // if
      return nullptr;
    }
    if (!isa<PHINode>(Op)) {
      PassPrediction::PassPeeper(3308); // if
      continue;
    }
    auto *OpPHI = cast<PHINode>(Op);
    // No point in doing this for one-operand phis.
    if (OpPHI->getNumOperands() == 1) {
      PassPrediction::PassPeeper(3309); // if
      continue;
    }
    if (!DebugCounter::shouldExecute(PHIOfOpsCounter)) {
      PassPrediction::PassPeeper(3310); // if
      return nullptr;
    }
    SmallVector<std::pair<Value *, BasicBlock *>, 4> Ops;
    auto *PHIBlock = getBlockForValue(OpPHI);
    for (auto PredBB : OpPHI->blocks()) {
      PassPrediction::PassPeeper(3311); // for-range
      Value *FoundVal = nullptr;
      // We could just skip unreachable edges entirely but it's tricky to do
      // with rewriting existing phi nodes.
      if (ReachableEdges.count({PredBB, PHIBlock})) {
        // Clone the instruction, create an expression from it, and see if we
        // have a leader.
        PassPrediction::PassPeeper(3312); // if
        Instruction *ValueOp = I->clone();
        if (MemAccess) {
          PassPrediction::PassPeeper(3313); // if
          TempToMemory.insert({ValueOp, MemAccess});
        }

        for (auto &Op : ValueOp->operands()) {
          PassPrediction::PassPeeper(3314); // for-range
          Op = Op->DoPHITranslation(PHIBlock, PredBB);
          // When this operand changes, it could change whether there is a
          // leader for us or not.
          addAdditionalUsers(Op, I);
        }
        // Make sure it's marked as a temporary instruction.
        AllTempInstructions.insert(ValueOp);
        // and make sure anything that tries to add it's DFS number is
        // redirected to the instruction we are making a phi of ops
        // for.
        InstrDFS.insert({ValueOp, IDFSNum});
        const Expression *E = performSymbolicEvaluation(ValueOp, Visited);
        InstrDFS.erase(ValueOp);
        AllTempInstructions.erase(ValueOp);
        ValueOp->deleteValue();
        if (MemAccess) {
          PassPrediction::PassPeeper(3315); // if
          TempToMemory.erase(ValueOp);
        }
        if (!E) {
          PassPrediction::PassPeeper(3316); // if
          return nullptr;
        }
        FoundVal = findPhiOfOpsLeader(E, PredBB);
        if (!FoundVal) {
          PassPrediction::PassPeeper(3317); // if
          ExpressionToPhiOfOps[E].insert(I);
          return nullptr;
        }
        if (auto *SI = dyn_cast<StoreInst>(FoundVal)) {
          PassPrediction::PassPeeper(3318); // if
          FoundVal = SI->getValueOperand();
        }
      } else {
        DEBUG(dbgs() << "Skipping phi of ops operand for incoming block "
                     << getBlockName(PredBB)
                     << " because the block is unreachable\n");
        FoundVal = UndefValue::get(I->getType());
      }

      Ops.push_back({FoundVal, PredBB});
      DEBUG(dbgs() << "Found phi of ops operand " << *FoundVal << " in "
                   << getBlockName(PredBB) << "\n");
    }
    auto *ValuePHI = RealToTemp.lookup(I);
    bool NewPHI = false;
    if (!ValuePHI) {
      PassPrediction::PassPeeper(3319); // if
      ValuePHI = PHINode::Create(I->getType(), OpPHI->getNumOperands());
      addPhiOfOps(ValuePHI, PHIBlock, I);
      NewPHI = true;
      NumGVNPHIOfOpsCreated++;
    }
    if (NewPHI) {
      PassPrediction::PassPeeper(3320); // if
      for (auto PHIOp : Ops) {
        PassPrediction::PassPeeper(3322); // for-range
        ValuePHI->addIncoming(PHIOp.first, PHIOp.second);
      }
    } else {
      PassPrediction::PassPeeper(3321); // else
      unsigned int i = 0;
      for (auto PHIOp : Ops) {
        PassPrediction::PassPeeper(3323); // for-range
        ValuePHI->setIncomingValue(i, PHIOp.first);
        ValuePHI->setIncomingBlock(i, PHIOp.second);
        ++i;
      }
    }

    DEBUG(dbgs() << "Created phi of ops " << *ValuePHI << " for " << *I
                 << "\n");
    return performSymbolicEvaluation(ValuePHI, Visited);
  }
  return nullptr;
}

// The algorithm initially places the values of the routine in the TOP
// congruence class. The leader of TOP is the undetermined value `undef`.
// When the algorithm has finished, values still in TOP are unreachable.
void NewGVN::initializeCongruenceClasses(Function &F) {
  NextCongruenceNum = 0;

  // Note that even though we use the live on entry def as a representative
  // MemoryAccess, it is *not* the same as the actual live on entry def. We
  // have no real equivalemnt to undef for MemoryAccesses, and so we really
  // should be checking whether the MemoryAccess is top if we want to know if it
  // is equivalent to everything.  Otherwise, what this really signifies is that
  // the access "it reaches all the way back to the beginning of the function"

  // Initialize all other instructions to be in TOP class.
  TOPClass = createCongruenceClass(nullptr, nullptr);
  TOPClass->setMemoryLeader(MSSA->getLiveOnEntryDef());
  //  The live on entry def gets put into it's own class
  MemoryAccessToClass[MSSA->getLiveOnEntryDef()] =
      createMemoryClass(MSSA->getLiveOnEntryDef());

  for (auto DTN : nodes(DT)) {
    PassPrediction::PassPeeper(3324); // for-range
    BasicBlock *BB = DTN->getBlock();
    // All MemoryAccesses are equivalent to live on entry to start. They must
    // be initialized to something so that initial changes are noticed. For
    // the maximal answer, we initialize them all to be the same as
    // liveOnEntry.
    auto *MemoryBlockDefs = MSSA->getBlockDefs(BB);
    if (MemoryBlockDefs) {
      PassPrediction::PassPeeper(3325); // if
      for (const auto &Def : *MemoryBlockDefs) {
        PassPrediction::PassPeeper(3326); // for-range
        MemoryAccessToClass[&Def] = TOPClass;
        auto *MD = dyn_cast<MemoryDef>(&Def);
        // Insert the memory phis into the member list.
        if (!MD) {
          PassPrediction::PassPeeper(3327); // if
          const MemoryPhi *MP = cast<MemoryPhi>(&Def);
          TOPClass->memory_insert(MP);
          MemoryPhiState.insert({MP, MPS_TOP});
        }

        if (MD && isa<StoreInst>(MD->getMemoryInst())) {
          PassPrediction::PassPeeper(3328); // if
          TOPClass->incStoreCount();
        }
      }
    }
    for (auto &I : *BB) {
      // TODO: Move to helper
      PassPrediction::PassPeeper(3329); // for-range
      if (isa<PHINode>(&I)) {
        PassPrediction::PassPeeper(3330); // if
        for (auto *U : I.users()) {
          PassPrediction::PassPeeper(3331); // for-range
          if (auto *UInst = dyn_cast<Instruction>(U)) {
            PassPrediction::PassPeeper(3332); // if
            if (InstrToDFSNum(UInst) != 0 && okayForPHIOfOps(UInst)) {
              PassPrediction::PassPeeper(3333); // if
              PHINodeUses.insert(UInst);
            }
          }
        }
      }
      // Don't insert void terminators into the class. We don't value number
      // them, and they just end up sitting in TOP.
      if (isa<TerminatorInst>(I) && I.getType()->isVoidTy()) {
        PassPrediction::PassPeeper(3334); // if
        continue;
      }
      TOPClass->insert(&I);
      ValueToClass[&I] = TOPClass;
    }
  }

  // Initialize arguments to be in their own unique congruence classes
  for (auto &FA : F.args()) {
    PassPrediction::PassPeeper(3335); // for-range
    createSingletonCongruenceClass(&FA);
  }
}

void NewGVN::cleanupTables() {
  for (unsigned i = 0, e = CongruenceClasses.size(); i != e; ++i) {
    DEBUG(dbgs() << "Congruence class " << CongruenceClasses[i]->getID()
                 << " has " << CongruenceClasses[i]->size() << " members\n");
    // Make sure we delete the congruence class (probably worth switching to
    // a unique_ptr at some point.
    delete CongruenceClasses[i];
    CongruenceClasses[i] = nullptr;
  }

  // Destroy the value expressions
  SmallVector<Instruction *, 8> TempInst(AllTempInstructions.begin(),
                                         AllTempInstructions.end());
  AllTempInstructions.clear();

  // We have to drop all references for everything first, so there are no uses
  // left as we delete them.
  for (auto *I : TempInst) {
    PassPrediction::PassPeeper(3336); // for-range
    I->dropAllReferences();
  }

  while (!TempInst.empty()) {
    PassPrediction::PassPeeper(3337); // while
    auto *I = TempInst.back();
    TempInst.pop_back();
    I->deleteValue();
  }

  ValueToClass.clear();
  ArgRecycler.clear(ExpressionAllocator);
  ExpressionAllocator.Reset();
  CongruenceClasses.clear();
  ExpressionToClass.clear();
  ValueToExpression.clear();
  RealToTemp.clear();
  AdditionalUsers.clear();
  ExpressionToPhiOfOps.clear();
  TempToBlock.clear();
  TempToMemory.clear();
  PHIOfOpsPHIs.clear();
  ReachableBlocks.clear();
  ReachableEdges.clear();
#ifndef NDEBUG
  ProcessedCount.clear();
#endif
  InstrDFS.clear();
  InstructionsToErase.clear();
  DFSToInstr.clear();
  BlockInstRange.clear();
  TouchedInstructions.clear();
  MemoryAccessToClass.clear();
  PredicateToUsers.clear();
  MemoryToUsers.clear();
}

// Assign local DFS number mapping to instructions, and leave space for Value
// PHI's.
std::pair<unsigned, unsigned> NewGVN::assignDFSNumbers(BasicBlock *B,
                                                       unsigned Start) {
  unsigned End = Start;
  if (MemoryAccess *MemPhi = getMemoryAccess(B)) {
    PassPrediction::PassPeeper(3338); // if
    InstrDFS[MemPhi] = End++;
    DFSToInstr.emplace_back(MemPhi);
  }

  // Then the real block goes next.
  for (auto &I : *B) {
    // There's no need to call isInstructionTriviallyDead more than once on
    // an instruction. Therefore, once we know that an instruction is dead
    // we change its DFS number so that it doesn't get value numbered.
    PassPrediction::PassPeeper(3339); // for-range
    if (isInstructionTriviallyDead(&I, TLI)) {
      PassPrediction::PassPeeper(3340); // if
      InstrDFS[&I] = 0;
      DEBUG(dbgs() << "Skipping trivially dead instruction " << I << "\n");
      markInstructionForDeletion(&I);
      continue;
    }
    InstrDFS[&I] = End++;
    DFSToInstr.emplace_back(&I);
  }

  // All of the range functions taken half-open ranges (open on the end side).
  // So we do not subtract one from count, because at this point it is one
  // greater than the last instruction.
  return std::make_pair(Start, End);
}

void NewGVN::updateProcessedCount(const Value *V) {
#ifndef NDEBUG
  if (ProcessedCount.count(V) == 0) {
    PassPrediction::PassPeeper(3341); // if
    ProcessedCount.insert({V, 1});
  } else {
    PassPrediction::PassPeeper(3342); // else
    ++ProcessedCount[V];
    assert(ProcessedCount[V] < 100 &&
           "Seem to have processed the same Value a lot");
  }
#endif
}
// Evaluate MemoryPhi nodes symbolically, just like PHI nodes
void NewGVN::valueNumberMemoryPhi(MemoryPhi *MP) {
  // If all the arguments are the same, the MemoryPhi has the same value as the
  // argument.  Filter out unreachable blocks and self phis from our operands.
  // TODO: We could do cycle-checking on the memory phis to allow valueizing for
  // self-phi checking.
  const BasicBlock *PHIBlock = MP->getBlock();
  auto Filtered = make_filter_range(MP->operands(), [&](const Use &U) {
    return cast<MemoryAccess>(U) != MP &&
           !isMemoryAccessTOP(cast<MemoryAccess>(U)) &&
           ReachableEdges.count({MP->getIncomingBlock(U), PHIBlock});
  });
  // If all that is left is nothing, our memoryphi is undef. We keep it as
  // InitialClass.  Note: The only case this should happen is if we have at
  // least one self-argument.
  if (Filtered.begin() == Filtered.end()) {
    PassPrediction::PassPeeper(3343); // if
    if (setMemoryClass(MP, TOPClass)) {
      PassPrediction::PassPeeper(3344); // if
      markMemoryUsersTouched(MP);
    }
    return;
  }

  // Transform the remaining operands into operand leaders.
  // FIXME: mapped_iterator should have a range version.
  auto LookupFunc = [&](const Use &U) {
    return lookupMemoryLeader(cast<MemoryAccess>(U));
  };
  auto MappedBegin = map_iterator(Filtered.begin(), LookupFunc);
  auto MappedEnd = map_iterator(Filtered.end(), LookupFunc);

  // and now check if all the elements are equal.
  // Sadly, we can't use std::equals since these are random access iterators.
  const auto *AllSameValue = *MappedBegin;
  ++MappedBegin;
  bool AllEqual = std::all_of(
      MappedBegin, MappedEnd,
      [&AllSameValue](const MemoryAccess *V) { return V == AllSameValue; });

  if (AllEqual) {
    DEBUG(dbgs() << "Memory Phi value numbered to " << *AllSameValue << "\n");
  } else {
    DEBUG(dbgs() << "Memory Phi value numbered to itself\n");
  }
  // If it's equal to something, it's in that class. Otherwise, it has to be in
  // a class where it is the leader (other things may be equivalent to it, but
  // it needs to start off in its own class, which means it must have been the
  // leader, and it can't have stopped being the leader because it was never
  // removed).
  CongruenceClass *CC =
      AllEqual ? getMemoryClass(AllSameValue) : ensureLeaderOfMemoryClass(MP);
  auto OldState = MemoryPhiState.lookup(MP);
  assert(OldState != MPS_Invalid && "Invalid memory phi state");
  auto NewState = AllEqual ? MPS_Equivalent : MPS_Unique;
  MemoryPhiState[MP] = NewState;
  if (setMemoryClass(MP, CC) || OldState != NewState) {
    PassPrediction::PassPeeper(3345); // if
    markMemoryUsersTouched(MP);
  }
}

// Value number a single instruction, symbolically evaluating, performing
// congruence finding, and updating mappings.
void NewGVN::valueNumberInstruction(Instruction *I) {
  DEBUG(dbgs() << "Processing instruction " << *I << "\n");
  if (!I->isTerminator()) {
    PassPrediction::PassPeeper(3346); // if
    const Expression *Symbolized = nullptr;
    SmallPtrSet<Value *, 2> Visited;
    if (DebugCounter::shouldExecute(VNCounter)) {
      PassPrediction::PassPeeper(3348); // if
      Symbolized = performSymbolicEvaluation(I, Visited);
      // Make a phi of ops if necessary
      if (Symbolized && !isa<ConstantExpression>(Symbolized) &&
          !isa<VariableExpression>(Symbolized) && PHINodeUses.count(I)) {
        PassPrediction::PassPeeper(3350); // if
        auto *PHIE = makePossiblePhiOfOps(I, Visited);
        if (PHIE) {
          PassPrediction::PassPeeper(3351); // if
          Symbolized = PHIE;
        }
      }

    } else {
      // Mark the instruction as unused so we don't value number it again.
      PassPrediction::PassPeeper(3349); // else
      InstrDFS[I] = 0;
    }
    // If we couldn't come up with a symbolic expression, use the unknown
    // expression
    if (Symbolized == nullptr) {
      PassPrediction::PassPeeper(3352); // if
      Symbolized = createUnknownExpression(I);
    }
    performCongruenceFinding(I, Symbolized);
  } else {
    // Handle terminators that return values. All of them produce values we
    // don't currently understand.  We don't place non-value producing
    // terminators in a class.
    PassPrediction::PassPeeper(3347); // else
    if (!I->getType()->isVoidTy()) {
      PassPrediction::PassPeeper(3353); // if
      auto *Symbolized = createUnknownExpression(I);
      performCongruenceFinding(I, Symbolized);
    }
    processOutgoingEdges(dyn_cast<TerminatorInst>(I), I->getParent());
  }
}

// Check if there is a path, using single or equal argument phi nodes, from
// First to Second.
bool NewGVN::singleReachablePHIPath(
    SmallPtrSet<const MemoryAccess *, 8> &Visited, const MemoryAccess *First,
    const MemoryAccess *Second) const {
  if (First == Second) {
    PassPrediction::PassPeeper(3354); // if
    return true;
  }
  if (MSSA->isLiveOnEntryDef(First)) {
    PassPrediction::PassPeeper(3355); // if
    return false;
  }

  // This is not perfect, but as we're just verifying here, we can live with
  // the loss of precision. The real solution would be that of doing strongly
  // connected component finding in this routine, and it's probably not worth
  // the complexity for the time being. So, we just keep a set of visited
  // MemoryAccess and return true when we hit a cycle.
  if (Visited.count(First)) {
    PassPrediction::PassPeeper(3356); // if
    return true;
  }
  Visited.insert(First);

  const auto *EndDef = First;
  for (auto *ChainDef : optimized_def_chain(First)) {
    PassPrediction::PassPeeper(3357); // for-range
    if (ChainDef == Second) {
      PassPrediction::PassPeeper(3358); // if
      return true;
    }
    if (MSSA->isLiveOnEntryDef(ChainDef)) {
      PassPrediction::PassPeeper(3359); // if
      return false;
    }
    EndDef = ChainDef;
  }
  auto *MP = cast<MemoryPhi>(EndDef);
  auto ReachableOperandPred = [&](const Use &U) {
    return ReachableEdges.count({MP->getIncomingBlock(U), MP->getBlock()});
  };
  auto FilteredPhiArgs =
      make_filter_range(MP->operands(), ReachableOperandPred);
  SmallVector<const Value *, 32> OperandList;
  std::copy(FilteredPhiArgs.begin(), FilteredPhiArgs.end(),
            std::back_inserter(OperandList));
  bool Okay = OperandList.size() == 1;
  if (!Okay) {
    PassPrediction::PassPeeper(3360); // if
    Okay =
        std::equal(OperandList.begin(), OperandList.end(), OperandList.begin());
  }
  if (Okay) {
    PassPrediction::PassPeeper(3361); // if
    return singleReachablePHIPath(Visited, cast<MemoryAccess>(OperandList[0]),
                                  Second);
  }
  return false;
}

// Verify the that the memory equivalence table makes sense relative to the
// congruence classes.  Note that this checking is not perfect, and is currently
// subject to very rare false negatives. It is only useful for
// testing/debugging.
void NewGVN::verifyMemoryCongruency() const {
#ifndef NDEBUG
  // Verify that the memory table equivalence and memory member set match
  for (const auto *CC : CongruenceClasses) {
    PassPrediction::PassPeeper(3362); // for-range
    if (CC == TOPClass || CC->isDead()) {
      PassPrediction::PassPeeper(3363); // if
      continue;
    }
    if (CC->getStoreCount() != 0) {
      assert((CC->getStoredValue() || !isa<StoreInst>(CC->getLeader())) &&
             "Any class with a store as a leader should have a "
             "representative stored value");
      assert(CC->getMemoryLeader() &&
             "Any congruence class with a store should have a "
             "representative access");
    }

    if (CC->getMemoryLeader()) {
      assert(MemoryAccessToClass.lookup(CC->getMemoryLeader()) == CC &&
             "Representative MemoryAccess does not appear to be reverse "
             "mapped properly");
    }
    for (auto M : CC->memory()) {
      assert(MemoryAccessToClass.lookup(M) == CC &&
             "Memory member does not appear to be reverse mapped properly");
    }
  }

  // Anything equivalent in the MemoryAccess table should be in the same
  // congruence class.

  // Filter out the unreachable and trivially dead entries, because they may
  // never have been updated if the instructions were not processed.
  auto ReachableAccessPred =
      [&](const std::pair<const MemoryAccess *, CongruenceClass *> Pair) {
        bool Result = ReachableBlocks.count(Pair.first->getBlock());
        if (!Result || MSSA->isLiveOnEntryDef(Pair.first) ||
            MemoryToDFSNum(Pair.first) == 0) {
          PassPrediction::PassPeeper(3364); // if
          return false;
        }
        if (auto *MemDef = dyn_cast<MemoryDef>(Pair.first)) {
          PassPrediction::PassPeeper(3365); // if
          return !isInstructionTriviallyDead(MemDef->getMemoryInst());
        }

        // We could have phi nodes which operands are all trivially dead,
        // so we don't process them.
        if (auto *MemPHI = dyn_cast<MemoryPhi>(Pair.first)) {
          PassPrediction::PassPeeper(3366); // if
          for (auto &U : MemPHI->incoming_values()) {
            PassPrediction::PassPeeper(3367); // for-range
            if (Instruction *I = dyn_cast<Instruction>(U.get())) {
              PassPrediction::PassPeeper(3368); // if
              if (!isInstructionTriviallyDead(I)) {
                PassPrediction::PassPeeper(3369); // if
                return true;
              }
            }
          }
          return false;
        }

        return true;
      };

  auto Filtered = make_filter_range(MemoryAccessToClass, ReachableAccessPred);
  for (auto KV : Filtered) {
    PassPrediction::PassPeeper(3370); // for-range
    if (auto *FirstMUD = dyn_cast<MemoryUseOrDef>(KV.first)) {
      PassPrediction::PassPeeper(3371); // if
      auto *SecondMUD = dyn_cast<MemoryUseOrDef>(KV.second->getMemoryLeader());
      if (FirstMUD && SecondMUD) {
        PassPrediction::PassPeeper(3372); // if
        SmallPtrSet<const MemoryAccess *, 8> VisitedMAS;
        assert((singleReachablePHIPath(VisitedMAS, FirstMUD, SecondMUD) ||
                ValueToClass.lookup(FirstMUD->getMemoryInst()) ==
                    ValueToClass.lookup(SecondMUD->getMemoryInst())) &&
               "The instructions for these memory operations should have "
               "been in the same congruence class or reachable through"
               "a single argument phi");
      }
    } else if (auto *FirstMP = dyn_cast<MemoryPhi>(KV.first)) {
      // We can only sanely verify that MemoryDefs in the operand list all have
      // the same class.
      PassPrediction::PassPeeper(3373); // if
      auto ReachableOperandPred = [&](const Use &U) {
        return ReachableEdges.count(
                   {FirstMP->getIncomingBlock(U), FirstMP->getBlock()}) &&
               isa<MemoryDef>(U);

      };
      // All arguments should in the same class, ignoring unreachable arguments
      auto FilteredPhiArgs =
          make_filter_range(FirstMP->operands(), ReachableOperandPred);
      SmallVector<const CongruenceClass *, 16> PhiOpClasses;
      std::transform(FilteredPhiArgs.begin(), FilteredPhiArgs.end(),
                     std::back_inserter(PhiOpClasses), [&](const Use &U) {
                       const MemoryDef *MD = cast<MemoryDef>(U);
                       return ValueToClass.lookup(MD->getMemoryInst());
                     });
      assert(std::equal(PhiOpClasses.begin(), PhiOpClasses.end(),
                        PhiOpClasses.begin()) &&
             "All MemoryPhi arguments should be in the same class");
    }
  }
#endif
}

// Verify that the sparse propagation we did actually found the maximal fixpoint
// We do this by storing the value to class mapping, touching all instructions,
// and redoing the iteration to see if anything changed.
void NewGVN::verifyIterationSettled(Function &F) {
#ifndef NDEBUG
  DEBUG(dbgs() << "Beginning iteration verification\n");
  if (DebugCounter::isCounterSet(VNCounter)) {
    PassPrediction::PassPeeper(3374); // if
    DebugCounter::setCounterValue(VNCounter, StartingVNCounter);
  }

  // Note that we have to store the actual classes, as we may change existing
  // classes during iteration.  This is because our memory iteration propagation
  // is not perfect, and so may waste a little work.  But it should generate
  // exactly the same congruence classes we have now, with different IDs.
  std::map<const Value *, CongruenceClass> BeforeIteration;

  for (auto &KV : ValueToClass) {
    PassPrediction::PassPeeper(3375); // for-range
    if (auto *I = dyn_cast<Instruction>(KV.first)) {
      // Skip unused/dead instructions.
      PassPrediction::PassPeeper(3376); // if
      if (InstrToDFSNum(I) == 0) {
        PassPrediction::PassPeeper(3377); // if
        continue;
      }
    }
    BeforeIteration.insert({KV.first, *KV.second});
  }

  TouchedInstructions.set();
  TouchedInstructions.reset(0);
  iterateTouchedInstructions();
  DenseSet<std::pair<const CongruenceClass *, const CongruenceClass *>>
      EqualClasses;
  for (const auto &KV : ValueToClass) {
    PassPrediction::PassPeeper(3378); // for-range
    if (auto *I = dyn_cast<Instruction>(KV.first)) {
      // Skip unused/dead instructions.
      PassPrediction::PassPeeper(3379); // if
      if (InstrToDFSNum(I) == 0) {
        PassPrediction::PassPeeper(3380); // if
        continue;
      }
    }
    // We could sink these uses, but i think this adds a bit of clarity here as
    // to what we are comparing.
    auto *BeforeCC = &BeforeIteration.find(KV.first)->second;
    auto *AfterCC = KV.second;
    // Note that the classes can't change at this point, so we memoize the set
    // that are equal.
    if (!EqualClasses.count({BeforeCC, AfterCC})) {
      assert(BeforeCC->isEquivalentTo(AfterCC) &&
             "Value number changed after main loop completed!");
      EqualClasses.insert({BeforeCC, AfterCC});
    }
  }
#endif
}

// Verify that for each store expression in the expression to class mapping,
// only the latest appears, and multiple ones do not appear.
// Because loads do not use the stored value when doing equality with stores,
// if we don't erase the old store expressions from the table, a load can find
// a no-longer valid StoreExpression.
void NewGVN::verifyStoreExpressions() const {
#ifndef NDEBUG
  // This is the only use of this, and it's not worth defining a complicated
  // densemapinfo hash/equality function for it.
  std::set<
      std::pair<const Value *,
                std::tuple<const Value *, const CongruenceClass *, Value *>>>
      StoreExpressionSet;
  for (const auto &KV : ExpressionToClass) {
    PassPrediction::PassPeeper(3381); // for-range
    if (auto *SE = dyn_cast<StoreExpression>(KV.first)) {
      // Make sure a version that will conflict with loads is not already there
      PassPrediction::PassPeeper(3382); // if
      auto Res = StoreExpressionSet.insert(
          {SE->getOperand(0), std::make_tuple(SE->getMemoryLeader(), KV.second,
                                              SE->getStoredValue())});
      bool Okay = Res.second;
      // It's okay to have the same expression already in there if it is
      // identical in nature.
      // This can happen when the leader of the stored value changes over time.
      if (!Okay) {
        PassPrediction::PassPeeper(3383); // if
        Okay = (std::get<1>(Res.first->second) == KV.second) &&
               (lookupOperandLeader(std::get<2>(Res.first->second)) ==
                lookupOperandLeader(SE->getStoredValue()));
      }
      assert(Okay && "Stored expression conflict exists in expression table");
      auto *ValueExpr = ValueToExpression.lookup(SE->getStoreInst());
      assert(ValueExpr && ValueExpr->equals(*SE) &&
             "StoreExpression in ExpressionToClass is not latest "
             "StoreExpression for value");
    }
  }
#endif
}

// This is the main value numbering loop, it iterates over the initial touched
// instruction set, propagating value numbers, marking things touched, etc,
// until the set of touched instructions is completely empty.
void NewGVN::iterateTouchedInstructions() {
  unsigned int Iterations = 0;
  // Figure out where touchedinstructions starts
  int FirstInstr = TouchedInstructions.find_first();
  // Nothing set, nothing to iterate, just return.
  if (FirstInstr == -1) {
    PassPrediction::PassPeeper(3384); // if
    return;
  }
  const BasicBlock *LastBlock = getBlockForValue(InstrFromDFSNum(FirstInstr));
  while (TouchedInstructions.any()) {
    PassPrediction::PassPeeper(3385); // while
    ++Iterations;
    // Walk through all the instructions in all the blocks in RPO.
    // TODO: As we hit a new block, we should push and pop equalities into a
    // table lookupOperandLeader can use, to catch things PredicateInfo
    // might miss, like edge-only equivalences.
    for (unsigned InstrNum : TouchedInstructions.set_bits()) {

      // This instruction was found to be dead. We don't bother looking
      // at it again.
      PassPrediction::PassPeeper(3386); // for-range
      if (InstrNum == 0) {
        PassPrediction::PassPeeper(3387); // if
        TouchedInstructions.reset(InstrNum);
        continue;
      }

      Value *V = InstrFromDFSNum(InstrNum);
      const BasicBlock *CurrBlock = getBlockForValue(V);

      // If we hit a new block, do reachability processing.
      if (CurrBlock != LastBlock) {
        PassPrediction::PassPeeper(3388); // if
        LastBlock = CurrBlock;
        bool BlockReachable = ReachableBlocks.count(CurrBlock);
        const auto &CurrInstRange = BlockInstRange.lookup(CurrBlock);

        // If it's not reachable, erase any touched instructions and move on.
        if (!BlockReachable) {
          PassPrediction::PassPeeper(3389); // if
          TouchedInstructions.reset(CurrInstRange.first, CurrInstRange.second);
          DEBUG(dbgs() << "Skipping instructions in block "
                       << getBlockName(CurrBlock)
                       << " because it is unreachable\n");
          continue;
        }
        updateProcessedCount(CurrBlock);
      }
      // Reset after processing (because we may mark ourselves as touched when
      // we propagate equalities).
      TouchedInstructions.reset(InstrNum);

      if (auto *MP = dyn_cast<MemoryPhi>(V)) {
        DEBUG(dbgs() << "Processing MemoryPhi " << *MP << "\n");
        valueNumberMemoryPhi(MP);
      } else if (auto *I = dyn_cast<Instruction>(V)) {
        PassPrediction::PassPeeper(3390); // if
        valueNumberInstruction(I);
      } else {
        llvm_unreachable("Should have been a MemoryPhi or Instruction");
      }
      updateProcessedCount(V);
    }
  }
  NumGVNMaxIterations = std::max(NumGVNMaxIterations.getValue(), Iterations);
}

// This is the main transformation entry point.
bool NewGVN::runGVN() {
  if (DebugCounter::isCounterSet(VNCounter)) {
    PassPrediction::PassPeeper(3391); // if
    StartingVNCounter = DebugCounter::getCounterValue(VNCounter);
  }
  bool Changed = false;
  NumFuncArgs = F.arg_size();
  MSSAWalker = MSSA->getWalker();
  SingletonDeadExpression = new (ExpressionAllocator) DeadExpression();

  // Count number of instructions for sizing of hash tables, and come
  // up with a global dfs numbering for instructions.
  unsigned ICount = 1;
  // Add an empty instruction to account for the fact that we start at 1
  DFSToInstr.emplace_back(nullptr);
  // Note: We want ideal RPO traversal of the blocks, which is not quite the
  // same as dominator tree order, particularly with regard whether backedges
  // get visited first or second, given a block with multiple successors.
  // If we visit in the wrong order, we will end up performing N times as many
  // iterations.
  // The dominator tree does guarantee that, for a given dom tree node, it's
  // parent must occur before it in the RPO ordering. Thus, we only need to sort
  // the siblings.
  ReversePostOrderTraversal<Function *> RPOT(&F);
  unsigned Counter = 0;
  for (auto &B : RPOT) {
    PassPrediction::PassPeeper(3392); // for-range
    auto *Node = DT->getNode(B);
    assert(Node && "RPO and Dominator tree should have same reachability");
    RPOOrdering[Node] = ++Counter;
  }
  // Sort dominator tree children arrays into RPO.
  for (auto &B : RPOT) {
    PassPrediction::PassPeeper(3393); // for-range
    auto *Node = DT->getNode(B);
    if (Node->getChildren().size() > 1) {
      PassPrediction::PassPeeper(3394); // if
      std::sort(Node->begin(), Node->end(),
                [&](const DomTreeNode *A, const DomTreeNode *B) {
                  return RPOOrdering[A] < RPOOrdering[B];
                });
    }
  }

  // Now a standard depth first ordering of the domtree is equivalent to RPO.
  for (auto DTN : depth_first(DT->getRootNode())) {
    PassPrediction::PassPeeper(3395); // for-range
    BasicBlock *B = DTN->getBlock();
    const auto &BlockRange = assignDFSNumbers(B, ICount);
    BlockInstRange.insert({B, BlockRange});
    ICount += BlockRange.second - BlockRange.first;
  }
  initializeCongruenceClasses(F);

  TouchedInstructions.resize(ICount);
  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  ExpressionToClass.reserve(ICount);

  // Initialize the touched instructions to include the entry block.
  const auto &InstRange = BlockInstRange.lookup(&F.getEntryBlock());
  TouchedInstructions.set(InstRange.first, InstRange.second);
  DEBUG(dbgs() << "Block " << getBlockName(&F.getEntryBlock())
               << " marked reachable\n");
  ReachableBlocks.insert(&F.getEntryBlock());

  iterateTouchedInstructions();
  verifyMemoryCongruency();
  verifyIterationSettled(F);
  verifyStoreExpressions();

  Changed |= eliminateInstructions(F);

  // Delete all instructions marked for deletion.
  for (Instruction *ToErase : InstructionsToErase) {
    PassPrediction::PassPeeper(3396); // for-range
    if (!ToErase->use_empty()) {
      PassPrediction::PassPeeper(3397); // if
      ToErase->replaceAllUsesWith(UndefValue::get(ToErase->getType()));
    }

    if (ToErase->getParent()) {
      PassPrediction::PassPeeper(3398); // if
      ToErase->eraseFromParent();
    }
  }

  // Delete all unreachable blocks.
  auto UnreachableBlockPred = [&](const BasicBlock &BB) {
    return !ReachableBlocks.count(&BB);
  };

  for (auto &BB : make_filter_range(F, UnreachableBlockPred)) {
    DEBUG(dbgs() << "We believe block " << getBlockName(&BB)
                 << " is unreachable\n");
    deleteInstructionsInBlock(&BB);
    Changed = true;
  }

  cleanupTables();
  return Changed;
}

struct NewGVN::ValueDFS {
  int DFSIn = 0;
  int DFSOut = 0;
  int LocalNum = 0;
  // Only one of Def and U will be set.
  // The bool in the Def tells us whether the Def is the stored value of a
  // store.
  PointerIntPair<Value *, 1, bool> Def;
  Use *U = nullptr;
  bool operator<(const ValueDFS &Other) const {
    // It's not enough that any given field be less than - we have sets
    // of fields that need to be evaluated together to give a proper ordering.
    // For example, if you have;
    // DFS (1, 3)
    // Val 0
    // DFS (1, 2)
    // Val 50
    // We want the second to be less than the first, but if we just go field
    // by field, we will get to Val 0 < Val 50 and say the first is less than
    // the second. We only want it to be less than if the DFS orders are equal.
    //
    // Each LLVM instruction only produces one value, and thus the lowest-level
    // differentiator that really matters for the stack (and what we use as as a
    // replacement) is the local dfs number.
    // Everything else in the structure is instruction level, and only affects
    // the order in which we will replace operands of a given instruction.
    //
    // For a given instruction (IE things with equal dfsin, dfsout, localnum),
    // the order of replacement of uses does not matter.
    // IE given,
    //  a = 5
    //  b = a + a
    // When you hit b, you will have two valuedfs with the same dfsin, out, and
    // localnum.
    // The .val will be the same as well.
    // The .u's will be different.
    // You will replace both, and it does not matter what order you replace them
    // in (IE whether you replace operand 2, then operand 1, or operand 1, then
    // operand 2).
    // Similarly for the case of same dfsin, dfsout, localnum, but different
    // .val's
    //  a = 5
    //  b  = 6
    //  c = a + b
    // in c, we will a valuedfs for a, and one for b,with everything the same
    // but .val  and .u.
    // It does not matter what order we replace these operands in.
    // You will always end up with the same IR, and this is guaranteed.
    return std::tie(DFSIn, DFSOut, LocalNum, Def, U) <
           std::tie(Other.DFSIn, Other.DFSOut, Other.LocalNum, Other.Def,
                    Other.U);
  }
};

// This function converts the set of members for a congruence class from values,
// to sets of defs and uses with associated DFS info.  The total number of
// reachable uses for each value is stored in UseCount, and instructions that
// seem
// dead (have no non-dead uses) are stored in ProbablyDead.
void NewGVN::convertClassToDFSOrdered(
    const CongruenceClass &Dense, SmallVectorImpl<ValueDFS> &DFSOrderedSet,
    DenseMap<const Value *, unsigned int> &UseCounts,
    SmallPtrSetImpl<Instruction *> &ProbablyDead) const {
  for (auto D : Dense) {
    // First add the value.
    PassPrediction::PassPeeper(3399); // for-range
    BasicBlock *BB = getBlockForValue(D);
    // Constants are handled prior to ever calling this function, so
    // we should only be left with instructions as members.
    assert(BB && "Should have figured out a basic block for value");
    ValueDFS VDDef;
    DomTreeNode *DomNode = DT->getNode(BB);
    VDDef.DFSIn = DomNode->getDFSNumIn();
    VDDef.DFSOut = DomNode->getDFSNumOut();
    // If it's a store, use the leader of the value operand, if it's always
    // available, or the value operand.  TODO: We could do dominance checks to
    // find a dominating leader, but not worth it ATM.
    if (auto *SI = dyn_cast<StoreInst>(D)) {
      PassPrediction::PassPeeper(3400); // if
      auto Leader = lookupOperandLeader(SI->getValueOperand());
      if (alwaysAvailable(Leader)) {
        PassPrediction::PassPeeper(3402); // if
        VDDef.Def.setPointer(Leader);
      } else {
        PassPrediction::PassPeeper(3403); // else
        VDDef.Def.setPointer(SI->getValueOperand());
        VDDef.Def.setInt(true);
      }
    } else {
      PassPrediction::PassPeeper(3401); // else
      VDDef.Def.setPointer(D);
    }
    assert(isa<Instruction>(D) &&
           "The dense set member should always be an instruction");
    Instruction *Def = cast<Instruction>(D);
    VDDef.LocalNum = InstrToDFSNum(D);
    DFSOrderedSet.push_back(VDDef);
    // If there is a phi node equivalent, add it
    if (auto *PN = RealToTemp.lookup(Def)) {
      PassPrediction::PassPeeper(3404); // if
      auto *PHIE =
          dyn_cast_or_null<PHIExpression>(ValueToExpression.lookup(Def));
      if (PHIE) {
        PassPrediction::PassPeeper(3405); // if
        VDDef.Def.setInt(false);
        VDDef.Def.setPointer(PN);
        VDDef.LocalNum = 0;
        DFSOrderedSet.push_back(VDDef);
      }
    }

    unsigned int UseCount = 0;
    // Now add the uses.
    for (auto &U : Def->uses()) {
      PassPrediction::PassPeeper(3406); // for-range
      if (auto *I = dyn_cast<Instruction>(U.getUser())) {
        // Don't try to replace into dead uses
        PassPrediction::PassPeeper(3407); // if
        if (InstructionsToErase.count(I)) {
          PassPrediction::PassPeeper(3408); // if
          continue;
        }
        ValueDFS VDUse;
        // Put the phi node uses in the incoming block.
        BasicBlock *IBlock;
        if (auto *P = dyn_cast<PHINode>(I)) {
          PassPrediction::PassPeeper(3409); // if
          IBlock = P->getIncomingBlock(U);
          // Make phi node users appear last in the incoming block
          // they are from.
          VDUse.LocalNum = InstrDFS.size() + 1;
        } else {
          PassPrediction::PassPeeper(3410); // else
          IBlock = getBlockForValue(I);
          VDUse.LocalNum = InstrToDFSNum(I);
        }

        // Skip uses in unreachable blocks, as we're going
        // to delete them.
        if (ReachableBlocks.count(IBlock) == 0) {
          PassPrediction::PassPeeper(3411); // if
          continue;
        }

        DomTreeNode *DomNode = DT->getNode(IBlock);
        VDUse.DFSIn = DomNode->getDFSNumIn();
        VDUse.DFSOut = DomNode->getDFSNumOut();
        VDUse.U = &U;
        ++UseCount;
        DFSOrderedSet.emplace_back(VDUse);
      }
    }

    // If there are no uses, it's probably dead (but it may have side-effects,
    // so not definitely dead. Otherwise, store the number of uses so we can
    // track if it becomes dead later).
    if (UseCount == 0) {
      PassPrediction::PassPeeper(3412); // if
      ProbablyDead.insert(Def);
    } else {
      PassPrediction::PassPeeper(3413); // else
      UseCounts[Def] = UseCount;
    }
  }
}

// This function converts the set of members for a congruence class from values,
// to the set of defs for loads and stores, with associated DFS info.
void NewGVN::convertClassToLoadsAndStores(
    const CongruenceClass &Dense,
    SmallVectorImpl<ValueDFS> &LoadsAndStores) const {
  for (auto D : Dense) {
    PassPrediction::PassPeeper(3414); // for-range
    if (!isa<LoadInst>(D) && !isa<StoreInst>(D)) {
      PassPrediction::PassPeeper(3415); // if
      continue;
    }

    BasicBlock *BB = getBlockForValue(D);
    ValueDFS VD;
    DomTreeNode *DomNode = DT->getNode(BB);
    VD.DFSIn = DomNode->getDFSNumIn();
    VD.DFSOut = DomNode->getDFSNumOut();
    VD.Def.setPointer(D);

    // If it's an instruction, use the real local dfs number.
    if (auto *I = dyn_cast<Instruction>(D)) {
      PassPrediction::PassPeeper(3416); // if
      VD.LocalNum = InstrToDFSNum(I);
    } else {
      llvm_unreachable("Should have been an instruction");
    }

    LoadsAndStores.emplace_back(VD);
  }
}

static void patchReplacementInstruction(Instruction *I, Value *Repl) {
  auto *ReplInst = dyn_cast<Instruction>(Repl);
  if (!ReplInst) {
    PassPrediction::PassPeeper(3417); // if
    return;
  }

  // Patch the replacement so that it is not more restrictive than the value
  // being replaced.
  // Note that if 'I' is a load being replaced by some operation,
  // for example, by an arithmetic operation, then andIRFlags()
  // would just erase all math flags from the original arithmetic
  // operation, which is clearly not wanted and not needed.
  if (!isa<LoadInst>(I)) {
    PassPrediction::PassPeeper(3418); // if
    ReplInst->andIRFlags(I);
  }

  // FIXME: If both the original and replacement value are part of the
  // same control-flow region (meaning that the execution of one
  // guarantees the execution of the other), then we can combine the
  // noalias scopes here and do better than the general conservative
  // answer used in combineMetadata().

  // In general, GVN unifies expressions over different control-flow
  // regions, and so we need a conservative combination of the noalias
  // scopes.
  static const unsigned KnownIDs[] = {
      LLVMContext::MD_tbaa,           LLVMContext::MD_alias_scope,
      LLVMContext::MD_noalias,        LLVMContext::MD_range,
      LLVMContext::MD_fpmath,         LLVMContext::MD_invariant_load,
      LLVMContext::MD_invariant_group};
  combineMetadata(ReplInst, I, KnownIDs);
}

static void patchAndReplaceAllUsesWith(Instruction *I, Value *Repl) {
  patchReplacementInstruction(I, Repl);
  I->replaceAllUsesWith(Repl);
}

void NewGVN::deleteInstructionsInBlock(BasicBlock *BB) {
  DEBUG(dbgs() << "  BasicBlock Dead:" << *BB);
  ++NumGVNBlocksDeleted;

  // Delete the instructions backwards, as it has a reduced likelihood of having
  // to update as many def-use and use-def chains. Start after the terminator.
  auto StartPoint = BB->rbegin();
  ++StartPoint;
  // Note that we explicitly recalculate BB->rend() on each iteration,
  // as it may change when we remove the first instruction.
  for (BasicBlock::reverse_iterator I(StartPoint); I != BB->rend();) {
    PassPrediction::PassPeeper(3419); // for
    Instruction &Inst = *I++;
    if (!Inst.use_empty()) {
      PassPrediction::PassPeeper(3420); // if
      Inst.replaceAllUsesWith(UndefValue::get(Inst.getType()));
    }
    if (isa<LandingPadInst>(Inst)) {
      PassPrediction::PassPeeper(3421); // if
      continue;
    }

    Inst.eraseFromParent();
    ++NumGVNInstrDeleted;
  }
  // Now insert something that simplifycfg will turn into an unreachable.
  Type *Int8Ty = Type::getInt8Ty(BB->getContext());
  new StoreInst(UndefValue::get(Int8Ty),
                Constant::getNullValue(Int8Ty->getPointerTo()),
                BB->getTerminator());
}

void NewGVN::markInstructionForDeletion(Instruction *I) {
  DEBUG(dbgs() << "Marking " << *I << " for deletion\n");
  InstructionsToErase.insert(I);
}

void NewGVN::replaceInstruction(Instruction *I, Value *V) {

  DEBUG(dbgs() << "Replacing " << *I << " with " << *V << "\n");
  patchAndReplaceAllUsesWith(I, V);
  // We save the actual erasing to avoid invalidating memory
  // dependencies until we are done with everything.
  markInstructionForDeletion(I);
}

namespace {

// This is a stack that contains both the value and dfs info of where
// that value is valid.
class ValueDFSStack {
public:
  Value *back() const { return ValueStack.back(); }
  std::pair<int, int> dfs_back() const { return DFSStack.back(); }

  void push_back(Value *V, int DFSIn, int DFSOut) {
    ValueStack.emplace_back(V);
    DFSStack.emplace_back(DFSIn, DFSOut);
  }
  bool empty() const { return DFSStack.empty(); }
  bool isInScope(int DFSIn, int DFSOut) const {
    if (empty()) {
      PassPrediction::PassPeeper(3422); // if
      return false;
    }
    return DFSIn >= DFSStack.back().first && DFSOut <= DFSStack.back().second;
  }

  void popUntilDFSScope(int DFSIn, int DFSOut) {

    // These two should always be in sync at this point.
    assert(ValueStack.size() == DFSStack.size() &&
           "Mismatch between ValueStack and DFSStack");
    while (!DFSStack.empty() && !(DFSIn >= DFSStack.back().first &&
                                  DFSOut <= DFSStack.back().second)) {
      PassPrediction::PassPeeper(3423); // while
      DFSStack.pop_back();
      ValueStack.pop_back();
    }
  }

private:
  SmallVector<Value *, 8> ValueStack;
  SmallVector<std::pair<int, int>, 8> DFSStack;
};
} // namespace

// Given a value and a basic block we are trying to see if it is available in,
// see if the value has a leader available in that block.
Value *NewGVN::findPhiOfOpsLeader(const Expression *E,
                                  const BasicBlock *BB) const {
  // It would already be constant if we could make it constant
  if (auto *CE = dyn_cast<ConstantExpression>(E)) {
    PassPrediction::PassPeeper(3424); // if
    return CE->getConstantValue();
  }
  if (auto *VE = dyn_cast<VariableExpression>(E)) {
    PassPrediction::PassPeeper(3425); // if
    return VE->getVariableValue();
  }

  auto *CC = ExpressionToClass.lookup(E);
  if (!CC) {
    PassPrediction::PassPeeper(3426); // if
    return nullptr;
  }
  if (alwaysAvailable(CC->getLeader())) {
    PassPrediction::PassPeeper(3427); // if
    return CC->getLeader();
  }

  for (auto Member : *CC) {
    PassPrediction::PassPeeper(3428); // for-range
    auto *MemberInst = dyn_cast<Instruction>(Member);
    // Anything that isn't an instruction is always available.
    if (!MemberInst) {
      PassPrediction::PassPeeper(3429); // if
      return Member;
    }
    // If we are looking for something in the same block as the member, it must
    // be a leader because this function is looking for operands for a phi node.
    if (MemberInst->getParent() == BB ||
        DT->dominates(MemberInst->getParent(), BB)) {
      PassPrediction::PassPeeper(3430); // if
      return Member;
    }
  }
  return nullptr;
}

bool NewGVN::eliminateInstructions(Function &F) {
  // This is a non-standard eliminator. The normal way to eliminate is
  // to walk the dominator tree in order, keeping track of available
  // values, and eliminating them.  However, this is mildly
  // pointless. It requires doing lookups on every instruction,
  // regardless of whether we will ever eliminate it.  For
  // instructions part of most singleton congruence classes, we know we
  // will never eliminate them.

  // Instead, this eliminator looks at the congruence classes directly, sorts
  // them into a DFS ordering of the dominator tree, and then we just
  // perform elimination straight on the sets by walking the congruence
  // class member uses in order, and eliminate the ones dominated by the
  // last member.   This is worst case O(E log E) where E = number of
  // instructions in a single congruence class.  In theory, this is all
  // instructions.   In practice, it is much faster, as most instructions are
  // either in singleton congruence classes or can't possibly be eliminated
  // anyway (if there are no overlapping DFS ranges in class).
  // When we find something not dominated, it becomes the new leader
  // for elimination purposes.
  // TODO: If we wanted to be faster, We could remove any members with no
  // overlapping ranges while sorting, as we will never eliminate anything
  // with those members, as they don't dominate anything else in our set.

  bool AnythingReplaced = false;

  // Since we are going to walk the domtree anyway, and we can't guarantee the
  // DFS numbers are updated, we compute some ourselves.
  DT->updateDFSNumbers();

  // Go through all of our phi nodes, and kill the arguments associated with
  // unreachable edges.
  auto ReplaceUnreachablePHIArgs = [&](PHINode &PHI, BasicBlock *BB) {
    for (auto &Operand : PHI.incoming_values()) {
      PassPrediction::PassPeeper(3431); // for-range
      if (!ReachableEdges.count({PHI.getIncomingBlock(Operand), BB})) {
        DEBUG(dbgs() << "Replacing incoming value of " << PHI << " for block "
                     << getBlockName(PHI.getIncomingBlock(Operand))
                     << " with undef due to it being unreachable\n");
        Operand.set(UndefValue::get(PHI.getType()));
      }
    }
  };
  SmallPtrSet<BasicBlock *, 8> BlocksWithPhis;
  for (auto &B : F) {
    PassPrediction::PassPeeper(3432); // for-range
    if ((!B.empty() && isa<PHINode>(*B.begin())) ||
        (PHIOfOpsPHIs.find(&B) != PHIOfOpsPHIs.end())) {
      PassPrediction::PassPeeper(3433); // if
      BlocksWithPhis.insert(&B);
    }
  }
  DenseMap<const BasicBlock *, unsigned> ReachablePredCount;
  for (auto KV : ReachableEdges) {
    PassPrediction::PassPeeper(3434); // for-range
    ReachablePredCount[KV.getEnd()]++;
  }
  for (auto *BB : BlocksWithPhis) {
    // TODO: It would be faster to use getNumIncomingBlocks() on a phi node in
    // the block and subtract the pred count, but it's more complicated.
    PassPrediction::PassPeeper(3435); // for-range
    if (ReachablePredCount.lookup(BB) !=
        unsigned(std::distance(pred_begin(BB), pred_end(BB)))) {
      PassPrediction::PassPeeper(3436); // if
      for (auto II = BB->begin(); isa<PHINode>(II); ++II) {
        PassPrediction::PassPeeper(3437); // for
        auto &PHI = cast<PHINode>(*II);
        ReplaceUnreachablePHIArgs(PHI, BB);
      }
      for_each_found(PHIOfOpsPHIs, BB, [&](PHINode *PHI) {
        ReplaceUnreachablePHIArgs(*PHI, BB);
      });
    }
  }

  // Map to store the use counts
  DenseMap<const Value *, unsigned int> UseCounts;
  for (auto *CC : reverse(CongruenceClasses)) {
    DEBUG(dbgs() << "Eliminating in congruence class " << CC->getID() << "\n");
    // Track the equivalent store info so we can decide whether to try
    // dead store elimination.
    SmallVector<ValueDFS, 8> PossibleDeadStores;
    SmallPtrSet<Instruction *, 8> ProbablyDead;
    if (CC->isDead() || CC->empty()) {
      PassPrediction::PassPeeper(3438); // if
      continue;
    }
    // Everything still in the TOP class is unreachable or dead.
    if (CC == TOPClass) {
      PassPrediction::PassPeeper(3439); // if
      for (auto M : *CC) {
        PassPrediction::PassPeeper(3440); // for-range
        auto *VTE = ValueToExpression.lookup(M);
        if (VTE && isa<DeadExpression>(VTE)) {
          PassPrediction::PassPeeper(3441); // if
          markInstructionForDeletion(cast<Instruction>(M));
        }
        assert((!ReachableBlocks.count(cast<Instruction>(M)->getParent()) ||
                InstructionsToErase.count(cast<Instruction>(M))) &&
               "Everything in TOP should be unreachable or dead at this "
               "point");
      }
      continue;
    }

    assert(CC->getLeader() && "We should have had a leader");
    // If this is a leader that is always available, and it's a
    // constant or has no equivalences, just replace everything with
    // it. We then update the congruence class with whatever members
    // are left.
    Value *Leader =
        CC->getStoredValue() ? CC->getStoredValue() : CC->getLeader();
    if (alwaysAvailable(Leader)) {
      PassPrediction::PassPeeper(3442); // if
      CongruenceClass::MemberSet MembersLeft;
      for (auto M : *CC) {
        PassPrediction::PassPeeper(3444); // for-range
        Value *Member = M;
        // Void things have no uses we can replace.
        if (Member == Leader || !isa<Instruction>(Member) ||
            Member->getType()->isVoidTy()) {
          PassPrediction::PassPeeper(3445); // if
          MembersLeft.insert(Member);
          continue;
        }
        DEBUG(dbgs() << "Found replacement " << *(Leader) << " for " << *Member
                     << "\n");
        auto *I = cast<Instruction>(Member);
        assert(Leader != I && "About to accidentally remove our leader");
        replaceInstruction(I, Leader);
        AnythingReplaced = true;
      }
      CC->swap(MembersLeft);
    } else {
      // If this is a singleton, we can skip it.
      PassPrediction::PassPeeper(3443); // else
      if (CC->size() != 1 || RealToTemp.lookup(Leader)) {
        // This is a stack because equality replacement/etc may place
        // constants in the middle of the member list, and we want to use
        // those constant values in preference to the current leader, over
        // the scope of those constants.
        PassPrediction::PassPeeper(3446); // if
        ValueDFSStack EliminationStack;

        // Convert the members to DFS ordered sets and then merge them.
        SmallVector<ValueDFS, 8> DFSOrderedSet;
        convertClassToDFSOrdered(*CC, DFSOrderedSet, UseCounts, ProbablyDead);

        // Sort the whole thing.
        std::sort(DFSOrderedSet.begin(), DFSOrderedSet.end());
        for (auto &VD : DFSOrderedSet) {
          PassPrediction::PassPeeper(3447); // for-range
          int MemberDFSIn = VD.DFSIn;
          int MemberDFSOut = VD.DFSOut;
          Value *Def = VD.Def.getPointer();
          bool FromStore = VD.Def.getInt();
          Use *U = VD.U;
          // We ignore void things because we can't get a value from them.
          if (Def && Def->getType()->isVoidTy()) {
            PassPrediction::PassPeeper(3448); // if
            continue;
          }
          auto *DefInst = dyn_cast_or_null<Instruction>(Def);
          if (DefInst && AllTempInstructions.count(DefInst)) {
            PassPrediction::PassPeeper(3449); // if
            auto *PN = cast<PHINode>(DefInst);

            // If this is a value phi and that's the expression we used, insert
            // it into the program
            // remove from temp instruction list.
            AllTempInstructions.erase(PN);
            auto *DefBlock = getBlockForValue(Def);
            DEBUG(dbgs() << "Inserting fully real phi of ops" << *Def
                         << " into block "
                         << getBlockName(getBlockForValue(Def)) << "\n");
            PN->insertBefore(&DefBlock->front());
            Def = PN;
            NumGVNPHIOfOpsEliminations++;
          }

          if (EliminationStack.empty()) {
            DEBUG(dbgs() << "Elimination Stack is empty\n");
          } else {
            DEBUG(dbgs() << "Elimination Stack Top DFS numbers are ("
                         << EliminationStack.dfs_back().first << ","
                         << EliminationStack.dfs_back().second << ")\n");
          }

          DEBUG(dbgs() << "Current DFS numbers are (" << MemberDFSIn << ","
                       << MemberDFSOut << ")\n");
          // First, we see if we are out of scope or empty.  If so,
          // and there equivalences, we try to replace the top of
          // stack with equivalences (if it's on the stack, it must
          // not have been eliminated yet).
          // Then we synchronize to our current scope, by
          // popping until we are back within a DFS scope that
          // dominates the current member.
          // Then, what happens depends on a few factors
          // If the stack is now empty, we need to push
          // If we have a constant or a local equivalence we want to
          // start using, we also push.
          // Otherwise, we walk along, processing members who are
          // dominated by this scope, and eliminate them.
          bool ShouldPush = Def && EliminationStack.empty();
          bool OutOfScope =
              !EliminationStack.isInScope(MemberDFSIn, MemberDFSOut);

          if (OutOfScope || ShouldPush) {
            // Sync to our current scope.
            PassPrediction::PassPeeper(3450); // if
            EliminationStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
            bool ShouldPush = Def && EliminationStack.empty();
            if (ShouldPush) {
              PassPrediction::PassPeeper(3451); // if
              EliminationStack.push_back(Def, MemberDFSIn, MemberDFSOut);
            }
          }

          // Skip the Def's, we only want to eliminate on their uses.  But mark
          // dominated defs as dead.
          if (Def) {
            // For anything in this case, what and how we value number
            // guarantees that any side-effets that would have occurred (ie
            // throwing, etc) can be proven to either still occur (because it's
            // dominated by something that has the same side-effects), or never
            // occur.  Otherwise, we would not have been able to prove it value
            // equivalent to something else. For these things, we can just mark
            // it all dead.  Note that this is different from the "ProbablyDead"
            // set, which may not be dominated by anything, and thus, are only
            // easy to prove dead if they are also side-effect free. Note that
            // because stores are put in terms of the stored value, we skip
            // stored values here. If the stored value is really dead, it will
            // still be marked for deletion when we process it in its own class.
            PassPrediction::PassPeeper(3452); // if
            if (!EliminationStack.empty() && Def != EliminationStack.back() &&
                isa<Instruction>(Def) && !FromStore) {
              PassPrediction::PassPeeper(3453); // if
              markInstructionForDeletion(cast<Instruction>(Def));
            }
            continue;
          }
          // At this point, we know it is a Use we are trying to possibly
          // replace.

          assert(isa<Instruction>(U->get()) &&
                 "Current def should have been an instruction");
          assert(isa<Instruction>(U->getUser()) &&
                 "Current user should have been an instruction");

          // If the thing we are replacing into is already marked to be dead,
          // this use is dead.  Note that this is true regardless of whether
          // we have anything dominating the use or not.  We do this here
          // because we are already walking all the uses anyway.
          Instruction *InstUse = cast<Instruction>(U->getUser());
          if (InstructionsToErase.count(InstUse)) {
            PassPrediction::PassPeeper(3454); // if
            auto &UseCount = UseCounts[U->get()];
            if (--UseCount == 0) {
              PassPrediction::PassPeeper(3455); // if
              ProbablyDead.insert(cast<Instruction>(U->get()));
            }
          }

          // If we get to this point, and the stack is empty we must have a use
          // with nothing we can use to eliminate this use, so just skip it.
          if (EliminationStack.empty()) {
            PassPrediction::PassPeeper(3456); // if
            continue;
          }

          Value *DominatingLeader = EliminationStack.back();

          auto *II = dyn_cast<IntrinsicInst>(DominatingLeader);
          if (II && II->getIntrinsicID() == Intrinsic::ssa_copy) {
            PassPrediction::PassPeeper(3457); // if
            DominatingLeader = II->getOperand(0);
          }

          // Don't replace our existing users with ourselves.
          if (U->get() == DominatingLeader) {
            PassPrediction::PassPeeper(3458); // if
            continue;
          }
          DEBUG(dbgs() << "Found replacement " << *DominatingLeader << " for "
                       << *U->get() << " in " << *(U->getUser()) << "\n");

          // If we replaced something in an instruction, handle the patching of
          // metadata.  Skip this if we are replacing predicateinfo with its
          // original operand, as we already know we can just drop it.
          auto *ReplacedInst = cast<Instruction>(U->get());
          auto *PI = PredInfo->getPredicateInfoFor(ReplacedInst);
          if (!PI || DominatingLeader != PI->OriginalOp) {
            PassPrediction::PassPeeper(3459); // if
            patchReplacementInstruction(ReplacedInst, DominatingLeader);
          }
          U->set(DominatingLeader);
          // This is now a use of the dominating leader, which means if the
          // dominating leader was dead, it's now live!
          auto &LeaderUseCount = UseCounts[DominatingLeader];
          // It's about to be alive again.
          if (LeaderUseCount == 0 && isa<Instruction>(DominatingLeader)) {
            PassPrediction::PassPeeper(3460); // if
            ProbablyDead.erase(cast<Instruction>(DominatingLeader));
          }
          if (LeaderUseCount == 0 && II) {
            PassPrediction::PassPeeper(3461); // if
            ProbablyDead.insert(II);
          }
          ++LeaderUseCount;
          AnythingReplaced = true;
        }
      }
    }

    // At this point, anything still in the ProbablyDead set is actually dead if
    // would be trivially dead.
    for (auto *I : ProbablyDead) {
      PassPrediction::PassPeeper(3462); // for-range
      if (wouldInstructionBeTriviallyDead(I)) {
        PassPrediction::PassPeeper(3463); // if
        markInstructionForDeletion(I);
      }
    }

    // Cleanup the congruence class.
    CongruenceClass::MemberSet MembersLeft;
    for (auto *Member : *CC) {
      PassPrediction::PassPeeper(3464); // for-range
      if (!isa<Instruction>(Member) ||
          !InstructionsToErase.count(cast<Instruction>(Member))) {
        PassPrediction::PassPeeper(3465); // if
        MembersLeft.insert(Member);
      }
    }
    CC->swap(MembersLeft);

    // If we have possible dead stores to look at, try to eliminate them.
    if (CC->getStoreCount() > 0) {
      PassPrediction::PassPeeper(3466); // if
      convertClassToLoadsAndStores(*CC, PossibleDeadStores);
      std::sort(PossibleDeadStores.begin(), PossibleDeadStores.end());
      ValueDFSStack EliminationStack;
      for (auto &VD : PossibleDeadStores) {
        PassPrediction::PassPeeper(3467); // for-range
        int MemberDFSIn = VD.DFSIn;
        int MemberDFSOut = VD.DFSOut;
        Instruction *Member = cast<Instruction>(VD.Def.getPointer());
        if (EliminationStack.empty() ||
            !EliminationStack.isInScope(MemberDFSIn, MemberDFSOut)) {
          // Sync to our current scope.
          PassPrediction::PassPeeper(3468); // if
          EliminationStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
          if (EliminationStack.empty()) {
            PassPrediction::PassPeeper(3469); // if
            EliminationStack.push_back(Member, MemberDFSIn, MemberDFSOut);
            continue;
          }
        }
        // We already did load elimination, so nothing to do here.
        if (isa<LoadInst>(Member)) {
          PassPrediction::PassPeeper(3470); // if
          continue;
        }
        assert(!EliminationStack.empty());
        Instruction *Leader = cast<Instruction>(EliminationStack.back());
        (void)Leader;
        assert(DT->dominates(Leader->getParent(), Member->getParent()));
        // Member is dominater by Leader, and thus dead
        DEBUG(dbgs() << "Marking dead store " << *Member
                     << " that is dominated by " << *Leader << "\n");
        markInstructionForDeletion(Member);
        CC->erase(Member);
        ++NumGVNDeadStores;
      }
    }
  }
  return AnythingReplaced;
}

// This function provides global ranking of operations so that we can place them
// in a canonical order.  Note that rank alone is not necessarily enough for a
// complete ordering, as constants all have the same rank.  However, generally,
// we will simplify an operation with all constants so that it doesn't matter
// what order they appear in.
unsigned int NewGVN::getRank(const Value *V) const {
  // Prefer constants to undef to anything else
  // Undef is a constant, have to check it first.
  // Prefer smaller constants to constantexprs
  if (isa<ConstantExpr>(V)) {
    PassPrediction::PassPeeper(3471); // if
    return 2;
  }
  if (isa<UndefValue>(V)) {
    PassPrediction::PassPeeper(3472); // if
    return 1;
  }
  if (isa<Constant>(V)) {
    PassPrediction::PassPeeper(3473); // if
    return 0;
  } else if (auto *A = dyn_cast<Argument>(V)) {
    PassPrediction::PassPeeper(3474); // if
    return 3 + A->getArgNo();
  }

  // Need to shift the instruction DFS by number of arguments + 3 to account for
  // the constant and argument ranking above.
  unsigned Result = InstrToDFSNum(V);
  if (Result > 0) {
    PassPrediction::PassPeeper(3475); // if
    return 4 + NumFuncArgs + Result;
  }
  // Unreachable or something else, just return a really large number.
  return ~0;
}

// This is a function that says whether two commutative operations should
// have their order swapped when canonicalizing.
bool NewGVN::shouldSwapOperands(const Value *A, const Value *B) const {
  // Because we only care about a total ordering, and don't rewrite expressions
  // in this order, we order by rank, which will give a strict weak ordering to
  // everything but constants, and then we order by pointer address.
  return std::make_pair(getRank(A), A) > std::make_pair(getRank(B), B);
}

namespace {
class NewGVNLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid.
  NewGVNLegacyPass() : FunctionPass(ID) {
    initializeNewGVNLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override;

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
} // namespace

bool NewGVNLegacyPass::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    PassPrediction::PassPeeper(3476); // if
    return false;
  }
  return NewGVN(F, &getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
                &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F),
                &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(),
                &getAnalysis<AAResultsWrapperPass>().getAAResults(),
                &getAnalysis<MemorySSAWrapperPass>().getMSSA(),
                F.getParent()->getDataLayout())
      .runGVN();
}

INITIALIZE_PASS_BEGIN(NewGVNLegacyPass, "newgvn", "Global Value Numbering",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_END(NewGVNLegacyPass, "newgvn", "Global Value Numbering", false,
                    false)

char NewGVNLegacyPass::ID = 0;

// createGVNPass - The public interface to this file.
FunctionPass *llvm::createNewGVNPass() { return new NewGVNLegacyPass(); }

PreservedAnalyses NewGVNPass::run(Function &F, AnalysisManager<Function> &AM) {
  // Apparently the order in which we get these results matter for
  // the old GVN (see Chandler's comment in GVN.cpp). I'll keep
  // the same order here, just in case.
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  auto &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  bool Changed =
      NewGVN(F, &DT, &AC, &TLI, &AA, &MSSA, F.getParent()->getDataLayout())
          .runGVN();
  if (!Changed) {
    PassPrediction::PassPeeper(3477); // if
    return PreservedAnalyses::all();
  }
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}
