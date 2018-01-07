#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
//===--- Scalarizer.cpp - Scalarize vector operations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass converts vector operations into scalar operations, in order
// to expose optimization opportunities on the individual scalar operations.
// It is mainly intended for targets that do not have vector units, but it
// may also be useful for revectorizing code to different vector widths.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "scalarizer"

namespace {
// Used to store the scattered form of a vector.
typedef SmallVector<Value *, 8> ValueVector;

// Used to map a vector Value to its scattered form.  We use std::map
// because we want iterators to persist across insertion and because the
// values are relatively large.
typedef std::map<Value *, ValueVector> ScatterMap;

// Lists Instructions that have been replaced with scalar implementations,
// along with a pointer to their scattered forms.
typedef SmallVector<std::pair<Instruction *, ValueVector *>, 16> GatherList;

// Provides a very limited vector-like interface for lazily accessing one
// component of a scattered vector or vector pointer.
class Scatterer {
public:
  Scatterer() {}

  // Scatter V into Size components.  If new instructions are needed,
  // insert them before BBI in BB.  If Cache is nonnull, use it to cache
  // the results.
  Scatterer(BasicBlock *bb, BasicBlock::iterator bbi, Value *v,
            ValueVector *cachePtr = nullptr);

  // Return component I, creating a new Value for it if necessary.
  Value *operator[](unsigned I);

  // Return the number of components.
  unsigned size() const { return Size; }

private:
  BasicBlock *BB;
  BasicBlock::iterator BBI;
  Value *V;
  ValueVector *CachePtr;
  PointerType *PtrTy;
  ValueVector Tmp;
  unsigned Size;
};

// FCmpSpliiter(FCI)(Builder, X, Y, Name) uses Builder to create an FCmp
// called Name that compares X and Y in the same way as FCI.
struct FCmpSplitter {
  FCmpSplitter(FCmpInst &fci) : FCI(fci) {}
  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateFCmp(FCI.getPredicate(), Op0, Op1, Name);
  }
  FCmpInst &FCI;
};

// ICmpSpliiter(ICI)(Builder, X, Y, Name) uses Builder to create an ICmp
// called Name that compares X and Y in the same way as ICI.
struct ICmpSplitter {
  ICmpSplitter(ICmpInst &ici) : ICI(ici) {}
  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateICmp(ICI.getPredicate(), Op0, Op1, Name);
  }
  ICmpInst &ICI;
};

// BinarySpliiter(BO)(Builder, X, Y, Name) uses Builder to create
// a binary operator like BO called Name with operands X and Y.
struct BinarySplitter {
  BinarySplitter(BinaryOperator &bo) : BO(bo) {}
  Value *operator()(IRBuilder<> &Builder, Value *Op0, Value *Op1,
                    const Twine &Name) const {
    return Builder.CreateBinOp(BO.getOpcode(), Op0, Op1, Name);
  }
  BinaryOperator &BO;
};

// Information about a load or store that we're scalarizing.
struct VectorLayout {
  VectorLayout() : VecTy(nullptr), ElemTy(nullptr), VecAlign(0), ElemSize(0) {}

  // Return the alignment of element I.
  uint64_t getElemAlign(unsigned I) { return MinAlign(VecAlign, I * ElemSize); }

  // The type of the vector.
  VectorType *VecTy;

  // The type of each element.
  Type *ElemTy;

  // The alignment of the vector.
  uint64_t VecAlign;

  // The size of each element.
  uint64_t ElemSize;
};

class Scalarizer : public FunctionPass, public InstVisitor<Scalarizer, bool> {
public:
  static char ID;

  Scalarizer() : FunctionPass(ID) {
    initializeScalarizerPass(*PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  // InstVisitor methods.  They return true if the instruction was scalarized,
  // false if nothing changed.
  bool visitInstruction(Instruction &) { return false; }
  bool visitSelectInst(SelectInst &SI);
  bool visitICmpInst(ICmpInst &);
  bool visitFCmpInst(FCmpInst &);
  bool visitBinaryOperator(BinaryOperator &);
  bool visitGetElementPtrInst(GetElementPtrInst &);
  bool visitCastInst(CastInst &);
  bool visitBitCastInst(BitCastInst &);
  bool visitShuffleVectorInst(ShuffleVectorInst &);
  bool visitPHINode(PHINode &);
  bool visitLoadInst(LoadInst &);
  bool visitStoreInst(StoreInst &);
  bool visitCallInst(CallInst &I);

  static void registerOptions() {
    // This is disabled by default because having separate loads and stores
    // makes it more likely that the -combiner-alias-analysis limits will be
    // reached.
    OptionRegistry::registerOption<bool, Scalarizer,
                                   &Scalarizer::ScalarizeLoadStore>(
        "scalarize-load-store",
        "Allow the scalarizer pass to scalarize loads and store", false);
  }

private:
  Scatterer scatter(Instruction *, Value *);
  void gather(Instruction *, const ValueVector &);
  bool canTransferMetadata(unsigned Kind);
  void transferMetadata(Instruction *, const ValueVector &);
  bool getVectorLayout(Type *, unsigned, VectorLayout &, const DataLayout &);
  bool finish();

  template <typename T> bool splitBinary(Instruction &, const T &);

  bool splitCall(CallInst &CI);

  ScatterMap Scattered;
  GatherList Gathered;
  unsigned ParallelLoopAccessMDKind;
  bool ScalarizeLoadStore;
};

char Scalarizer::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_WITH_OPTIONS(Scalarizer, "scalarizer",
                             "Scalarize vector operations", false, false)

Scatterer::Scatterer(BasicBlock *bb, BasicBlock::iterator bbi, Value *v,
                     ValueVector *cachePtr)
    : BB(bb), BBI(bbi), V(v), CachePtr(cachePtr) {
  Type *Ty = V->getType();
  PtrTy = dyn_cast<PointerType>(Ty);
  if (PtrTy) {
    PassPrediction::PassPeeper(__FILE__, 1577); // if
    Ty = PtrTy->getElementType();
  }
  Size = Ty->getVectorNumElements();
  if (!CachePtr) {
    PassPrediction::PassPeeper(__FILE__, 1578); // if
    Tmp.resize(Size, nullptr);
  } else if (CachePtr->empty()) {
    PassPrediction::PassPeeper(__FILE__, 1579); // if
    CachePtr->resize(Size, nullptr);
  } else {
    assert(Size == CachePtr->size() && "Inconsistent vector sizes");
  }
}

// Return component I, creating a new Value for it if necessary.
Value *Scatterer::operator[](unsigned I) {
  ValueVector &CV = (CachePtr ? *CachePtr : Tmp);
  // Try to reuse a previous value.
  if (CV[I]) {
    PassPrediction::PassPeeper(__FILE__, 1580); // if
    return CV[I];
  }
  IRBuilder<> Builder(BB, BBI);
  if (PtrTy) {
    PassPrediction::PassPeeper(__FILE__, 1581); // if
    if (!CV[0]) {
      PassPrediction::PassPeeper(__FILE__, 1583); // if
      Type *Ty =
          PointerType::get(PtrTy->getElementType()->getVectorElementType(),
                           PtrTy->getAddressSpace());
      CV[0] = Builder.CreateBitCast(V, Ty, V->getName() + ".i0");
    }
    if (I != 0) {
      PassPrediction::PassPeeper(__FILE__, 1584); // if
      CV[I] = Builder.CreateConstGEP1_32(nullptr, CV[0], I,
                                         V->getName() + ".i" + Twine(I));
    }
  } else {
    // Search through a chain of InsertElementInsts looking for element I.
    // Record other elements in the cache.  The new V is still suitable
    // for all uncached indices.
    PassPrediction::PassPeeper(__FILE__, 1582); // else
    for (;;) {
      PassPrediction::PassPeeper(__FILE__, 1585); // for
      InsertElementInst *Insert = dyn_cast<InsertElementInst>(V);
      if (!Insert) {
        PassPrediction::PassPeeper(__FILE__, 1586); // if
        break;
      }
      ConstantInt *Idx = dyn_cast<ConstantInt>(Insert->getOperand(2));
      if (!Idx) {
        PassPrediction::PassPeeper(__FILE__, 1587); // if
        break;
      }
      unsigned J = Idx->getZExtValue();
      V = Insert->getOperand(0);
      if (I == J) {
        PassPrediction::PassPeeper(__FILE__, 1588); // if
        CV[J] = Insert->getOperand(1);
        return CV[J];
      } else if (!CV[J]) {
        // Only cache the first entry we find for each index we're not actively
        // searching for. This prevents us from going too far up the chain and
        // caching incorrect entries.
        PassPrediction::PassPeeper(__FILE__, 1589); // if
        CV[J] = Insert->getOperand(1);
      }
    }
    CV[I] = Builder.CreateExtractElement(V, Builder.getInt32(I),
                                         V->getName() + ".i" + Twine(I));
  }
  return CV[I];
}

bool Scalarizer::doInitialization(Module &M) {
  ParallelLoopAccessMDKind =
      M.getContext().getMDKindID("llvm.mem.parallel_loop_access");
  ScalarizeLoadStore =
      M.getContext()
          .getOption<bool, Scalarizer, &Scalarizer::ScalarizeLoadStore>();
  return false;
}

bool Scalarizer::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    PassPrediction::PassPeeper(__FILE__, 1590); // if
    return false;
  }
  assert(Gathered.empty() && Scattered.empty());
  for (BasicBlock &BB : F) {
    PassPrediction::PassPeeper(__FILE__, 1591); // for-range
    for (BasicBlock::iterator II = BB.begin(), IE = BB.end(); II != IE;) {
      PassPrediction::PassPeeper(__FILE__, 1592); // for
      Instruction *I = &*II;
      bool Done = visit(I);
      ++II;
      if (Done && I->getType()->isVoidTy()) {
        PassPrediction::PassPeeper(__FILE__, 1593); // if
        I->eraseFromParent();
      }
    }
  }
  return finish();
}

// Return a scattered form of V that can be accessed by Point.  V must be a
// vector or a pointer to a vector.
Scatterer Scalarizer::scatter(Instruction *Point, Value *V) {
  if (Argument *VArg = dyn_cast<Argument>(V)) {
    // Put the scattered form of arguments in the entry block,
    // so that it can be used everywhere.
    PassPrediction::PassPeeper(__FILE__, 1594); // if
    Function *F = VArg->getParent();
    BasicBlock *BB = &F->getEntryBlock();
    return Scatterer(BB, BB->begin(), V, &Scattered[V]);
  }
  if (Instruction *VOp = dyn_cast<Instruction>(V)) {
    // Put the scattered form of an instruction directly after the
    // instruction.
    PassPrediction::PassPeeper(__FILE__, 1595); // if
    BasicBlock *BB = VOp->getParent();
    return Scatterer(BB, std::next(BasicBlock::iterator(VOp)), V,
                     &Scattered[V]);
  }
  // In the fallback case, just put the scattered before Point and
  // keep the result local to Point.
  return Scatterer(Point->getParent(), Point->getIterator(), V);
}

// Replace Op with the gathered form of the components in CV.  Defer the
// deletion of Op and creation of the gathered form to the end of the pass,
// so that we can avoid creating the gathered form if all uses of Op are
// replaced with uses of CV.
void Scalarizer::gather(Instruction *Op, const ValueVector &CV) {
  // Since we're not deleting Op yet, stub out its operands, so that it
  // doesn't make anything live unnecessarily.
  for (unsigned I = 0, E = Op->getNumOperands(); I != E; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1596); // for
    Op->setOperand(I, UndefValue::get(Op->getOperand(I)->getType()));
  }

  transferMetadata(Op, CV);

  // If we already have a scattered form of Op (created from ExtractElements
  // of Op itself), replace them with the new form.
  ValueVector &SV = Scattered[Op];
  if (!SV.empty()) {
    PassPrediction::PassPeeper(__FILE__, 1597); // if
    for (unsigned I = 0, E = SV.size(); I != E; ++I) {
      PassPrediction::PassPeeper(__FILE__, 1598); // for
      Value *V = SV[I];
      if (V == nullptr) {
        PassPrediction::PassPeeper(__FILE__, 1599); // if
        continue;
      }

      Instruction *Old = cast<Instruction>(V);
      CV[I]->takeName(Old);
      Old->replaceAllUsesWith(CV[I]);
      Old->eraseFromParent();
    }
  }
  SV = CV;
  Gathered.push_back(GatherList::value_type(Op, &SV));
}

// Return true if it is safe to transfer the given metadata tag from
// vector to scalar instructions.
bool Scalarizer::canTransferMetadata(unsigned Tag) {
  return (Tag == LLVMContext::MD_tbaa || Tag == LLVMContext::MD_fpmath ||
          Tag == LLVMContext::MD_tbaa_struct ||
          Tag == LLVMContext::MD_invariant_load ||
          Tag == LLVMContext::MD_alias_scope ||
          Tag == LLVMContext::MD_noalias || Tag == ParallelLoopAccessMDKind);
}

// Transfer metadata from Op to the instructions in CV if it is known
// to be safe to do so.
void Scalarizer::transferMetadata(Instruction *Op, const ValueVector &CV) {
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  Op->getAllMetadataOtherThanDebugLoc(MDs);
  for (unsigned I = 0, E = CV.size(); I != E; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1600); // for
    if (Instruction *New = dyn_cast<Instruction>(CV[I])) {
      PassPrediction::PassPeeper(__FILE__, 1601); // if
      for (const auto &MD : MDs) {
        PassPrediction::PassPeeper(__FILE__, 1602); // for-range
        if (canTransferMetadata(MD.first)) {
          PassPrediction::PassPeeper(__FILE__, 1603); // if
          New->setMetadata(MD.first, MD.second);
        }
      }
      if (Op->getDebugLoc() && !New->getDebugLoc()) {
        PassPrediction::PassPeeper(__FILE__, 1604); // if
        New->setDebugLoc(Op->getDebugLoc());
      }
    }
  }
}

// Try to fill in Layout from Ty, returning true on success.  Alignment is
// the alignment of the vector, or 0 if the ABI default should be used.
bool Scalarizer::getVectorLayout(Type *Ty, unsigned Alignment,
                                 VectorLayout &Layout, const DataLayout &DL) {
  // Make sure we're dealing with a vector.
  Layout.VecTy = dyn_cast<VectorType>(Ty);
  if (!Layout.VecTy) {
    PassPrediction::PassPeeper(__FILE__, 1605); // if
    return false;
  }

  // Check that we're dealing with full-byte elements.
  Layout.ElemTy = Layout.VecTy->getElementType();
  if (DL.getTypeSizeInBits(Layout.ElemTy) !=
      DL.getTypeStoreSizeInBits(Layout.ElemTy)) {
    PassPrediction::PassPeeper(__FILE__, 1606); // if
    return false;
  }

  if (Alignment) {
    PassPrediction::PassPeeper(__FILE__, 1607); // if
    Layout.VecAlign = Alignment;
  } else {
    PassPrediction::PassPeeper(__FILE__, 1608); // else
    Layout.VecAlign = DL.getABITypeAlignment(Layout.VecTy);
  }
  Layout.ElemSize = DL.getTypeStoreSize(Layout.ElemTy);
  return true;
}

// Scalarize two-operand instruction I, using Split(Builder, X, Y, Name)
// to create an instruction like I with operands X and Y and name Name.
template <typename Splitter>
bool Scalarizer::splitBinary(Instruction &I, const Splitter &Split) {
  VectorType *VT = dyn_cast<VectorType>(I.getType());
  if (!VT) {
    PassPrediction::PassPeeper(__FILE__, 1575); // if
    return false;
  }

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(&I);
  Scatterer Op0 = scatter(&I, I.getOperand(0));
  Scatterer Op1 = scatter(&I, I.getOperand(1));
  assert(Op0.size() == NumElems && "Mismatched binary operation");
  assert(Op1.size() == NumElems && "Mismatched binary operation");
  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned Elem = 0; Elem < NumElems; ++Elem) {
    PassPrediction::PassPeeper(__FILE__, 1576); // for
    Res[Elem] =
        Split(Builder, Op0[Elem], Op1[Elem], I.getName() + ".i" + Twine(Elem));
  }
  gather(&I, Res);
  return true;
}

static bool isTriviallyScalariable(Intrinsic::ID ID) {
  return isTriviallyVectorizable(ID);
}

// All of the current scalarizable intrinsics only have one mangled type.
static Function *getScalarIntrinsicDeclaration(Module *M, Intrinsic::ID ID,
                                               VectorType *Ty) {
  return Intrinsic::getDeclaration(M, ID, {Ty->getScalarType()});
}

/// If a call to a vector typed intrinsic function, split into a scalar call per
/// element if possible for the intrinsic.
bool Scalarizer::splitCall(CallInst &CI) {
  VectorType *VT = dyn_cast<VectorType>(CI.getType());
  if (!VT) {
    PassPrediction::PassPeeper(__FILE__, 1609); // if
    return false;
  }

  Function *F = CI.getCalledFunction();
  if (!F) {
    PassPrediction::PassPeeper(__FILE__, 1610); // if
    return false;
  }

  Intrinsic::ID ID = F->getIntrinsicID();
  if (ID == Intrinsic::not_intrinsic || !isTriviallyScalariable(ID)) {
    PassPrediction::PassPeeper(__FILE__, 1611); // if
    return false;
  }

  unsigned NumElems = VT->getNumElements();
  unsigned NumArgs = CI.getNumArgOperands();

  ValueVector ScalarOperands(NumArgs);
  SmallVector<Scatterer, 8> Scattered(NumArgs);

  Scattered.resize(NumArgs);

  // Assumes that any vector type has the same number of elements as the return
  // vector type, which is true for all current intrinsics.
  for (unsigned I = 0; I != NumArgs; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1612); // for
    Value *OpI = CI.getOperand(I);
    if (OpI->getType()->isVectorTy()) {
      PassPrediction::PassPeeper(__FILE__, 1613); // if
      Scattered[I] = scatter(&CI, OpI);
      assert(Scattered[I].size() == NumElems && "mismatched call operands");
    } else {
      PassPrediction::PassPeeper(__FILE__, 1614); // else
      ScalarOperands[I] = OpI;
    }
  }

  ValueVector Res(NumElems);
  ValueVector ScalarCallOps(NumArgs);

  Function *NewIntrin = getScalarIntrinsicDeclaration(F->getParent(), ID, VT);
  IRBuilder<> Builder(&CI);

  // Perform actual scalarization, taking care to preserve any scalar operands.
  for (unsigned Elem = 0; Elem < NumElems; ++Elem) {
    PassPrediction::PassPeeper(__FILE__, 1615); // for
    ScalarCallOps.clear();

    for (unsigned J = 0; J != NumArgs; ++J) {
      PassPrediction::PassPeeper(__FILE__, 1616); // for
      if (hasVectorInstrinsicScalarOpd(ID, J)) {
        PassPrediction::PassPeeper(__FILE__, 1617); // if
        ScalarCallOps.push_back(ScalarOperands[J]);
      } else {
        PassPrediction::PassPeeper(__FILE__, 1618); // else
        ScalarCallOps.push_back(Scattered[J][Elem]);
      }
    }

    Res[Elem] = Builder.CreateCall(NewIntrin, ScalarCallOps,
                                   CI.getName() + ".i" + Twine(Elem));
  }

  gather(&CI, Res);
  return true;
}

bool Scalarizer::visitSelectInst(SelectInst &SI) {
  VectorType *VT = dyn_cast<VectorType>(SI.getType());
  if (!VT) {
    PassPrediction::PassPeeper(__FILE__, 1619); // if
    return false;
  }

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(&SI);
  Scatterer Op1 = scatter(&SI, SI.getOperand(1));
  Scatterer Op2 = scatter(&SI, SI.getOperand(2));
  assert(Op1.size() == NumElems && "Mismatched select");
  assert(Op2.size() == NumElems && "Mismatched select");
  ValueVector Res;
  Res.resize(NumElems);

  if (SI.getOperand(0)->getType()->isVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 1620); // if
    Scatterer Op0 = scatter(&SI, SI.getOperand(0));
    assert(Op0.size() == NumElems && "Mismatched select");
    for (unsigned I = 0; I < NumElems; ++I) {
      PassPrediction::PassPeeper(__FILE__, 1622); // for
      Res[I] = Builder.CreateSelect(Op0[I], Op1[I], Op2[I],
                                    SI.getName() + ".i" + Twine(I));
    }
  } else {
    PassPrediction::PassPeeper(__FILE__, 1621); // else
    Value *Op0 = SI.getOperand(0);
    for (unsigned I = 0; I < NumElems; ++I) {
      PassPrediction::PassPeeper(__FILE__, 1623); // for
      Res[I] = Builder.CreateSelect(Op0, Op1[I], Op2[I],
                                    SI.getName() + ".i" + Twine(I));
    }
  }
  gather(&SI, Res);
  return true;
}

bool Scalarizer::visitICmpInst(ICmpInst &ICI) {
  return splitBinary(ICI, ICmpSplitter(ICI));
}

bool Scalarizer::visitFCmpInst(FCmpInst &FCI) {
  return splitBinary(FCI, FCmpSplitter(FCI));
}

bool Scalarizer::visitBinaryOperator(BinaryOperator &BO) {
  return splitBinary(BO, BinarySplitter(BO));
}

bool Scalarizer::visitGetElementPtrInst(GetElementPtrInst &GEPI) {
  VectorType *VT = dyn_cast<VectorType>(GEPI.getType());
  if (!VT) {
    PassPrediction::PassPeeper(__FILE__, 1624); // if
    return false;
  }

  IRBuilder<> Builder(&GEPI);
  unsigned NumElems = VT->getNumElements();
  unsigned NumIndices = GEPI.getNumIndices();

  // The base pointer might be scalar even if it's a vector GEP. In those cases,
  // splat the pointer into a vector value, and scatter that vector.
  Value *Op0 = GEPI.getOperand(0);
  if (!Op0->getType()->isVectorTy()) {
    PassPrediction::PassPeeper(__FILE__, 1625); // if
    Op0 = Builder.CreateVectorSplat(NumElems, Op0);
  }
  Scatterer Base = scatter(&GEPI, Op0);

  SmallVector<Scatterer, 8> Ops;
  Ops.resize(NumIndices);
  for (unsigned I = 0; I < NumIndices; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1626); // for
    Value *Op = GEPI.getOperand(I + 1);

    // The indices might be scalars even if it's a vector GEP. In those cases,
    // splat the scalar into a vector value, and scatter that vector.
    if (!Op->getType()->isVectorTy()) {
      PassPrediction::PassPeeper(__FILE__, 1627); // if
      Op = Builder.CreateVectorSplat(NumElems, Op);
    }

    Ops[I] = scatter(&GEPI, Op);
  }

  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned I = 0; I < NumElems; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1628); // for
    SmallVector<Value *, 8> Indices;
    Indices.resize(NumIndices);
    for (unsigned J = 0; J < NumIndices; ++J) {
      PassPrediction::PassPeeper(__FILE__, 1629); // for
      Indices[J] = Ops[J][I];
    }
    Res[I] = Builder.CreateGEP(GEPI.getSourceElementType(), Base[I], Indices,
                               GEPI.getName() + ".i" + Twine(I));
    if (GEPI.isInBounds()) {
      PassPrediction::PassPeeper(__FILE__, 1630); // if
      if (GetElementPtrInst *NewGEPI = dyn_cast<GetElementPtrInst>(Res[I])) {
        PassPrediction::PassPeeper(__FILE__, 1631); // if
        NewGEPI->setIsInBounds();
      }
    }
  }
  gather(&GEPI, Res);
  return true;
}

bool Scalarizer::visitCastInst(CastInst &CI) {
  VectorType *VT = dyn_cast<VectorType>(CI.getDestTy());
  if (!VT) {
    PassPrediction::PassPeeper(__FILE__, 1632); // if
    return false;
  }

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(&CI);
  Scatterer Op0 = scatter(&CI, CI.getOperand(0));
  assert(Op0.size() == NumElems && "Mismatched cast");
  ValueVector Res;
  Res.resize(NumElems);
  for (unsigned I = 0; I < NumElems; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1633); // for
    Res[I] = Builder.CreateCast(CI.getOpcode(), Op0[I], VT->getElementType(),
                                CI.getName() + ".i" + Twine(I));
  }
  gather(&CI, Res);
  return true;
}

bool Scalarizer::visitBitCastInst(BitCastInst &BCI) {
  VectorType *DstVT = dyn_cast<VectorType>(BCI.getDestTy());
  VectorType *SrcVT = dyn_cast<VectorType>(BCI.getSrcTy());
  if (!DstVT || !SrcVT) {
    PassPrediction::PassPeeper(__FILE__, 1634); // if
    return false;
  }

  unsigned DstNumElems = DstVT->getNumElements();
  unsigned SrcNumElems = SrcVT->getNumElements();
  IRBuilder<> Builder(&BCI);
  Scatterer Op0 = scatter(&BCI, BCI.getOperand(0));
  ValueVector Res;
  Res.resize(DstNumElems);

  if (DstNumElems == SrcNumElems) {
    PassPrediction::PassPeeper(__FILE__, 1635); // if
    for (unsigned I = 0; I < DstNumElems; ++I) {
      PassPrediction::PassPeeper(__FILE__, 1636); // for
      Res[I] = Builder.CreateBitCast(Op0[I], DstVT->getElementType(),
                                     BCI.getName() + ".i" + Twine(I));
    }
  } else if (DstNumElems > SrcNumElems) {
    // <M x t1> -> <N*M x t2>.  Convert each t1 to <N x t2> and copy the
    // individual elements to the destination.
    PassPrediction::PassPeeper(__FILE__, 1637); // if
    unsigned FanOut = DstNumElems / SrcNumElems;
    Type *MidTy = VectorType::get(DstVT->getElementType(), FanOut);
    unsigned ResI = 0;
    for (unsigned Op0I = 0; Op0I < SrcNumElems; ++Op0I) {
      PassPrediction::PassPeeper(__FILE__, 1639); // for
      Value *V = Op0[Op0I];
      Instruction *VI;
      // Look through any existing bitcasts before converting to <N x t2>.
      // In the best case, the resulting conversion might be a no-op.
      while ((VI = dyn_cast<Instruction>(V)) &&
             VI->getOpcode() == Instruction::BitCast) {
        PassPrediction::PassPeeper(__FILE__, 1640); // while
        V = VI->getOperand(0);
      }
      V = Builder.CreateBitCast(V, MidTy, V->getName() + ".cast");
      Scatterer Mid = scatter(&BCI, V);
      for (unsigned MidI = 0; MidI < FanOut; ++MidI) {
        PassPrediction::PassPeeper(__FILE__, 1641); // for
        Res[ResI++] = Mid[MidI];
      }
    }
  } else {
    // <N*M x t1> -> <M x t2>.  Convert each group of <N x t1> into a t2.
    PassPrediction::PassPeeper(__FILE__, 1638); // else
    unsigned FanIn = SrcNumElems / DstNumElems;
    Type *MidTy = VectorType::get(SrcVT->getElementType(), FanIn);
    unsigned Op0I = 0;
    for (unsigned ResI = 0; ResI < DstNumElems; ++ResI) {
      PassPrediction::PassPeeper(__FILE__, 1642); // for
      Value *V = UndefValue::get(MidTy);
      for (unsigned MidI = 0; MidI < FanIn; ++MidI) {
        PassPrediction::PassPeeper(__FILE__, 1643); // for
        V = Builder.CreateInsertElement(V, Op0[Op0I++], Builder.getInt32(MidI),
                                        BCI.getName() + ".i" + Twine(ResI) +
                                            ".upto" + Twine(MidI));
      }
      Res[ResI] = Builder.CreateBitCast(V, DstVT->getElementType(),
                                        BCI.getName() + ".i" + Twine(ResI));
    }
  }
  gather(&BCI, Res);
  return true;
}

bool Scalarizer::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  VectorType *VT = dyn_cast<VectorType>(SVI.getType());
  if (!VT) {
    PassPrediction::PassPeeper(__FILE__, 1644); // if
    return false;
  }

  unsigned NumElems = VT->getNumElements();
  Scatterer Op0 = scatter(&SVI, SVI.getOperand(0));
  Scatterer Op1 = scatter(&SVI, SVI.getOperand(1));
  ValueVector Res;
  Res.resize(NumElems);

  for (unsigned I = 0; I < NumElems; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1645); // for
    int Selector = SVI.getMaskValue(I);
    if (Selector < 0) {
      PassPrediction::PassPeeper(__FILE__, 1646); // if
      Res[I] = UndefValue::get(VT->getElementType());
    } else if (unsigned(Selector) < Op0.size()) {
      PassPrediction::PassPeeper(__FILE__, 1647); // if
      Res[I] = Op0[Selector];
    } else {
      PassPrediction::PassPeeper(__FILE__, 1648); // else
      Res[I] = Op1[Selector - Op0.size()];
    }
  }
  gather(&SVI, Res);
  return true;
}

bool Scalarizer::visitPHINode(PHINode &PHI) {
  VectorType *VT = dyn_cast<VectorType>(PHI.getType());
  if (!VT) {
    PassPrediction::PassPeeper(__FILE__, 1649); // if
    return false;
  }

  unsigned NumElems = VT->getNumElements();
  IRBuilder<> Builder(&PHI);
  ValueVector Res;
  Res.resize(NumElems);

  unsigned NumOps = PHI.getNumOperands();
  for (unsigned I = 0; I < NumElems; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1650); // for
    Res[I] = Builder.CreatePHI(VT->getElementType(), NumOps,
                               PHI.getName() + ".i" + Twine(I));
  }

  for (unsigned I = 0; I < NumOps; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1651); // for
    Scatterer Op = scatter(&PHI, PHI.getIncomingValue(I));
    BasicBlock *IncomingBlock = PHI.getIncomingBlock(I);
    for (unsigned J = 0; J < NumElems; ++J) {
      PassPrediction::PassPeeper(__FILE__, 1652); // for
      cast<PHINode>(Res[J])->addIncoming(Op[J], IncomingBlock);
    }
  }
  gather(&PHI, Res);
  return true;
}

bool Scalarizer::visitLoadInst(LoadInst &LI) {
  if (!ScalarizeLoadStore) {
    PassPrediction::PassPeeper(__FILE__, 1653); // if
    return false;
  }
  if (!LI.isSimple()) {
    PassPrediction::PassPeeper(__FILE__, 1654); // if
    return false;
  }

  VectorLayout Layout;
  if (!getVectorLayout(LI.getType(), LI.getAlignment(), Layout,
                       LI.getModule()->getDataLayout())) {
    PassPrediction::PassPeeper(__FILE__, 1655); // if
    return false;
  }

  unsigned NumElems = Layout.VecTy->getNumElements();
  IRBuilder<> Builder(&LI);
  Scatterer Ptr = scatter(&LI, LI.getPointerOperand());
  ValueVector Res;
  Res.resize(NumElems);

  for (unsigned I = 0; I < NumElems; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1656); // for
    Res[I] = Builder.CreateAlignedLoad(Ptr[I], Layout.getElemAlign(I),
                                       LI.getName() + ".i" + Twine(I));
  }
  gather(&LI, Res);
  return true;
}

bool Scalarizer::visitStoreInst(StoreInst &SI) {
  if (!ScalarizeLoadStore) {
    PassPrediction::PassPeeper(__FILE__, 1657); // if
    return false;
  }
  if (!SI.isSimple()) {
    PassPrediction::PassPeeper(__FILE__, 1658); // if
    return false;
  }

  VectorLayout Layout;
  Value *FullValue = SI.getValueOperand();
  if (!getVectorLayout(FullValue->getType(), SI.getAlignment(), Layout,
                       SI.getModule()->getDataLayout())) {
    PassPrediction::PassPeeper(__FILE__, 1659); // if
    return false;
  }

  unsigned NumElems = Layout.VecTy->getNumElements();
  IRBuilder<> Builder(&SI);
  Scatterer Ptr = scatter(&SI, SI.getPointerOperand());
  Scatterer Val = scatter(&SI, FullValue);

  ValueVector Stores;
  Stores.resize(NumElems);
  for (unsigned I = 0; I < NumElems; ++I) {
    PassPrediction::PassPeeper(__FILE__, 1660); // for
    unsigned Align = Layout.getElemAlign(I);
    Stores[I] = Builder.CreateAlignedStore(Val[I], Ptr[I], Align);
  }
  transferMetadata(&SI, Stores);
  return true;
}

bool Scalarizer::visitCallInst(CallInst &CI) { return splitCall(CI); }

// Delete the instructions that we scalarized.  If a full vector result
// is still needed, recreate it using InsertElements.
bool Scalarizer::finish() {
  // The presence of data in Gathered or Scattered indicates changes
  // made to the Function.
  if (Gathered.empty() && Scattered.empty()) {
    PassPrediction::PassPeeper(__FILE__, 1661); // if
    return false;
  }
  for (const auto &GMI : Gathered) {
    PassPrediction::PassPeeper(__FILE__, 1662); // for-range
    Instruction *Op = GMI.first;
    ValueVector &CV = *GMI.second;
    if (!Op->use_empty()) {
      // The value is still needed, so recreate it using a series of
      // InsertElements.
      PassPrediction::PassPeeper(__FILE__, 1663); // if
      Type *Ty = Op->getType();
      Value *Res = UndefValue::get(Ty);
      BasicBlock *BB = Op->getParent();
      unsigned Count = Ty->getVectorNumElements();
      IRBuilder<> Builder(Op);
      if (isa<PHINode>(Op)) {
        PassPrediction::PassPeeper(__FILE__, 1664); // if
        Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
      }
      for (unsigned I = 0; I < Count; ++I) {
        PassPrediction::PassPeeper(__FILE__, 1665); // for
        Res = Builder.CreateInsertElement(Res, CV[I], Builder.getInt32(I),
                                          Op->getName() + ".upto" + Twine(I));
      }
      Res->takeName(Op);
      Op->replaceAllUsesWith(Res);
    }
    Op->eraseFromParent();
  }
  Gathered.clear();
  Scattered.clear();
  return true;
}

FunctionPass *llvm::createScalarizerPass() { return new Scalarizer(); }
