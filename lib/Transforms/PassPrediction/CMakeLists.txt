add_llvm_library(LLVMPassPrediction
  PassPrediction-Instrumentation.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/PassPrediction

  )
