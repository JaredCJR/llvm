#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h>
#include <string>
#include <sstream>
#include <fstream>

namespace PassPrediction {
  std::string getDemangledFunctionName(std::string mangledName) {
    int status = -99;
    const char *OrigName = abi::__cxa_demangle(mangledName.c_str(), NULL, NULL, &status);
    if (OrigName == NULL) {
        OrigName = mangledName.c_str();
    }
    return std::string(OrigName);
  }

  // Whether the demangled function name is worth to extract features.
  bool isWorthToExtract(std::string &FuncName) {
    if (FuncName.find(std::string("std::")) == 0) {
      return false;
    }
    if (FuncName.find(std::string("__cxx")) == 0) {
      return false;
    }
    if (FuncName.find(std::string("__gnu_cxx::")) == 0) {
      return false;
    }
    if (FuncName.find(std::string("_GLOBAL")) == 0) {
      return false;
    }
    std::string cpp(".cpp");
    if (FuncName.rfind(cpp) == (FuncName.size() - cpp.size())) { //end with
      return false;
    }
    return true;
  }

  void PassPeeper(const std::string& file, unsigned FeatureId) {
    // Get the feature object, which is a singleton
    FeatureRecorder &InstrumentRec = FeatureRecorder::getInstance();
    // if instrumentation is enabled, record it.
    if (InstrumentRec.isInstrumentationActivated()){
      std::string demangledFuncName = 
        PassPrediction::getDemangledFunctionName(InstrumentRec.getCurrFuncName());
      // Do not extract features from "std::"
      if (isWorthToExtract(demangledFuncName)) {
        InstrumentRec.encounterFeatures(FeatureId);
      }
    }  
  }

  void BuildWorkerDestMap(
      std::unordered_map<std::string, std::pair<std::string, std::string> > &WorkerDestMap) {
#if defined(DAEMON_WORKER_ID)
    FeatureRecorder &InstrumentRec = FeatureRecorder::getInstance();
    InstrumentRec.setWorkerID(DAEMON_WORKER_ID);
    // Read ConnectionInfo to get ip and port for worker
    const char* env_p = std::getenv("LLVM_THESIS_InstrumentHome");
    if (!env_p) {
      exit(EXIT_FAILURE);
    }
    std::string ConnectionInfoLoc = std::string(env_p) + std::string("/training/ClangConnectInfo");
    std::ifstream stream(ConnectionInfoLoc);
    std::string line, workerEntry, tcpIP, tcpPort;
    std::getline(stream, line); //skip the first line
    // build IP and port map for workers
    while(stream) {
      std::getline(stream, line);
      if (stream) {
        std::istringstream ss(line);
        std::getline(ss, workerEntry, ',');
        std::getline(ss, tcpIP, ',');
        std::getline(ss, tcpPort, ',');
        WorkerDestMap[workerEntry] =
          std::make_pair(tcpIP, tcpPort);
      }
    }
#else
    #error "workerID should be defined in cmake build command for training framework."
    exit(EXIT_FAILURE);
#endif
  }

}
