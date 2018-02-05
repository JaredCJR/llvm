#include "llvm/PassPrediction/PassPrediction-Instrumentation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

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
    // Something like "void std::"
    std::istringstream iss(FuncName);
    std::string tmp;
    if (std::getline(iss, tmp, ' ')) {
      std::getline(iss, tmp, ' ');
      if (tmp.find(std::string("std::")) == 0) {
        return false;
      }
    }

    return true;
  }

  void PassPeeper(unsigned FeatureId) {
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
// This macro must be defined in the command of cmake
// ex. -DDAEMON_WORKER_ID="1"
#if defined(DAEMON_WORKER_ID)
    FeatureRecorder &InstrumentRec = FeatureRecorder::getInstance();
    InstrumentRec.setWorkerID(DAEMON_WORKER_ID);
    // Read ConnectionInfo to get ip and port for worker
    const char* env_p = std::getenv("LLVM_THESIS_InstrumentHome");
    if (!env_p) {
      exit(EXIT_FAILURE);
    }
    std::string ConnectionInfoLoc = std::string(env_p) + std::string("/Connection/ClangConnectInfo");
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
        // remove the spaces
        workerEntry.erase(std::remove(workerEntry.begin(), workerEntry.end(), ' '), workerEntry.end());
        tcpIP.erase(std::remove(tcpIP.begin(), tcpIP.end(), ' '), tcpIP.end());
        tcpPort.erase(std::remove(tcpPort.begin(), tcpPort.end(), ' '), tcpPort.end());
        WorkerDestMap[workerEntry] =
          std::make_pair(tcpIP, tcpPort);
      }
    }
#else
    #error "workerID should be defined in cmake build command for training framework."
    exit(EXIT_FAILURE);
#endif
  }

  void FeatureRecorder::writeAllFeatures(std::string Path) {
    std::ofstream file;
    // append the feature file, you should delete it manually.
    file.open(Path, std::ofstream::app);
		for (auto it : FeatureMap) {
		  file << getDemangledFunctionName(it.first) << " @ "; // record as demangled function name
      file << getFeatureAsString(it.first, std::string(", "));
			file << "\n";
		}
    file.close();
  }

}
