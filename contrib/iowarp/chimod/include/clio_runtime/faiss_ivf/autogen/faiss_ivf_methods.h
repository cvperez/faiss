#ifndef CHIMAERA_FAISS_IVF_AUTOGEN_METHODS_H_
#define CHIMAERA_FAISS_IVF_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

/**
 * Auto-generated method definitions for faiss_ivf
 */

namespace clio::run::faiss_ivf {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST chi::u32 kCreate = 0;
GLOBAL_CROSS_CONST chi::u32 kDestroy = 1;
GLOBAL_CROSS_CONST chi::u32 kMonitor = 9;

// faiss_ivf-specific methods
GLOBAL_CROSS_CONST chi::u32 kOpenIndex = 10;
GLOBAL_CROSS_CONST chi::u32 kSearch = 11;
GLOBAL_CROSS_CONST chi::u32 kStats = 12;

GLOBAL_CROSS_CONST chi::u32 kMaxMethodId = 13;

inline const std::vector<std::string>& GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[0] = "Create";
    v[1] = "Destroy";
    v[9] = "Monitor";
    v[10] = "OpenIndex";
    v[11] = "Search";
    v[12] = "Stats";
    return v;
  }();
  return names;
}
}  // namespace Method

}  // namespace clio::run::faiss_ivf

#endif  // CHIMAERA_FAISS_IVF_AUTOGEN_METHODS_H_
