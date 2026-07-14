/*
 * faiss_ivf ChiMod — client implementation.
 *
 * Contains global variables and singletons for client-side code.
 */

#include "clio_runtime/faiss_ivf/faiss_ivf_client.h"
#include "clio_runtime/faiss_ivf/faiss_ivf_tasks.h"

namespace clio::run::faiss_ivf {

// Define static constexpr member for proper linkage when address is taken
constexpr const char* CreateParams::chimod_lib_name;

// Client implementation is mostly header-only
// This file exists for any global client-side state or initialization

}  // namespace clio::run::faiss_ivf
