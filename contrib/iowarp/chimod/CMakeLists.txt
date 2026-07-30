# faiss_ivf ChiMod — out-of-tree CLIO module hosting FAISS IVF search.
#
# Includable from a parent via add_subdirectory(). The parent is expected
# to provide:
#   IOWARP_INCLUDE_DIRS  - list of include dirs for the iowarp-core headers
#   IOWARP_LIB_DIR       - directory holding the iowarp-core .so libraries
#   faiss                - imported (or built) faiss target
cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(clio_faiss_ivf LANGUAGES CXX)
endif()

if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# iowarp-side link libraries for the client and runtime chimod libs.
# IOWARP_USE_DEV: use the exported imported targets (includes + CTP defines
# arrive transitively). Otherwise: absolute wheel .so paths.
if(IOWARP_USE_DEV)
  set(FAISS_IVF_CLIENT_IOWARP_LIBS ${IOWARP_CLIENT_TARGETS})
  set(FAISS_IVF_RUNTIME_IOWARP_LIBS ${IOWARP_RUNTIME_TARGETS})
else()
  set(FAISS_IVF_CLIENT_IOWARP_LIBS
    ${IOWARP_LIB_DIR}/libclio_run_cxx.so
    ${IOWARP_LIB_DIR}/libclio_ctp_host.so
    ${IOWARP_LIB_DIR}/libclio_admin_client.so
    ${IOWARP_LIB_DIR}/libclio_bdev_client.so
    pthread
    rt
  )
  set(FAISS_IVF_RUNTIME_IOWARP_LIBS
    ${FAISS_IVF_CLIENT_IOWARP_LIBS}
    ${IOWARP_LIB_DIR}/libclio_admin_runtime.so
    ${IOWARP_LIB_DIR}/libclio_bdev_runtime.so
    ${IOWARP_LIB_DIR}/libclio_cte_core_client.so
  )
endif()

#------------------------------------------------------------------------------
# Client library
#------------------------------------------------------------------------------
add_library(clio_faiss_ivf_client SHARED
  src/faiss_ivf_client.cc
)
target_compile_features(clio_faiss_ivf_client PUBLIC cxx_std_20)
if(IOWARP_COMPILE_DEFS)
  target_compile_definitions(clio_faiss_ivf_client PUBLIC ${IOWARP_COMPILE_DEFS})
endif()
target_include_directories(clio_faiss_ivf_client PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${IOWARP_INCLUDE_DIRS}
)
target_link_libraries(clio_faiss_ivf_client PUBLIC
  ${FAISS_IVF_CLIENT_IOWARP_LIBS}
)
set_target_properties(clio_faiss_ivf_client PROPERTIES
  OUTPUT_NAME clio_faiss_ivf_client
  BUILD_RPATH "${IOWARP_RPATH_DIR}"
  INSTALL_RPATH "${IOWARP_RPATH_DIR}"
)

#------------------------------------------------------------------------------
# Runtime library (loaded by the CLIO runtime as chimod "clio_faiss_ivf")
#------------------------------------------------------------------------------
add_library(clio_faiss_ivf_runtime SHARED
  src/faiss_ivf_runtime.cc
  src/autogen/faiss_ivf_lib_exec.cc
)
target_compile_features(clio_faiss_ivf_runtime PUBLIC cxx_std_20)
target_compile_definitions(clio_faiss_ivf_runtime PRIVATE CLIO_RUNTIME=1)
if(IOWARP_COMPILE_DEFS)
  target_compile_definitions(clio_faiss_ivf_runtime PUBLIC ${IOWARP_COMPILE_DEFS})
endif()
target_include_directories(clio_faiss_ivf_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${IOWARP_INCLUDE_DIRS}
)
target_link_libraries(clio_faiss_ivf_runtime PUBLIC
  ${FAISS_IVF_RUNTIME_IOWARP_LIBS}
  clio_faiss_ivf_client
  ${FAISS_LINK_TARGET}
  OpenMP::OpenMP_CXX
)
set_target_properties(clio_faiss_ivf_runtime PROPERTIES
  OUTPUT_NAME clio_faiss_ivf_runtime
  BUILD_RPATH "${IOWARP_RPATH_DIR}"
  INSTALL_RPATH "${IOWARP_RPATH_DIR}"
)
