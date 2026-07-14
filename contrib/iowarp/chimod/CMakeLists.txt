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

# Libraries common to client and runtime
set(FAISS_IVF_COMMON_LIBS
  ${IOWARP_LIB_DIR}/libclio_run_cxx.so
  ${IOWARP_LIB_DIR}/libclio_ctp_host.so
  ${IOWARP_LIB_DIR}/libclio_admin_client.so
  ${IOWARP_LIB_DIR}/libclio_bdev_client.so
  pthread
  rt
)

#------------------------------------------------------------------------------
# Client library
#------------------------------------------------------------------------------
add_library(clio_faiss_ivf_client SHARED
  src/faiss_ivf_client.cc
)
target_compile_features(clio_faiss_ivf_client PUBLIC cxx_std_20)
target_compile_definitions(clio_faiss_ivf_client PUBLIC ${IOWARP_COMPILE_DEFS})
target_include_directories(clio_faiss_ivf_client PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${IOWARP_INCLUDE_DIRS}
)
target_link_libraries(clio_faiss_ivf_client PUBLIC
  ${FAISS_IVF_COMMON_LIBS}
)
set_target_properties(clio_faiss_ivf_client PROPERTIES
  OUTPUT_NAME clio_faiss_ivf_client
  BUILD_RPATH "${IOWARP_LIB_DIR}"
  INSTALL_RPATH "${IOWARP_LIB_DIR}"
)

#------------------------------------------------------------------------------
# Runtime library (loaded by the CLIO runtime as chimod "clio_faiss_ivf")
#------------------------------------------------------------------------------
add_library(clio_faiss_ivf_runtime SHARED
  src/faiss_ivf_runtime.cc
  src/autogen/faiss_ivf_lib_exec.cc
)
target_compile_features(clio_faiss_ivf_runtime PUBLIC cxx_std_20)
target_compile_definitions(clio_faiss_ivf_runtime PRIVATE CHIMAERA_RUNTIME=1)
target_compile_definitions(clio_faiss_ivf_runtime PUBLIC ${IOWARP_COMPILE_DEFS})
target_include_directories(clio_faiss_ivf_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${IOWARP_INCLUDE_DIRS}
)
target_link_libraries(clio_faiss_ivf_runtime PUBLIC
  ${FAISS_IVF_COMMON_LIBS}
  ${IOWARP_LIB_DIR}/libclio_admin_runtime.so
  ${IOWARP_LIB_DIR}/libclio_bdev_runtime.so
  ${IOWARP_LIB_DIR}/libclio_cte_core_client.so
  clio_faiss_ivf_client
  ${FAISS_LINK_TARGET}
  OpenMP::OpenMP_CXX
)
set_target_properties(clio_faiss_ivf_runtime PROPERTIES
  OUTPUT_NAME clio_faiss_ivf_runtime
  BUILD_RPATH "${IOWARP_LIB_DIR}"
  INSTALL_RPATH "${IOWARP_LIB_DIR}"
)
