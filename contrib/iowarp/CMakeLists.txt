# FAISS x IOWarp contribution — standalone CMake project.
#
# Binaries link against the iowarp-core pip wheel's native libraries
# (discovered through the Python package) and compile against a pinned
# clio-core source checkout that matches the wheel version exactly
# (v2.1.0). There is no clio-core CMake package in the wheel, hence the
# manual discovery below.
#
# Configure:
#   cmake -B build . -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_PREFIX_PATH="$HOME/faiss-install" \
#     -DIOWARP_HEADERS_DIR=/path/to/clio-core-v2.1.0
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(faiss_iowarp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- clio-core provider: dev source install (find_package) OR pip wheel --
# IOWARP_USE_DEV=ON  : consume a clio-core *dev* source install through its
#   exported find_package(iowarp-core) package. The imported targets carry the
#   ABI-critical CTP compile definitions + include dirs transitively, so no
#   IOWARP_COMPILE_DEFS / IOWARP_INCLUDE_DIRS list is maintained by hand.
#   Needs -DCMAKE_PREFIX_PATH="<dev-install>;<dep-prefix>/usr".
# IOWARP_USE_DEV=OFF : legacy path — pip wheel .so + pinned v2.1.0 headers.
option(IOWARP_USE_DEV "Link against a clio-core dev source install via find_package(iowarp-core)" OFF)

if(IOWARP_USE_DEV)
    # The exported imported targets reference Threads::Threads transitively.
    find_package(Threads REQUIRED)
    find_package(iowarp-core REQUIRED)
    # Client-side targets (bench, tools, chimod client). ctp::* + include dirs
    # + CTP defines arrive transitively from these imported targets.
    set(IOWARP_CLIENT_TARGETS
        clio::cte::core_client
        clio::run::cxx
        clio::run::admin_client
        clio::run::bdev_client
        pthread rt)
    # Runtime chimod additionally needs the runtime-side module libs.
    set(IOWARP_RUNTIME_TARGETS
        clio::cte::core_client
        clio::run::cxx
        clio::run::admin_client clio::run::admin_runtime
        clio::run::bdev_client  clio::run::bdev_runtime
        pthread rt)
    set(IOWARP_INCLUDE_DIRS "")          # provided transitively
    set(IOWARP_COMPILE_DEFS "")          # provided transitively
    set(IOWARP_LINK_LIBS ${IOWARP_CLIENT_TARGETS})
    set(IOWARP_RPATH_DIR "${IOWARP_CORE_LIB_DIR}")
    message(STATUS "iowarp-core: DEV source install at ${IOWARP_CORE_PREFIX}")
else()
    # --- iowarp-core pip wheel: native libraries + binaries -----------------
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(NOT IOWARP_WHEEL_DIR)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                "import iowarp_core, os; print(os.path.dirname(iowarp_core.__file__))"
            OUTPUT_VARIABLE IOWARP_WHEEL_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _iowarp_rc)
        if(NOT _iowarp_rc EQUAL 0)
            message(FATAL_ERROR
                "iowarp-core pip wheel not found (pip install iowarp-core), "
                "or pass -DIOWARP_WHEEL_DIR=<site-packages>/iowarp_core")
        endif()
    endif()
    set(IOWARP_LIB_DIR "${IOWARP_WHEEL_DIR}/lib")
    message(STATUS "iowarp-core wheel libs: ${IOWARP_LIB_DIR}")

    # --- clio-core headers (pinned to the wheel's source tag) ----------------
    set(IOWARP_HEADERS_DIR "/mnt/common/cvazquezperezdelacru/clio-core-v2.1.0"
        CACHE PATH "clio-core source checkout matching the wheel version")
    set(IOWARP_INCLUDE_DIRS
        "${IOWARP_HEADERS_DIR}/context-runtime/include"
        "${IOWARP_HEADERS_DIR}/context-runtime/modules/admin/include"
        "${IOWARP_HEADERS_DIR}/context-runtime/modules/bdev/include"
        "${IOWARP_HEADERS_DIR}/context-transfer-engine/core/include"
        "${IOWARP_HEADERS_DIR}/context-transport-primitives/include"
        # Vendored headers for deps the wheel statically links but doesn't ship
        # headers for: zmq.h 4.3.5, msgpack-c 6.1.0, cereal 1.3.2 (versions from
        # installers/pip/build_deps_manylinux.sh at the wheel's tag).
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zmq"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/msgpack"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal")
    foreach(_dir IN LISTS IOWARP_INCLUDE_DIRS)
        if(NOT EXISTS "${_dir}")
            message(FATAL_ERROR "Missing include dir: ${_dir} "
                "(set -DIOWARP_HEADERS_DIR to a clio-core checkout at the wheel's tag)")
        endif()
    endforeach()

    # Wheel client libraries every target needs.
    set(IOWARP_CLIENT_LIBS
        "${IOWARP_LIB_DIR}/libclio_cte_core_client.so"
        "${IOWARP_LIB_DIR}/libclio_run_cxx.so"
        "${IOWARP_LIB_DIR}/libclio_ctp_host.so"
        "${IOWARP_LIB_DIR}/libclio_admin_client.so"
        "${IOWARP_LIB_DIR}/libclio_bdev_client.so"
        pthread rt)

    # Compile definitions the in-tree clio-core build propagates to consumers,
    # evaluated with the wheel's exact build options (pyproject.toml: ZMQ on,
    # OpenMP/MPI/GPU off, pthreads model off -> StdThread, CTP_LOG_LEVEL=1).
    set(IOWARP_COMPILE_DEFS
        CTP_ENABLE_LIGHTBEAM=1
        CTP_ENABLE_ZMQ=1
        CTP_ENABLE_LIBFABRIC=0
        CTP_ENABLE_THALLIUM=0
        CTP_ENABLE_NIXL=0
        CTP_ENABLE_NVSHMEM=0
        CTP_ENABLE_OPENMP=0
        CTP_ENABLE_PTHREADS=0
        CTP_ENABLE_WINDOWS_THREADS=0
        CTP_DEFAULT_THREAD_MODEL=ctp::thread::StdThread
        CTP_DEFAULT_THREAD_MODEL_GPU=ctp::thread::StdThread
        CTP_ENABLE_MPI=0
        CTP_COMPILER_MSVC=0
        CTP_COMPILER_GNU=1
        CTP_ENABLE_WINDOWS_SYSINFO=0
        CTP_ENABLE_PROCFS_SYSINFO=1
        CTP_ENABLE_DOXYGEN=0
        CTP_DEBUG_LOCK=0
        CTP_DEFAULT_ALLOC_T=ctp::ipc::ThreadLocalAllocator
        CTP_ENABLE_DLL_EXPORT=1
        CTP_LOG_LEVEL=1)
    set(IOWARP_LINK_LIBS ${IOWARP_CLIENT_LIBS})
    set(IOWARP_RPATH_DIR "${IOWARP_LIB_DIR}")
endif()

function(iowarp_target tgt)
    if(IOWARP_INCLUDE_DIRS)
        target_include_directories(${tgt} PRIVATE ${IOWARP_INCLUDE_DIRS})
    endif()
    if(IOWARP_COMPILE_DEFS)
        target_compile_definitions(${tgt} PRIVATE ${IOWARP_COMPILE_DEFS})
    endif()
    target_link_libraries(${tgt} PRIVATE ${IOWARP_LINK_LIBS})
    set_target_properties(${tgt} PROPERTIES
        BUILD_RPATH "${IOWARP_RPATH_DIR}"
        INSTALL_RPATH "${IOWARP_RPATH_DIR}")
endfunction()

# --- targets --------------------------------------------------------------
# Smoke test: wheel link sanity, no FAISS dependency.
add_executable(cte_smoke src/cte_smoke.cpp)
iowarp_target(cte_smoke)

# Tools + benchmark (need faiss and OpenMP).
find_package(faiss REQUIRED)
find_package(OpenMP REQUIRED)

# With FAISS_OPT_LEVEL=avx512 the SIMD kernels live in the faiss_avx512
# target; plain `faiss` is the generic-kernel build. Link the fastest
# variant available.
if(TARGET faiss_avx512)
    set(FAISS_LINK_TARGET faiss_avx512)
elseif(TARGET faiss_avx2)
    set(FAISS_LINK_TARGET faiss_avx2)
else()
    set(FAISS_LINK_TARGET faiss)
endif()
if(NOT FAISS_LINK_TARGET STREQUAL "faiss")
    # Upstream quirk: only the generic `faiss` imported target carries
    # INTERFACE_INCLUDE_DIRECTORIES; copy it onto the SIMD variant.
    get_target_property(_faiss_incs faiss INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${FAISS_LINK_TARGET} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_faiss_incs}")
endif()
message(STATUS "Linking FAISS variant: ${FAISS_LINK_TARGET}")

# Shared helpers: CTE client bootstrap + IVF->CTE ingestion.
add_library(iowarp_cte STATIC src/cte_client.cpp src/ivf_cte_ingest.cpp)
iowarp_target(iowarp_cte)
target_include_directories(iowarp_cte PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(iowarp_cte PRIVATE ${FAISS_LINK_TARGET})
set_target_properties(iowarp_cte PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(ivf_to_iowarp src/ivf_to_iowarp.cpp)
iowarp_target(ivf_to_iowarp)
target_link_libraries(ivf_to_iowarp PRIVATE iowarp_cte ${FAISS_LINK_TARGET})

add_executable(bench_ivf_qps benchmarks/bench_ivf_qps.cpp)
iowarp_target(bench_ivf_qps)
target_link_libraries(bench_ivf_qps PRIVATE iowarp_cte ${FAISS_LINK_TARGET}
    OpenMP::OpenMP_CXX)

# Mixed 80/20 read/write benchmark (performance study, Exp 2).
add_executable(bench_ivf_mixed benchmarks/bench_ivf_mixed.cpp)
iowarp_target(bench_ivf_mixed)
target_link_libraries(bench_ivf_mixed PRIVATE iowarp_cte ${FAISS_LINK_TARGET}
    OpenMP::OpenMP_CXX)

# faiss_ivf ChiMod (optional subdirectory; needs the same wheel/header setup).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chimod/CMakeLists.txt"
   AND NOT IOWARP_SKIP_CHIMOD)
    add_subdirectory(chimod)
    # ivf_to_iowarp uses the chimod headers only for the inline
    # ListOwnerContainer hash (owner-local multi-node ingest).
    foreach(_bench bench_ivf_qps bench_ivf_mixed ivf_to_iowarp)
        target_compile_definitions(${_bench} PRIVATE HAVE_FAISS_IVF_CHIMOD)
        target_include_directories(${_bench} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/chimod/include")
        target_link_libraries(${_bench} PRIVATE clio_faiss_ivf_client)
    endforeach()
endif()
