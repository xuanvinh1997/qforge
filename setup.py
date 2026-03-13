from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind11_build_ext
import os
import sys
import subprocess
import tempfile

cpp_dir = os.path.join("qforge", "cpp")


class CustomBuildExt(pybind11_build_ext):
    """Custom build_ext that handles .mm (Objective-C++) and .cu (CUDA) source files."""

    def build_extension(self, ext):
        # --- CUDA: pre-compile .cu files with nvcc and add .o to extra_objects ---
        cu_sources = [s for s in getattr(ext, 'cuda_sources', [])]
        if cu_sources:
            cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
            nvcc = os.path.join(cuda_home, 'bin', 'nvcc')

            # Find pybind11 include dirs
            pybind11_incs = []
            try:
                import pybind11
                pybind11_incs.append(pybind11.get_include())
            except ImportError:
                pass

            # Collect all include dirs for the extension
            inc_dirs = list(ext.include_dirs or []) + pybind11_incs

            # Build directory for object files
            build_temp = self.build_temp
            os.makedirs(build_temp, exist_ok=True)

            extra_objects = list(ext.extra_objects or [])
            for cu_src in cu_sources:
                obj = os.path.join(build_temp,
                                   os.path.basename(cu_src).replace('.cu', '.o'))
                cmd = [
                    nvcc, '-O3', '-std=c++17',
                    '--compiler-options', '-fPIC',
                    '-DQFORGE_HAS_CUDA',
                    '-arch=native',
                    '-c', cu_src,
                    '-o', obj,
                ] + [f'-I{d}' for d in inc_dirs]
                print(f'[CUDA] {" ".join(cmd)}')
                subprocess.check_call(cmd)
                extra_objects.append(obj)
            ext.extra_objects = extra_objects

        # --- Metal: handle .mm (Objective-C++) source files ---
        if any(src.endswith('.mm') for src in ext.sources):
            original_src_extensions = self.compiler.src_extensions
            if '.mm' not in self.compiler.src_extensions:
                self.compiler.src_extensions.append('.mm')

            original_compile = self.compiler._compile

            def patched_compile(obj, src, ext_param, cc_args, extra_postargs, pp_opts):
                if src.endswith('.mm'):
                    try:
                        original_compiler = self.compiler.compiler_so
                        self.compiler.set_executables(
                            compiler_so=original_compiler + ['-x', 'objective-c++']
                        )
                    except Exception:
                        pass
                return original_compile(obj, src, ext_param, cc_args, extra_postargs, pp_opts)

            self.compiler._compile = patched_compile

        super().build_extension(ext)


# --- CPU backend (always built) ---
cpu_sources = [
    os.path.join(cpp_dir, "src", f)
    for f in [
        "state_vector.cpp",
        "gates_single.cpp",
        "gates_controlled.cpp",
        "gates_swap.cpp",
        "gates_noise.cpp",
        "measurement.cpp",
        "data_ops.cpp",
        "cpu_backend.cpp",
    ]
] + [os.path.join(cpp_dir, "bindings", "py_qforge.cpp")]

ext_modules = [
    Pybind11Extension(
        "qforge._qforge_core",
        cpu_sources,
        include_dirs=[os.path.join(cpp_dir, "include")],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        language="c++",
    ),
]

# --- MPS/DMRG backend (requires Eigen3, always attempted) ---
def _find_eigen3():
    candidates = [
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
    # Check conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.insert(0, os.path.join(conda_prefix, "include", "eigen3"))
    try:
        result = subprocess.run(
            ["brew", "--prefix", "eigen"], capture_output=True, text=True)
        if result.returncode == 0:
            prefix = result.stdout.strip()
            candidates.insert(0, os.path.join(prefix, "include", "eigen3"))
    except Exception:
        pass
    for path in candidates:
        if os.path.isdir(os.path.join(path, "Eigen")):
            return path
    return None

_eigen_dir = _find_eigen3()
_mps_sources = [
    os.path.join(cpp_dir, "src", f)
    for f in [
        "mps_core.cpp",
        "mps_gates.cpp",
        "mps_meas.cpp",
        "mpo_core.cpp",
        "dmrg_sweep.cpp",
    ]
] + [os.path.join(cpp_dir, "bindings", "py_mps.cpp")]

_mps_include_dirs = [os.path.join(cpp_dir, "include")]
if _eigen_dir:
    _mps_include_dirs.append(_eigen_dir)

ext_modules.append(
    Pybind11Extension(
        "qforge._qforge_mps",
        _mps_sources,
        include_dirs=_mps_include_dirs,
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-std=c++17"],
        language="c++",
    )
)

# --- Metal backend (macOS only, opt-in via QFORGE_METAL=1) ---
if sys.platform == 'darwin' and os.environ.get('QFORGE_METAL', '0') == '1':
    pybind11_includes = []
    try:
        import pybind11
        pybind11_includes.append(pybind11.get_include())
        pybind11_includes.append(pybind11.get_include(user=True))
    except ImportError:
        pass

    metal_sources = [
        os.path.join(cpp_dir, "src", "metal_backend.mm"),
        os.path.join(cpp_dir, "bindings", "py_metal.mm"),
    ]

    ext_modules.append(
        Extension(
            "qforge._qforge_metal",
            metal_sources,
            include_dirs=[
                os.path.join(cpp_dir, "include"),
            ] + pybind11_includes,
            extra_compile_args=[
                "-O3", "-std=c++17", "-ffast-math",
                "-fobjc-arc",
                "-x", "objective-c++",
            ],
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
            ],
        )
    )

# --- CUDA backend (opt-in via QFORGE_CUDA=1) ---
if os.environ.get('QFORGE_CUDA', '0') == '1':
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    cuda_include = os.path.join(cuda_home, 'include')
    cuda_lib = os.path.join(cuda_home, 'lib64')

    cuda_ext = Pybind11Extension(
        "qforge._qforge_cuda",
        sources=[
            os.path.join(cpp_dir, "src", "cuda_backend.cpp"),
            os.path.join(cpp_dir, "bindings", "py_cuda.cpp"),
        ],
        include_dirs=[
            os.path.join(cpp_dir, "include"),
            cuda_include,
        ],
        library_dirs=[cuda_lib],
        libraries=["cudart"],
        extra_compile_args=["-O3", "-DQFORGE_HAS_CUDA"],
        language="c++",
    )
    # Attach .cu sources for the custom build step
    cuda_ext.cuda_sources = [
        os.path.join(cpp_dir, "src", "cuda_kernels.cu"),
    ]
    ext_modules.append(cuda_ext)

setup(
    name="qforge",
    version="3.0.0",
    description="qforge - Large-Scale Quantum Simulation Framework",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    install_requires=["numpy", "pybind11>=2.10"],
    python_requires=">=3.8",
)
