from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind11_build_ext
import os
import sys
import subprocess
import tempfile

cpp_dir = os.path.join("qforge", "cpp")

_IS_WIN = sys.platform == "win32"


def _compile_args_base():
    """Optimization flags for the current compiler."""
    if _IS_WIN:
        return ["/O2", "/fp:fast", "/EHsc"]
    return ["-O3", "-march=native", "-ffast-math"]


def _compile_args_cpp17():
    """Optimization + C++17 flags for the current compiler."""
    if _IS_WIN:
        return ["/O2", "/fp:fast", "/EHsc", "/std:c++17"]
    return ["-O3", "-march=native", "-ffast-math", "-std=c++17"]


def _find_cuda_home():
    """Locate the CUDA toolkit installation directory."""
    # Explicit env vars (highest priority)
    for var in ('CUDA_HOME', 'CUDA_PATH'):
        val = os.environ.get(var)
        if val and os.path.isdir(val):
            return val
    # Windows: scan Program Files for versioned CUDA installs
    if _IS_WIN:
        import glob
        pattern = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*'
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]  # newest version
    # Linux/macOS fallbacks
    for path in ('/usr/local/cuda', '/opt/cuda'):
        if os.path.isdir(path):
            return path
    return '/usr/local/cuda'


def _cuda_lib_dir(cuda_home):
    """Return the CUDA library directory for the current platform."""
    if _IS_WIN:
        return os.path.join(cuda_home, 'lib', 'x64')
    return os.path.join(cuda_home, 'lib64')


def _nvcc_host_flags():
    """Return nvcc --compiler-options flags appropriate for the host compiler."""
    if _IS_WIN:
        # MSVC: use /MD (dynamic CRT, matches Python's build)
        return ['--compiler-options', '/MD']
    return ['--compiler-options', '-fPIC']


def _obj_ext():
    """Object file extension for the current platform."""
    return '.obj' if _IS_WIN else '.o'


class CustomBuildExt(pybind11_build_ext):
    """Custom build_ext that handles .mm (Objective-C++) and .cu (CUDA) sources,
    and falls back gracefully when the compiler is unavailable."""

    def build_extension(self, ext):
        # --- CUDA: pre-compile .cu files with nvcc and add objects to linker ---
        cu_sources = [s for s in getattr(ext, 'cuda_sources', [])]
        if cu_sources:
            cuda_home = _find_cuda_home()
            nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
            if _IS_WIN:
                nvcc += '.exe'

            pybind11_incs = []
            try:
                import pybind11
                pybind11_incs.append(pybind11.get_include())
            except ImportError:
                pass

            inc_dirs = list(ext.include_dirs or []) + pybind11_incs
            build_temp = self.build_temp
            os.makedirs(build_temp, exist_ok=True)

            extra_objects = list(ext.extra_objects or [])
            for cu_src in cu_sources:
                obj = os.path.join(
                    build_temp,
                    os.path.basename(cu_src).replace('.cu', _obj_ext()),
                )
                cmd = [
                    nvcc, '-O3', '-std=c++17',
                    *_nvcc_host_flags(),
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

        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"\n  [WARNING] Could not build C++ extension '{ext.name}':")
            print(f"  {e}")
            print("  Qforge will run in pure-Python mode for this extension.")
            print("  To enable C++ acceleration on Windows, install Microsoft C++ Build Tools:")
            print("  https://visualstudio.microsoft.com/visual-cpp-build-tools/\n")

    def run(self):
        try:
            super().run()
        except Exception as e:
            print(f"\n  [WARNING] C++ build step failed: {e}")
            print("  Package will be installed without C++ extensions.\n")


# --- CPU backend ---
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
        extra_compile_args=_compile_args_base(),
        language="c++",
    ),
]

# --- MPS/DMRG backend (requires Eigen3) ---
def _find_eigen3():
    candidates = [
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]

    # Windows: vcpkg, conda, common install locations
    if _IS_WIN:
        candidates += [
            r"C:\vcpkg\installed\x64-windows\include\eigen3",
            r"C:\vcpkg\installed\x86-windows\include\eigen3",
            r"C:\tools\eigen3",
            r"C:\eigen3",
        ]
        # Conan cache
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.append(os.path.join(local_app, "conan", "data", "eigen", "include"))
        # vcpkg root env var
        vcpkg_root = os.environ.get("VCPKG_ROOT", "")
        if vcpkg_root:
            candidates.insert(0, os.path.join(vcpkg_root, "installed", "x64-windows", "include", "eigen3"))
            candidates.insert(0, os.path.join(vcpkg_root, "installed", "x64-windows", "include"))

    # conda/mamba environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.insert(0, os.path.join(conda_prefix, "include", "eigen3"))
        candidates.insert(0, os.path.join(conda_prefix, "Library", "include", "eigen3"))  # Windows conda

    # Homebrew (macOS)
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
        extra_compile_args=_compile_args_cpp17(),
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

    ext_modules.append(
        Extension(
            "qforge._qforge_metal",
            sources=[
                os.path.join(cpp_dir, "src", "metal_backend.mm"),
                os.path.join(cpp_dir, "bindings", "py_metal.mm"),
            ],
            include_dirs=[os.path.join(cpp_dir, "include")] + pybind11_includes,
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
    _cuda_home = _find_cuda_home()
    _cuda_include = os.path.join(_cuda_home, 'include')
    _cuda_lib = _cuda_lib_dir(_cuda_home)

    cuda_ext = Pybind11Extension(
        "qforge._qforge_cuda",
        sources=[
            os.path.join(cpp_dir, "src", "cuda_backend.cpp"),
            os.path.join(cpp_dir, "bindings", "py_cuda.cpp"),
        ],
        include_dirs=[os.path.join(cpp_dir, "include"), _cuda_include],
        library_dirs=[_cuda_lib],
        libraries=["cudart"],
        extra_compile_args=_compile_args_base() + ["/DQFORGE_HAS_CUDA" if _IS_WIN else "-DQFORGE_HAS_CUDA"],
        language="c++",
    )
    cuda_ext.cuda_sources = [os.path.join(cpp_dir, "src", "cuda_kernels.cu")]
    ext_modules.append(cuda_ext)


setup(
    name="qforge",
    version="3.0.0",
    description="qforge - Large-Scale Quantum Simulation Framework",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    install_requires=["numpy"],
    python_requires=">=3.8",
)
