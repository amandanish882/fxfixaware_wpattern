"""
Build script for fx_pricing_kernel C++ extension (RELEASE).

Usage:
    pip install .
    # or
    python setup.py build_ext --inplace

Debug builds:
    Windows: python setup_debug.py build_ext --inplace
    macOS:   python setup_debug_mac.py build_ext --inplace

The built .pyd (Windows) or .so (Linux/macOS) is copied to the project root
so it can be imported directly: `import fx_pricing_kernel`.
"""

import os
import sys
import shutil
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# ── Locate pybind11 ──────────────────────────────────────────────
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("ERROR: pybind11 not found. Install with: pip install pybind11")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(HERE, "include")
BINDINGS_DIR = os.path.join(HERE, "bindings")
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


class BuildExt(build_ext):
    """Custom build that copies the .pyd/.so to project root after build."""

    def build_extensions(self):
        # MSVC-specific flags on Windows
        if self.compiler.compiler_type == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args = [
                    "/std:c++17",   # C++17 standard
                    "/O2",          # Full optimization
                    "/EHsc",        # Exception handling
                    "/W3",          # Warning level 3
                    "/DNDEBUG",     # Release mode
                ]
        else:
            # GCC / Clang
            for ext in self.extensions:
                ext.extra_compile_args = [
                    "-std=c++17",
                    "-O3",
                    "-fPIC",
                    "-Wall",
                    "-Wextra",
                    "-DNDEBUG",
                ]
        build_ext.build_extensions(self)

    def run(self):
        build_ext.run(self)

        # Copy built extension to project root
        for ext in self.extensions:
            built_path = self.get_ext_fullpath(ext.name)
            if os.path.exists(built_path):
                dest = os.path.join(PROJECT_ROOT, os.path.basename(built_path))
                print(f"Copying {built_path} -> {dest}")
                shutil.copy2(built_path, dest)
                print(f"fx_pricing_kernel installed at: {dest}")


# ── Extension definition ─────────────────────────────────────────
ext_modules = [
    Extension(
        name="fx_pricing_kernel",
        sources=[os.path.join(BINDINGS_DIR, "pybind_module.cpp")],
        include_dirs=[
            INCLUDE_DIR,
            pybind11_include,
            pybind11.get_include(user=True),
        ],
        language="c++",
    ),
]

setup(
    name="fx_pricing_kernel",
    version="1.0.0",
    author="Quant 300 FX Project",
    description="C++ kernel for FX forward pricing, curve bootstrapping, and execution",
    long_description="Header-only C++17 library with pybind11 bindings for "
                     "discount curve construction, CIP-based FX forward pricing, "
                     "and Almgren-Chriss optimal execution.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
    zip_safe=False,
)
