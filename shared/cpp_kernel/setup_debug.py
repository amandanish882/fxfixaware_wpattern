"""
Debug build script for fx_pricing_kernel C++ extension (Windows / MSVC).

Usage:
    python setup_debug.py build_ext --inplace

For macOS / clang debug builds, see setup_debug_mac.py.

Builds with debug symbols (/Zi), no optimization (/Od), assertions enabled.
Attach Visual Studio debugger to the Python process to set breakpoints
in curve_engine.h, fx_pricer.h, execution_engine.h.
"""

import os
import sys
import shutil
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("ERROR: pybind11 not found. Install with: pip install pybind11")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(HERE, "include")
BINDINGS_DIR = os.path.join(HERE, "bindings")
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


class BuildExtDebug(build_ext):
    """Debug build with symbols, no optimization."""

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args = [
                    "/std:c++17",
                    "/Od",          # No optimization
                    "/Zi",          # Debug info
                    "/EHsc",
                    "/W4",          # Higher warning level
                    "/MDd",         # Debug runtime
                    "/FS",          # Serialize PDB writes
                ]
                ext.extra_link_args = ["/DEBUG:FULL"]
        else:
            for ext in self.extensions:
                ext.extra_compile_args = [
                    "-std=c++17",
                    "-O0",
                    "-g",
                    "-fPIC",
                    "-Wall",
                    "-Wextra",
                ]
        build_ext.build_extensions(self)

    def run(self):
        build_ext.run(self)
        for ext in self.extensions:
            built_path = self.get_ext_fullpath(ext.name)
            if os.path.exists(built_path):
                dest = os.path.join(PROJECT_ROOT, os.path.basename(built_path))
                print(f"Copying {built_path} -> {dest}")
                shutil.copy2(built_path, dest)
                print(f"fx_pricing_kernel (DEBUG) installed at: {dest}")


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
    version="1.0.0.dev1",
    author="Quant 300 FX Project",
    description="C++ kernel for FX pricing (DEBUG BUILD)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtDebug},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
    zip_safe=False,
)
