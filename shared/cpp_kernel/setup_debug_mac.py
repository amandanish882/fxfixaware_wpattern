"""
setup_debug_mac.py
==================
Debug build of the fx_pricing_kernel C++ extension on macOS (clang++).

Compiles with debug symbols (-g) and no optimisation (-O0) so you can:
  - Set breakpoints in curve_engine.h, fx_pricer.h, execution_engine.h
  - Step through C++ line by line from VSCode (lldb)
  - Inspect variable values at each node of the bootstrap loop

Usage:
    cd shared/cpp_kernel
    /Users/amand/miniconda3/bin/python3 setup_debug_mac.py build_ext --inplace

Produces (next to this script):
    fx_pricing_kernel.cpython-<ver>-darwin.so
    fx_pricing_kernel.cpython-<ver>-darwin.so.dSYM/    (debug symbol bundle)

The build also copies the .so to the project root so `import fx_pricing_kernel`
finds it.
"""

import os
import shutil
import sys
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(HERE, "include")
BINDINGS_DIR = os.path.join(HERE, "bindings")
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


class BuildExtDebugMac(build_ext):
    """macOS clang debug build that copies the .so to project root after build."""

    def build_extensions(self):
        for ext in self.extensions:
            ext.extra_compile_args = [
                "-O0",                  # no optimisation, line-by-line stepping
                "-g",                   # DWARF debug symbols
                "-std=c++17",
                "-fvisibility=hidden",  # match pybind11 default; smaller binary
                "-DDEBUG",
            ]
            ext.extra_link_args = [
                "-g",                   # keep debug info in the .so
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
                print(f"fx_pricing_kernel (DEBUG/macOS) installed at: {dest}")


ext_modules = [
    Extension(
        "fx_pricing_kernel",
        [os.path.join(BINDINGS_DIR, "pybind_module.cpp")],
        include_dirs=[INCLUDE_DIR, pybind11.get_include()],
        language="c++",
    ),
]

setup(
    name="fx_pricing_kernel_debug",
    version="1.0.0.dev1",
    description="Debug build of FX C++ pricing kernel (macOS / clang)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtDebugMac},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10"],
    zip_safe=False,
)
