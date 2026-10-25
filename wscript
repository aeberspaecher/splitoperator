#!/usr/bin/env python
#-*- coding:utf-8 -*-

from waflib.Build import BuildContext, CleanContext, InstallContext, UninstallContext
from waflib import Logs

top = "."
out = "build"


def options(opt):
    opt.load("compiler_c")
    opt.load("python")
    opt.load("cython")
    # add LAPACK/BLAS options:
    opt.add_option("--extra-flags", action="store", default="", dest="extraflags",
                   help="Extra flags to add to both C and Fortran compilation (e.g. '-march=native' with gcc/gfortran).")
    opt.add_option("--disable-openmp", action="store_true", dest="disable_openmp",
                   help="Disable use of OpenMP.")


def configure(conf):

    # setup different build environments; one each for debug/release builds
    # TODO: test Intel

    conf.load("compiler_c")
    conf.load("python")
    conf.check_python_version((2, 6))
    conf.load("cython")
    conf.check_python_headers()
    conf.check_python_module("numpy")
    conf.check_python_module("scipy")
    conf.check_python_module('matplotlib')
    try:
        conf.check_python_module('pyfftw')
    except conf.errors.ConfigurationError:
        print('could not find pyfftw (ignored)')
    try:
        conf.check_python_module('transparent_pyfftw')
    except conf.errors.ConfigurationError:
        print('could not find transparent_pyftw (ignored)')

    # make sure the NumPy headers are in the include path:
    import numpy as np
    numpy_header_path = np.get_include()
    conf.env.append_value("INCLUDES", numpy_header_path)

    # check if NumPy headers are found:
    #conf.check(header_name='numpy/arrayobject.h', features='c cprogram')
    # FIXME: for unknown reason, the check here fails

    extraflags = conf.options.extraflags.split()

    env = conf.env  # save old build env

    # debug builds:
    conf.setenv("debug", env)

    if("icc" in conf.env.CC[0]):
        conf.env.CFLAGS = ["-warn all", "-check all"] + extraflags
    elif("gcc" in conf.env.CC[0]):
        conf.env["CFLAGS"] = ["-g", "-O0", "-fPIC", "-Wall", "-Wextra"] + extraflags
    conf.env.CYTHONFLAGS = ["-Wextra", "--gdb", "--directive", "profile=True"]

    # release build:
    conf.setenv("release", env)

    if("icc" in conf.env.CC[0]):
        conf.env.CFLAGS = ["-fast"] + extraflags
    elif("gcc" in conf.env.CC[0]):
        conf.env.CFLAGS = ["-O2", "-fPIC"] + extraflags
    conf.env.CYTHONFLAGS = ["--directive", "profile=False",
                            "--directive", "cdivision=True",
                            "--directive", "boundscheck=False",
                            "--directive", "wraparound=False",
                            "--directive", "nonecheck=False"]


def build(bld):
    if(not bld.variant):
        bld.fatal("No build variant (release/debug) chosen! See help (./waf --help).")
    bld.recurse("so_schroedinger")


# create build variants - each comes with its own compiler/linker flags
for bldvariant in "debug release".split():
    for ctx in (BuildContext, CleanContext, InstallContext, UninstallContext):
        name = ctx.__name__.replace("Context", "").lower()
        class tmp(ctx):
            cmd = name + "_" + bldvariant
            variant = bldvariant
