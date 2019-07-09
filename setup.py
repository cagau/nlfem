from distutils.core import setup, Extension

modul = Extension("bermuda", sources=["bermuda.cpp"])
setup(
    name = "PyBermuda",
    version = "1.0",
    description = "Draw some game.",
    ext_modules = [modul]
)