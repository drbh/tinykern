from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tinykern',
    version='0.1.0',
    author='David Holtz (drbh)',
    description='Minimal CUDA kernel examples',
    ext_modules=[
        CUDAExtension(
            name='matmul_kernel',
            sources=[
                'tinykern/csrc/matmul_wrapper.cpp', 
                'tinykern/csrc/matmul_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-use_fast_math']
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch==2.1.2',
    ],
)
