# Chapter 5. TODO: ADD CHAPTER TITLE

This chapter covers the basics of memory management in CUDA devices. Topics include the different types of memory available on CUDA devices, and how to optimize memory access patterns for better performance.

An example of shared and global memory usage is provided via an optimized matrix multiplication kernel. By tiling the input matrices, we can improve memory access efficiency and highly improve the performance of the kernel.

P.S. The improved kernel function still doesn't utilize all the advanced features of CUDA. There are libraries like cuBLAS that provide very highly optimized matrix multiplication functions through APIs, and its recommended to use those libraries for production code.

TODO: add code that utilizes cuBLAS and compare its performance with a regular, non-optimized matrix multiplication kernel from ch. 3.
