# Test-your-CPU-and-GPU
Most of us enjoy testing our hardware capabilities.

This software intends to measure the computational performance of your CPU or GPU using mathematical workloads. It has two separate benchmarks: one for the CPU and one for the GPU. Both tests use computation rather than graphics rendering or operations with files, therefore giving you insights about your raw processing speed.

The CPU benchmark uses a prime number sieve algorithm. It continuously calculates prime numbers for 2 minutes, measuring how many primes can be found per second. Because the workload is entirely mathematical and distributed across all CPU cores, it takes most of the processor's attention and provides a realistic measure of multi-core performance.

The GPU benchmark uses an OpenCL kernel to perform a large number of floating-point multiply-add operations in parallel. It stresses the GPU’s compute units, tests memory bandwidth, and outputs the total number of floating-point operations executed per second (GFLOPS). The result shows the GPU’s ability to handle numerical tasks like simulations and machine learning.

Each test executes for any duration, if modified, and gives you information about these results. They can be used to compare different systems, or even test your cooling efficiency.

Changes are expected in the nearby future, including instructions. You are welcome to make your own modifications to this software.
