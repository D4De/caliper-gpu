# CALIPER-FRAMEWORK CUDA Version

## What is Caliper?

Caliper is a framework to compute the MTTF (Mean Time To Failure) of a multicore heterogeneous grid of cores (multicore CPU).
This framework was developed by a Research team in Politecnico di Milano and the Sequential version of it can be found at: https://github.com/D4De/caliper

## How Caliper Works?

Caliper use Montecarlo Algorithm to approximate MTTF value within a certain "confidence interval".\
On our [Documentation Slides](./Documentation/Caliper-Cuda-Presentation.pdf) you can find more details about it.

## What have we done?

As a project for the GPU & Heterogeneous system course, we created a CUDA version of Caliper.
The final version had a 72x speedup [Intel I5 8thGen w.r.t a Tesla K80 on AWS].

(All the versions we implemented are explained in the [slides](./Documentation/Caliper-Cuda-Presentation.pdf), together with benchmarks and charts).
