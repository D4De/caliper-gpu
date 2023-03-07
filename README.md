# CALIPER-FRAMEWORK CUDA Version

## What is Caliper?

Caliper is an OpenSource framework to compute MTTF (Mean Time To Failure) of a multicore homogeneous grid of cores (MultiCore CPU).
This framework was developed by a Research Team in Politecnico di Milano and it's Sequential implementation can be fount at: https://github.com/D4De/caliper

## How Caliper Works?
Caliper use Montecarlo Algorithm to approximate MTTF value within a certain "confidence interval".

## Scope of our caliper-gpu:

Sequential implementation of caliper has some time limitation since it become unfeasible to simulate big grid of cores.
A grid of core 42 * 42 take more than 3 hours to give the resulting MTTF on a Intel I5 8thGen.
Our study aim to adapt their algorithm to a CUDA capable gpu so to get results within minutes instead of hours.
We explored all possible CUDA implementation and documented major pros/cons of them.

On our [Documentation Slides](./Documentation/Caliper-Cuda-Presentation.pdf) you can find more details about it.

## What have we done?

As a project for the GPU & Heterogeneous system course, we created a CUDA version of Caliper.
The final version had a 72x speedup [Intel I5 8thGen w.r.t a Tesla K80 on AWS].

## Framework usage Instructions:

### TODO
