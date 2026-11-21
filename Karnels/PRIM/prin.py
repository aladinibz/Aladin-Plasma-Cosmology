v01 is kernel 01 of the ALADIN native-cylindrical GPU MHD Driver for Z-pinch and J×B research. It converts the conserved state (ρ, m, E) and the constrained-transport face magnetic field into primitive variables (ρ, v, p). It is one CUDA/C++ source that runs on the GPU through NVRTC and on the CPU through g++/OpenMP, in FP64 and FP32.

What PRIM provides:
