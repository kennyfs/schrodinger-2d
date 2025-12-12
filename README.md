# PP Final Project

This project is a **CUDA‑parallelized implementation** of the 2D Schrödinger Simulator, adapted from [Verstand's original code](https://github.com/VerstandTsai/schrodinger-2d).
Because I do not have access to a Linux environment, and using SDL2 in WSL2 provides a poor experience, so I make it support both **Windows** and **Linux** platforms.

---

## Linux

Navigate to the `linux` directory and run:

```bash
make cuda       # build the CUDA version
make sequential # build the sequential CPU version
make # build both CUDA and CPU version
```

## Windows
Requirment:
- Microsoft cl compiler
- nvcc
- SDL2

```Powershell
nmake cuda       # build the CUDA version
nmake sequential # build the sequential CPU version
nmake # build all
```

> Note: Windows paths may vary depending on your setup. Make sure to update the Makefile with the correct include/library paths for SDL2 and CUDA on your machine.


