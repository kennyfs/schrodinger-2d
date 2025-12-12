# Makefile for Windows (use with nmake)

.PHONY: all

all: sequential.exe cuda.exe

sequential.exe: sequential.cpp window.cpp
	cl /EHsc /O2 /std:c++17 sequential.cpp window.cpp /I"C:\Dev\SDL2-2.30.8\include" /link /LIBPATH:"C:\Dev\SDL2-2.30.8\lib\x64" SDL2.lib /OUT:sequential.exe

cuda.exe: cuda.cu window.obj
	nvcc -O3 -std=c++17 cuda.cu window.obj -o cuda.exe -I"C:\Dev\SDL2-2.30.8\include" -L"C:\Dev\SDL2-2.30.8\lib\x64" -lSDL2 -allow-unsupported-compiler

window.obj: window.cpp window.h
	cl /EHsc /O2 /std:c++17 /c window.cpp /I"C:\Dev\SDL2-2.30.8\include"

# Clean up build artifacts
clean:
	del /Q sequential.exe cuda.exe *.obj
