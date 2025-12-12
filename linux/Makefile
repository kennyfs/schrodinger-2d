CXXFLAGS = -Wall -O3
NVCC = nvcc
LDLIBS = -lSDL2
CUDAFLAG = -O3

.PHONY: all clean

all: sequential cuda

sequential: sequential.cpp window.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

cuda: cuda.cu window.o
	$(NVCC) $(CUDAFLAG) -o $@ $^ $(LDLIBS)

window.o: window.cpp window.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f cuda sequential *.o

