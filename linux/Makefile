CXXFLAGS = -Wall -O3 `pkg-config --cflags opencv4`
NVCC = nvcc
LDLIBS = -lSDL2 `pkg-config --libs opencv4`
CUDAFLAG = -O3 `pkg-config --cflags opencv4`

ifdef HEADLESS
CXXFLAGS += -DHEADLESS
CUDAFLAG += -DHEADLESS
LDLIBS =
OBJ_FILES =
endif

all: sequential cuda

sequential: sequential.cpp window.o
	$(CXX) $(CXXFLAGS) -o $@ sequential.cpp window.o $(LDLIBS)

cuda: cuda.cu window.o
	$(NVCC) $(CUDAFLAG) -o $@ cuda.cu window.o $(LDLIBS)


window.o: window.cpp window.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f cuda sequential *.o

