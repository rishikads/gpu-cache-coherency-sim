CXX = g++
CXXFLAGS = -std=c++17 -O3 -Iinclude

SRC = src/main.cpp src/sim.cpp src/cache.cpp src/memsys.cpp src/workload.cpp
OBJ = $(SRC:.cpp=.o)

all: sim

sim: $(OBJ)
	$(CXX) $(CXXFLAGS) -o sim $(OBJ)

clean:
	rm -f $(OBJ) sim
