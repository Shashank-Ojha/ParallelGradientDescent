APP_NAME=main

OBJS=regression.o sgd_designs.o main.o

default: $(APP_NAME)

# Compile for Xeon Phi
$(APP_NAME): CXX = icc -m64 -std=c++11
$(APP_NAME): CXXFLAGS = -I. -O3 -Wall -openmp -offload-attribute-target=mic -DRUN_MIC

# Compile for CPU
cpu: CXX = g++ -m64 -std=c++11
cpu: CXXFLAGS = -I. -O3 -Wall -fopenmp -Wno-unknown-pragmas

# Compilation Rules
$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

cpu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(APP_NAME) $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

submit:
	cd templates && ../scripts/batch_generate.sh
	cd job_outputs && ../scripts/sub/submit.sh
clean:
	/bin/rm -rf *~ *.o $(APP_NAME) templates/$(USER)_*.job job_outputs/$(USER)_* file_outputs/*
cleanf:
	/bin/rm -rf job_outputs/$(USER)_*

# For a given rule:
# $< = first prerequisite
# $@ = target
# $^ = all prerequisite
