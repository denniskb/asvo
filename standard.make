# macros
CPPFILES := $(wildcard src/*.cpp)
CUFILES := $(wildcard src/*.cu)
OBJFILES := $(patsubst %.cpp,%.obj,$(CPPFILES))
OBJFILES := $(patsubst src/%,obj/%,$(OBJFILES))
HFILES := $(wildcard inc/*.h)

ADDLIBS := -l cUtil64 -l glew64 -l glut64

# release target
release : bin/asvo_cuda_release.exe

bin/asvo_cuda_release.exe : $(OBJFILES) $(CUFILES) $(HFILES)
	nvcc -arch compute_20 --maxrregcount 40 -D CUDA -link $(CUFILES) $(OBJFILES) -o bin/asvo_cuda_release.exe $(ADDLIBS)

# compile .cpp files to the correspondand .obj files
obj/%.obj : src/%.cpp $(HFILES)
	nvcc -c $< -o obj/

# delete all .obj and .exe files
cleanup :
	del bin\*.exe obj\*.obj