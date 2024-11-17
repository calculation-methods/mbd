.PHONY: all clean
COMP_NAME = clang++
COMP_ARGS = -std=c++17
FILE_NAME = main


all: app

clean:
	rm -rf  *.o

app: finite_diff.cpp funcs.cpp  shooting.cpp main.cpp
	$(COMP_NAME) $(COMP_ARGS)  finite_diff.cpp funcs.cpp  shooting.cpp main.cpp  -o $(FILE_NAME).o