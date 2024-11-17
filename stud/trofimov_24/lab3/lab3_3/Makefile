.PHONY: all clean
COMP_NAME = clang++
COMP_ARGS = -std=c++17
FILE_NAME = main
 

all: app
 
clean:
	rm -rf  *.o

app: solve.cpp main.cpp
	$(COMP_NAME) $(COMP_ARGS)  main.cpp solve.cpp  -o $(FILE_NAME).o





