.PHONY: all clean
COMP_NAME = clang++
COMP_ARGS = -std=c++17
FILE_NAME = main


all: app

clean:
	rm -rf  *.o

app: runge_kutt.cpp funs.cpp euler.cpp main.cpp
	$(COMP_NAME) $(COMP_ARGS)  runge_kutt.cpp funs.cpp euler.cpp main.cpp  -o $(FILE_NAME).o