FLAGS_G++ = -g -pedantic -std=c++17 -Wall -Wextra -Werror
PROGRAN_NAME = solution


lab1: run

compile: main.cpp
	g++ $(FLAGS_G++) main.cpp -o $(PROGRAN_NAME)

run: compile
	$(PROGRAN_NAME) < input

clean:
	del $(PROGRAN_NAME).exe