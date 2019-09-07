CC = g++
FLAGS = -g -Wall -Werror

all: main

main: main.o NN.o
	${CC} ${FLAGS} -o $@ $^


.cpp.o: NN.hpp
	${CC} ${FLAGS}  -c $<

clean:
	rm *.o main