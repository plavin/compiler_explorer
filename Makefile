all: main.c
	gcc -o main main.c -fopenmp

.PHONY: clean 
clean: 
	rm -f main
