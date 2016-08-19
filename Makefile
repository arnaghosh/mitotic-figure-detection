all: bcd

bcd: BCD.cpp
	g++ -g $^ -o $@ -Wall `pkg-config opencv --cflags --libs`


	
