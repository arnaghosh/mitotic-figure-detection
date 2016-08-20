all: bcd

bcd: BCD.cpp
	g++ -g $^ -std=c++0x -o $@ -Wall `pkg-config opencv --cflags --libs`

em: EM.cpp
	g++ -g $^ -o $@ -Wall `pkg-config opencv --cflags --libs`

	
