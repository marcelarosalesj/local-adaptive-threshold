CC = g++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

all:demo

demo:demo.cpp
	g++ -o demo demo.cpp -g `pkg-config opencv --cflags --libs`

clean:
	\rm demo
