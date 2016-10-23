CC = g++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

all:at

at:adaptivethresholding.cpp
	g++ -o at adaptivethresholding.cpp `pkg-config opencv --cflags --libs`

clean:
	\rm at
