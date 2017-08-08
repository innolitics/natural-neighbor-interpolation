PROJ_NAME=nn

FLAGS= -Wall --std=c++11

all: nn.cpp
	g++ $(FLAGS) -o $(PROJ_NAME) $<
