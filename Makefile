PROJ_NAME=nn

FLAGS= -Wall --std=c++11

all: nn.cpp
	g++ $(FLAGS) -o $(PROJ_NAME) $<

module:
	python setup.py build

test:
	cp ./build/lib.macosx-10.12-x86_64-3.6/naturalneighbor.cpython-36m-darwin.so ./
	python test.py

clean:
	rm -f nn
	rm -rf build/*
