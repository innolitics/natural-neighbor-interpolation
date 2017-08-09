# FLAGS= -Wall --std=c++11

all:
	python setup.py build

demo:
	cp ./build/lib.macosx-10.12-x86_64-3.6/naturalneighbor.cpython-36m-darwin.so ./
	python test.py

clean:
	rm -f nn
	rm -rf build/*
