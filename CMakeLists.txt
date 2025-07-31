cmake_minimum_required(VERSION 3.10)
project(NN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(SOURCES
    main.cpp
    model.cpp
    lay.cpp
    dense.cpp
)


set(HEADERS
    dense.h
    lay.h
    model.h
)


add_executable(NN ${SOURCES} ${HEADERS})