cmake_minimum_required(VERSION 3.28)
project(elodin-db-grpc-examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(elodin-db-protos REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)

add_executable(elodin-db-grpc-client-batched grpc-client-batched.cpp)
target_link_libraries(elodin-db-grpc-client-batched
    PRIVATE
        elodin-db-protos::elodin-db-protos
        OpenSSL::Crypto)
