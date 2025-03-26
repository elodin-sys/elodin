# Postcard-C Example

This example demonstrates how to use the Postcard-C library to serialize and deserialize C++ structures.

## Generating Bindings
```bash
cargo run foo.ron > foo.hpp # generate without formatting
cargo run foo.ron | clang-format > foo.hpp # generate without formatting
```

## Building

```bash
clang++ -std=c++23 main.cpp -o postcard_example
```

## Running

```bash
./postcard_example
```

The program will:
1. Create a Foo struct with various fields
2. Encode it using Postcard
3. Print the encoded binary data
4. Decode it back into a new instance
5. Print the decoded values

