# Impeller 2 C Example
This simple C example creates a static VTable using the impeller2 cli, and then sends it using C

## Dependencies

- a C99 compiler
- impeller2-cli (../../cli)
- [just]()

## Usage 

To test you will need an impeller-db instance running at 127.0.0.1:2240. The easiest way to get one is to run:

```sh
cargo run --release --package impeller-db -- 127.0.0.1:2240 ./test
```

Then you can build and run the example using

```sh
just run
```
