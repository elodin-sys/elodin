# Impeller 2 C Example
This simple C example creates a static VTable using the elodin-db cli, and then sends it using C

## Dependencies

- a C99 compiler
- elodin-db (../..)
- [just]()

## Usage 

To test you will need an elodin-db instance running at 127.0.0.1:2240. The easiest way to get one is to run:

```sh
just db
```

Then you can build and run the example using

```sh
just client
```
