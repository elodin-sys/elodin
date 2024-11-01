+++
title = "NumPy, JAX & NOX"
description = "JAX & NOX"
draft = false
weight = 107
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 7
+++

When setting about to develop an open and composable physics engine, deciding how to support numerical computation in an efficient and
expressive manner was a key exploration. An obvious inspiration we looked to was the [Python NumPy](https://numpy.org/) ecosystem.

If you're not familiar with NumPy, imagine you're working with a massive dataset or a matrix of numbers, and you need to perform calculations
efficiently. Normally, doing that with regular Python lists would be slow and memory-hungry. Enter NumPy: a tool
for high-performance number crunching in Python. NumPy solves this problem by offloading expensive math operations to a fast hyper-optimized C library.

NumPy gives you `ndarrays`, which are multi-dimensional arrays that are faster and more memory-efficient than Python's built-in lists. You can
also do vectorized operations on these arrays—so instead of writing loops to process each element, NumPy lets you apply operations to entire
arrays at once, like performing element-wise math in parallel. NumPy's core innovation is that it lets you perform vectorized operations on
multi-dimensional arrays. A vectorized operation means that the CPU operates on multiple entries in an array simultaneously. NumPy's syntax
is also less verbose – so instead of writing [2 * x for x in array] you can just write 2 * array and NumPy handles the rest for you.

It's also the backbone for other major scientific libraries like SciPy and pandas. NumPy is the most popular way of expressing math in
Python. It's commonly used in virtually every field, and has developed a large eco-system of libraries.

As a result, NumPy has become a staple for numerical computation across schools and industries.

## JAX

NumPy sounds great! So why not use it for Elodin?

Millions of simulations running in parallel is a common use case for Elodin. When you are trying to perform a large series of mathematical
expressions, find a way to optimize becomes important. That's where [JAX](https://jax.readthedocs.io/en/latest/) comes in.

JAX is like NumPy’s supercharged cousin. It gives you all the things you love about NumPy—fast, efficient array operations—but with some
game-changing extras that make it ideal for machine learning and other performance-heavy tasks.

#### Automatic Differentiation

First, JAX can automatically compute derivatives of functions, which is super useful for things like training neural networks. You can
define a function, and JAX will figure out how to differentiate it for you, no extra coding required. This is called automatic differentiation,
popular in machine learning because finding the gradient of a function is crucial for training neural networks.

#### Just-in-Time Compilation

Second, JAX uses something called just-in-time (JIT) compilation. Normally, when you run Python code, it’s interpreted line by line,
which can be slow. But JAX can take your code and compile it to run way faster, like low-level machine code. So you get the flexibility of
Python but with near-C++ performance.

#### GPU & TPU Acceleration

Third, it can easily run your computations on GPUs and TPUs by compiling to [XLA](https://openxla.org/xla), without you having to manage
all that complexity. You just write Python code, and JAX handles sending it to the GPU in the background.

JAX is like NumPy, but optimized for high-performance computing, like running massive monte-carlo simulations.


## Nox

Enter Nox, a Rust implementation of JAX. Nox is a Rust library that provides a NumPy-like interface for numerical computing, but with the
performance and power of JAX. It's a Rust port of JAX, which means you can write your code in Rust and get all the benefits of JAX, like
automatic differentiation and just-in-time compilation.

JAX sounded great! So why re-write JAX in Rust to create Nox? For us, it boils down to 3 reasons.

#### Performance

One is that the JAX tracing process can get a little slow for more complex programs. By bringing Jaxprs into Rust, we can optimize the
compile path more. This doesn't matter for small simple programs, but for larger programs (especially with lots of unrolling) it could matter a lot.

#### Production-Ready

The 2nd reason is that we prefer Rust for making production-ready software. The strong type-system especially allows you to catch many errors
at compile time, and while Python is moving in that direction, it isn't there yet.

#### Integration

The 3rd reason is that we want this software to easily integrate with your existing flight software (the control software for your drone/satellite),
and using a "systems" language like Rust makes that easier. Eventually, we would love to build out a suite of flight software in Rust that can easily
integrate with the simulation but is still flight-ready.

A fun side-effect of this architecture is Nox allows you to write code for both simulations and flight software generically, allowing for seamless
"software in the loop" and "hardware in the loop" testing of your flight software.

## And The "Sharp Bits"

While it's tempting dive in thinking the use of JAX/NOX in Elodin will be just like NumPy, the
[Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) still apply, namely:

#### Immutable Arrays
In JAX, once you create an array, you can’t change it directly. Think of it like a "look but don’t touch" rule. Instead of modifying an
array in place, JAX forces you to create new arrays with your changes. This can feel weird if you’re used to NumPy, where you can just tweak
things in-place. The immutability is important for making sure JAX’s optimizations work smoothly, but it takes a bit of getting used to.

#### Pure Functions, No Side Effects
JAX loves pure functions—meaning no messing around with global variables, no printing from inside the function, nothing that changes outside
the function's scope. It sounds strict, but it’s necessary for JAX’s magic, like automatic differentiation and just-in-time compilation, to
work properly. So, if you try to sneak in a print statement while debugging, JAX will throw a fit!

#### Subtle Differences from NumPy
JAX feels a lot like NumPy, but there are some small gotchas. For example, some NumPy functions you’re used to might be missing or work a
little differently. JAX is really built for numbers and arrays—if you try to work with anything else, like random Python objects, JAX won’t be happy.
