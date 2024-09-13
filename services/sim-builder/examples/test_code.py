import os
import sys

if sys.argv.count("--dir") != 0:
    dir = sys.argv[sys.argv.index("--dir") + 1]
    os.makedirs(dir, exist_ok=True)
    outfile = os.path.join(dir, "output.txt")
    with open(outfile, "w") as file:
        file.write("helloworld\n")


def test_hello():
    assert True
