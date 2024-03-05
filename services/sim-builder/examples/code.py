import os, sys

dir = sys.argv[sys.argv.index("--dir") + 1]
os.makedirs(dir, exist_ok=True)

outfile = os.path.join(dir, "output.txt")
with open(outfile, "w") as file:
    file.write("helloworld\n")
