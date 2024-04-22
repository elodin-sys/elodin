# Elodin Unreal Demo

This is a simple show case of the Elodin <-> Unreal integration. At the moment the plugin only 
allows you to spawn a camera into the Unreal environment.

## Setup
1. Install Unreal Editor 4.2
2. Install nox-py either using maturin develop or using pip
3. Install ue4 cli using `pip install git+https://github.com/adamrehn/ue4cli.git`
4. Open the project with `just run`
5. From your virtual-env start the drone example with `python3 ../../libs/nox-py/examples/drone.py`
6. Profit!
