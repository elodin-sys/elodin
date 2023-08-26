
# SITL with PX4


Possible options that are available with PX4:

- Gazebo        Supported Vehicles: Quad, Standard VTOL, Plane
- FlightGear    Supported Vehicles: Plane, Autogyro, Rover
- JSBSim        Supported Vehicles: Plane, Quad, Hex
- jMAVSim       Supported Vehicles: Quad
- AirSim        Supported Vehicles: Iris (MultiRotor model and a configuration for PX4 QuadRotor in the X configuration).

From them, I first looked into how can you use Gazebo on MacOS (you can find the "Getting Started" guide further in this document).

After that, I looked into `AirSim`, because it seem like the best option when you want to have the most realistic render. One thing that won't work for us here is no existing option for fixed-wing simulation. Some people attempted it (like https://github.com/AOS55/Fixedwing-Airsim), but later those projects were abandoned. The main thing that I picked up is that you would want to use an external physics simulation if you want to add fixed-wing drones (you can find more notes about `AirSim` on `MacOS` further in this document).

Since we needed a simulation of fixed-wing drones I skipped over `jMAVSim`, and went to check out `JSBSim` and `FlightGear`.

`JSBSim` is not an option you would use if you want to have visuals, and it is primarily used with options that do provide them. Like for example it was used with `AirSim` in attempts to add fixed-wing simulation, or with `FlightGear` as an extra source of simulation models. So while this is useful information, it also indicates that `JSBSim` is not enough on its own if we want some visuals.

Quick Demo: https://youtu.be/fy41NWVqpLQ

[![Video](http://img.youtube.com/vi/fy41NWVqpLQ/0.jpg)](http://www.youtube.com/watch?v=fy41NWVqpLQ "PX4 + FlightGear")

## Possible workstation setups

### MacOS

Most of the things always "almost" work when you try to install them on MacOS. To describe my experiences I will mention situations I found myself in:

- `PX4` and target compilation at first don't cause any problems, but when you try to run certain simulations they will have errors, at the moment those errors can be fixed manually without much knowledge of internal workings since most of the errors are either related to "unused variables" or "function definition style". 

- `Unreal Engine` and possible plugins for it are usually a no-go without additional fixes. In a situation like this, you find yourself juggling between versions of UE, Xcode, and sometimes MacOS itself. For example, a lot of the things that worked with UE were tested on 4.27 which is not possible to compile with the latest Xcode, and often when you try to find a version that could compile it would require you to severely downgrade your OS. Often it's not just an older version, but something like 13.4.1 had a specific compilation bug fixed, but prev and next version has the same problem again.

- `i386` vs `arm64` modes, while some of the software will support `arm64` it's not possible to run the whole stack with it since the parts that need `i386` will require rosetta versions of brew to be installed which would create a lot of issues and conflicts. So for now I would not recommend trying this on MacOS unless you're ready to set up your terminal to primarily work with rosetta mode.


### Windows + WSL

Most of the related software can work on Windows directly, but `PX4` itself requires WSL. Overall I do not recommend using this setup as default for a couple of reasons:

- Even with the latest improvement to WSL (like the ability to run apps with GUI) it is not a 1:1 replication of Ubuntu, so for example `QGroundControl` which works perfectly fine on Ubuntu is not able to start in WSL without fixes.

- Since this setup relies on the communication between WSL and Windows over the "network" you'll often have a need to update IP addresses which can be changed with each WSL shutdown.

With 3+ applications in this setup and parts of the configuration changing periodically I don't think it would be productive to have this as a default setup for development purposes.

The main benefit of this or similar setup is that it works best when you plan to use `Unreal Engine` for rendering.

### Ubuntu

Overall I found this setup to be the easiest out of all of the others, but even then `PX4` targets required some fixes in the `PX4-Autopilot` project.

As far as I can tell based on the information available, this setup works the way it is because it's treated as the only way to do development work with PX4, and every other way is an afterthought that was probably not tested or fixed lately.

### Summary

In general, I would recommend either using `Ubuntu` or having more than one workstation and setting up communication between them in your local network, this way you can have `AirSim` with `Unreal Engine` running on `Windows` machine (as a renderer) while using `MacOS` or `Ubuntu` for development work.

---

## PX4 + Gazebo + MacOS - Getting Started

1. You need to use i386 mode. And by that, I mean either having intel Mac or a full setup terminal in rosetta mode, things like homebrew won't work correctly if they were installed in `arm64` mode.

2. Enable more open files by updating `.zshenv` file (required for gazebo installation, and I think px4 too but not sure)
    ```sh
    echo ulimit -S -n 2048 >> ~/.zshenv
    ```
    
3. Install Ruby again. Basically, every mac comes with ruby included, but it won't work here because of OSX System Integrity Protection (SIP), and while we can disable that (by going into recovery mode) - there's another way. Which is installing ruby again and using that:
    ```sh
    brew install ruby
    
    # Feel free to run this every time you need it, or update your ~/.zshrc
    export PATH=$(brew --prefix)/opt/ruby/bin:$PATH
    ```

4. Install px4-dev tools:
    ```sh
    brew tap PX4/px4
    brew install px4-dev  
    ```
    
5. Install required packages for px4:
    ```sh
    python3 -m pip install --user pyserial empty toml numpy pandas jinja2 pyyaml pyros-genmsg packaging kconfiglib future jsonschema  
    ```
    
6. Install gazebo:
    ```sh
    brew tap osrf/simulation
    brew install gz-garden
    
    # Confirm that it works by running it (you should just see green lines if everything is okay)
    gz sim -s -v
    ```

7. Run the macOS setup script. It uses my fork, and while it's not necessary for this script it'll be necessary for px4 compilation later.
    ```sh
    git clone https://github.com/ch-greams/PX4-Autopilot.git --recursive
    cd PX4-Autopilot/Tools/setup
    ./macos.sh
    ```
    
8. Start the simulator. For me this part shows errors on the first try, so just wait for gazebo to be visible and run the same line again.
    ```sh
    cd PX4-Autopilot
    PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL=x500 ./build/px4_sitl_default/bin/px4  
    ```
    
9. Try more things from https://docs.px4.io/main/en/sim_gazebo_gz/#examples

---

## PX4 + AirSim + MacOS - Notes

`PX4` + `AirSim` on MacOS is possible but it's really difficult without fixes for `AirSim`. Basically, you need to have very specific versions of `MacOS` + `Xcode` + `Unreal Engine` for it all to work together, in future work we should be able to add fixes to `AirSim` to work with the latest versions but atm it would be rather time-consuming.

I did make attempts to fix errors manually each time, but always hit the wall with something that couldn't be edited, it seems like the best bet would be replacing necessary plugins in the latest UE, but that requires more extensive knowledge of the `AirSim`.

To summarize the version conflicts that I encountered:
- `AirSim` needs `UE 4.27` 
    - Some necessary plugins (like PhysXVehicles) were deprecated with `UE 5.1` 
    - `UE 5.0` isn't compatible with the latest `Xcode` either
- `UE 4.27` needs `Xcode` 13.4.1
    - The latest `Xcode` is not able to compile libraries required for `UE` 4.27 (and it ignores compilation flags)
- Based on my investigation `Xcode` 13.4.1 is the only version that "could" work (found threads that before/after versions have other and similar issues), but it needs a MacOS downgrade which is quite overkill.

---
