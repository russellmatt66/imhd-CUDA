# Overview
This file discusses how to contribute to this project.

# On-Device
This part of the document discusses how to contribute to the `on-device` architecture. This is the codebase which solves the governing equations of Ideal MHD based on a data volume which can fit on a single GPU. 

Headers go in `include/on-device`, library files go in `lib/on-device`, and then the main code for the simulation runtime goes in `src/on-device`. Anything else goes in `on-device/utils`. These are the main locations for development. Yes, it leads to very long files, but that is the price which ended up being paid for separating the concerns. Any higher-order schemes than the 2nd-order Lax Wendroff should go in their own separate file. 

There are two options for creating a new problem to run, meaning a new set of initial conditions and/or a new ensemble of kernels. You can either create a new `.cu` file in `src/on-device/` with the kernels and initialization logic hard-coded, or you can add the initial conditions and/or new kernels to `include/on-device/configurers.hpp`. This file contains the definitions of the configurer classes which are the structures used to hash the keystrings in the simulation input file and select the desired set of kernels to run.    