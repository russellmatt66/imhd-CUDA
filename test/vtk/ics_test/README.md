# Overview
ics.cu/
- Initializes simulation data, and writes it to `.dat` files


# Current Tasks
(1) Fix segfault in `writeGridGDS`
- Stems from trying to write data allocated on device from host
    - Just use a kernel to do so instead

