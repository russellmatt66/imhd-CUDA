# Overview
This is where the debugging happens - very long argument list to the main executable makes debugging easier if the Python launcher is removed.

Basically, you can throw the binary here into `compute-sanitizer`

# Directory Structure
`src/`
- `main.cu` in here 
- CMakeLists.txt in here

`build/`
- Where `main.cu` is built

`data/`
- Where the data goes to debug 

`infs/`
- Using RAPIDs to analyze data for infinities / anomalously large numbers

`nulls/`
- Using RAPIDs to analyze data for nulls / nans

# Current Tasks
(1) Debug all the nulls / nans + infinities / ALNs out of application
