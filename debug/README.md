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
- This is DEPRECATED

`infs/`
- Using RAPIDs to analyze data for infinities / anomalously large numbers
- This is DEPRECATED

`nulls/`
- Using RAPIDs to analyze data for nulls / nans
- This is DEPRECATED

# Current Tasks
