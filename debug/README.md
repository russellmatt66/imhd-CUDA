# Overview
This is where the debugging happens - very long argument list to main executable makes debugging easier if Python launcher is removed

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
