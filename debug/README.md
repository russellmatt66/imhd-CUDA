# Overview
This is where the debugging happens

# Directory Structure
`src/`
- `main.cu` in here 
-- Q: Is a seperate `main.cu` really necessary?
- CMakeLists.txt in here

`build/`
- Where `main.cu` is built

`data/`
- Where the data goes to debug 

# Current Tasks
(1) Debug why nulls show up in the initial conditions downstream of their specification.
- Check data writing-out process 
