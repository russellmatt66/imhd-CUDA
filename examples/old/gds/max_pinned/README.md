# Overview
Code doesn't seem to change the value, not sure why.

Solution approaches: 
- (a) Try and increase maximum pinned size
-- The code for this should work, but doesn't seem to.
- (b) Track memory usage with a tool to see what's going on in more depth


# RUN INSTRUCTIONS
`$ nvcc -o test max_pinned.cu -L/usr/local/cuda/lib64 -lcufile -lcudart -lcuda -arch=sm_75`
- `lcufile`: Links against cuFile
- `lcudart`: Links against CUDA runtime library
- `lcuda`: Links against the CUDA driver library