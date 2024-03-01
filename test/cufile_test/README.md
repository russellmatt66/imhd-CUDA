# RUN INSTRUCTIONS
`$ nvcc -o test cufile_test.cu -L/usr/local/cuda/lib64 -lcufile -lcudart -lcuda -arch=sm_75`
- `lcufile`: Links against cuFile
- `lcudart`: Links against CUDA runtime library
- `lcuda`: Links against the CUDA driver library