/* 
~~~~~~~
SUMMARY
~~~~~~~
In `imhd-CUDA/test/vtk/ics_test/ics.cu`, the function `writeGridGDS` uses GPU Direct Storage to write the contents of a buffer, which contains the coordinates
of the computational grid, directly to storage. 

These values are written as raw bytes, so they must be read back as floats before being processed through a visualization pipeline, and into a rendering engine. 

This occurs in `imhd-CUDA/test/vtk/ics_test/visualization/grip.cpp`, however, when the values that are being read from the data file are streamed to console,
the result is that every single one of them is equal to 0. This indicates a bug somewhere in the process of creating, and writing the grid. 

The goal of this file is to debug where. 
*/

int main(int argc, char* argv[]){
    /* WIP */
    return 0;
}