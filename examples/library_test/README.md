# Overview
Current tasks:
(1) Add builds for C, and C++ executables that launch CUDA kernel

# Current Build Error
Problem with multiple definitions (there aren't), when trying to compile project where kernel code is launched from inside C++. 
```
$ make
[ 14%] Building CUDA object CMakeFiles/hellolauncher_lib.dir/library/helloLauncher.cu.o
[ 28%] Linking CUDA static library libhellolauncher_lib.a
[ 28%] Built target hellolauncher_lib
[ 42%] Building CUDA object CMakeFiles/hello_launcher_cu.dir/main.cu.o
[ 57%] Linking CUDA executable hello_launcher_cu
[ 57%] Built target hello_launcher_cu
[ 71%] Building CUDA object CMakeFiles/hello_launcher_cpp.dir/main.cpp.o
/home/matt/Desktop/imhd-CUDA/test/library_test/main.cpp(31): warning #550-D: variable "hello_status" was set but never used
      int hello_status = 0;
          ^

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

[ 85%] Building CUDA object CMakeFiles/hello_launcher_cpp.dir/library/callLauncher.cu.o
[100%] Linking CUDA executable hello_launcher_cpp
/usr/bin/ld: CMakeFiles/hello_launcher_cpp.dir/library/callLauncher.cu.o: in function `helloWrapper(long*, long)':
tmpxft_000023a1_00000000-6_callLauncher.cudafe1.cpp:(.text+0x1a): multiple definition of `helloWrapper(long*, long)'; CMakeFiles/hello_launcher_cpp.dir/main.cpp.o:tmpxft_00002389_00000000-6_main.cudafe1.cpp:(.text+0x1a): first defined here
/usr/bin/ld: CMakeFiles/hello_launcher_cpp.dir/library/callLauncher.cu.o: in function `HelloLauncher(unsigned long)':
tmpxft_000023a1_00000000-6_callLauncher.cudafe1.cpp:(.text+0x2fb): multiple definition of `HelloLauncher(unsigned long)'; CMakeFiles/hello_launcher_cpp.dir/main.cpp.o:tmpxft_00002389_00000000-6_main.cudafe1.cpp:(.text+0x3fa): first defined here
/usr/bin/ld: CMakeFiles/hello_launcher_cpp.dir/library/callLauncher.cu.o: in function `__device_stub__Z13HelloLauncherm(unsigned long)':
tmpxft_000023a1_00000000-6_callLauncher.cudafe1.cpp:(.text+0x1d5): multiple definition of `__device_stub__Z13HelloLauncherm(unsigned long)'; CMakeFiles/hello_launcher_cpp.dir/main.cpp.o:tmpxft_00002389_00000000-6_main.cudafe1.cpp:(.text+0x2d4): first defined here
collect2: error: ld returned 1 exit status
make[2]: *** [CMakeFiles/hello_launcher_cpp.dir/build.make:114: hello_launcher_cpp] Error 1
make[1]: *** [CMakeFiles/Makefile2:139: CMakeFiles/hello_launcher_cpp.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```

# Linker Error Reference
BELOW IS SOLVED. 

The fix was saving the `main.cu` in Vim xO

Keeping below for reference

Cannot get `cmake` to work, there's a linker issue that I keep getting: 

```
$ make
[ 25%] Building CUDA object CMakeFiles/hellolauncher_lib.dir/library/hellolauncher.cu.o
[ 50%] Linking CUDA static library libhellolauncher_lib.a
[ 50%] Built target hellolauncher_lib
[ 75%] Building CUDA object CMakeFiles/hello_launcher_cu.dir/main.cu.o
[100%] Linking CUDA executable hello_launcher_cu
/usr/bin/ld: /usr/lib/gcc/x86_64-linux-gnu/11/../../../x86_64-linux-gnu/Scrt1.o: in function `_start':
(.text+0x1b): undefined reference to `main'
collect2: error: ld returned 1 exit status
make[2]: *** [CMakeFiles/hello_launcher_cu.dir/build.make:98: hello_launcher_cu] Error 1
make[1]: *** [CMakeFiles/Makefile2:111: CMakeFiles/hello_launcher_cu.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```

Trying to compile manually doesn't work either:
```
$ nvcc -c -o main.o main.cu -arch=sm_75
$ nvcc -c -o hellolauncher.o library/hellolauncher.cu -arch=sm_75
$ nvcc -o hello_world6D main.o hellolauncher.o
/usr/bin/ld: /usr/lib/gcc/x86_64-linux-gnu/11/../../../x86_64-linux-gnu/Scrt1.o: in function `_start':
(.text+0x1b): undefined reference to `main'
collect2: error: ld returned 1 exit status
```

Only sensible interpretation I can come up with is that it thinks `library/hellolauncher.cu` is the main file for some reason

Here is the symbol table for `main.o`:
```
$ objdump -t main.o

main.o:     file format elf64-x86-64

SYMBOL TABLE:
0000000000000000 l    df *ABS*    0000000000000000 tmpxft_000052c1_00000000-6_main.cudafe1.cpp
0000000000000000 l    d  .text    0000000000000000 .text
0000000000000000 l    d  .bss    0000000000000000 .bss
0000000000000000 l     O .bss    0000000000000001 _ZL22__nv_inited_managed_rt
0000000000000008 l     O .bss    0000000000000008 _ZL32__nv_fatbinhandle_for_managed_rt
0000000000000000 l     F .text    000000000000001a _ZL37__nv_save_fatbinhandle_for_managed_rtPPv
0000000000000010 l     O .bss    0000000000000008 _ZZL22____nv_dummy_param_refPvE5__ref
000000000000001a l     F .text    0000000000000019 _ZL22____nv_dummy_param_refPv
0000000000000000 l     O __nv_module_id    000000000000000f _ZL15__module_id_str
0000000000000018 l     O .bss    0000000000000008 _ZL20__cudaFatCubinHandle
0000000000000033 l     F .text    0000000000000029 _ZL26__cudaUnregisterBinaryUtilv
000000000000005c l     F .text    000000000000001e _ZL32__nv_init_managed_rt_with_modulePPv
0000000000000000 l    d  .nv_fatbin    0000000000000000 .nv_fatbin
0000000000000000 l       .nv_fatbin    0000000000000000 fatbinData
0000000000000000 l    d  .nvFatBinSegment    0000000000000000 .nvFatBinSegment
0000000000000000 l     O .nvFatBinSegment    0000000000000018 _ZL15__fatDeviceText
0000000000000020 l     O .bss    0000000000000008 _ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref
000000000000007a l     F .text    000000000000002a _ZL31__nv_cudaEntityRegisterCallbackPPv
00000000000000a4 l     F .text    000000000000005e _ZL24__sti____cudaRegisterAllv
0000000000000000         *UND*    0000000000000000 __cudaUnregisterFatBinary
0000000000000000         *UND*    0000000000000000 __cudaInitModule
0000000000000000         *UND*    0000000000000000 __cudaRegisterFatBinary
0000000000000000         *UND*    0000000000000000 __cudaRegisterFatBinaryEnd
0000000000000000         *UND*    0000000000000000 atexit
```

