# Launch the `make_movie` binary for rendering contents of data folder as a movie
import os
import sys
import subprocess

path_to_data = sys.argv[1]
dset_name = sys.argv[2]

arg_list = []
arg_list.append("./makemovie_driver")
arg_list.append(path_to_data)
arg_list.append(dset_name)
arg_list.append(path_to_data + "grid.h5")

# Read data folder to get a measure of how many frames are in there
def numberOfFrames(path_to_data: str) -> int:
    Nt = 0
    all_data_files = [file for file in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, file))]

    for file in all_data_files:
        if file == "README.md": 
            continue
        file_name = file.split('.h5')[0]
        if file_name == "grid":
            continue
        frame_num = file_name.split('_')[1]
        if int(frame_num) > Nt: 
            Nt = int(frame_num)
         
    return Nt

Nt = numberOfFrames(path_to_data)

arg_list.append(str(Nt))

print(f"Running command: {' '.join(arg for arg in arg_list)}")
subprocess.run(arg_list)