# CLI Profiling
`ncu -o profile --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sumd,duration -f ./imhd-cuda_profile`
- Obtains data from hardware counters that can be used to calculate FLOPs 
-- `flop_count_sp` = `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` + `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` + `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum * 2`
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#quickstart

# Current Tasks
(1) Refactor code so that each problem size has its own directory in `./data`, and data files are identified by execution configuration

(2) Refactor code to change information in data files
- Add how many iterations go into recording a single event