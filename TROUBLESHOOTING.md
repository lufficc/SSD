# Troubleshooting

## RuntimeError: merge_sort: failed to synchronize: an illegal memory access was encountered

This is caused in multi-box loss. The sort method failed due to NaN numbers. This may be a bug in `log_softmax`: https://github.com/pytorch/pytorch/issues/14335 .Three ways to solve :
1. Use a smaller warmup factor, like 0.1. (append `SOLVER.WARMUP_FACTOR 0.1` to your train cmd's end).
1. Use a longer warmup iters, like 1000. (append `SOLVER.WARMUP_ITERS 1000` to your train cmd's end).
1. [Described in the forums by Jinserk Baik](https://discuss.pytorch.org/t/ctcloss-performance-of-pytorch-1-0-0/27524/29)