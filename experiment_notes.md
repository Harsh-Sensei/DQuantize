# Sampling times 
Time taken for pre-model operations: 6.985664367675781e-05 seconds
Time taken for model inference: 1.6756255626678467 seconds
Time taken for EOS logits setting: 2.384185791015625e-07 seconds
Time taken for add_gumbel_noise: 1.0728836059570312e-05 seconds
Time taken for torch.argmax: 4.696846008300781e-05 seconds
Time taken for confidence EOS/EOT setting: 2.384185791015625e-07 seconds
Time taken for F.softmax: 3.123283386230469e-05 seconds
Time taken for torch.gather: 0.00017905235290527344 seconds
Time taken for block_end inf setting: 0.00010943412780761719 seconds
Time taken for torch.where operations: 0.0003910064697265625 seconds
Time taken for token selection (topk loop): 2.679671049118042 seconds
Time taken for final update and device transfer: 0.00020456314086914062 seconds
Time taken for sampling (total): 2.6807169914245605 seconds


================================================================================

--------------------------------------------------------------------------------
PERFORMANCE BENCHMARKS
--------------------------------------------------------------------------------
     M      N      K   Triton(ms)    Naive(ms)     FP16(ms)    Speedup
--------------------------------------------------------------------------------
     1      1     16       0.0493       2.1080       0.0214      42.80x
     1    512    512       0.0700      67.3285       0.0275     961.77x
     4    512    512       0.0707      66.9687       0.0243     947.68x
     1   1024   1024       0.1154     134.0063       0.0268    1161.33x
     8   1024   1024       0.0442     133.5645       0.0258    3021.24x
     1   4096   4096       0.4303     529.6003       0.2002    1230.69x
    32   4096   4096       7.3431     527.4963       0.1749      71.84x
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
PERFORMANCE BENCHMARKS
--------------------------------------------------------------------------------
     M      N      K   Triton(ms)    Naive(ms)     FP16(ms)    Speedup
--------------------------------------------------------------------------------
     4    512    512       0.0707      66.9687       0.0243     947.68x
     8   1024   1024       0.0442     133.5645       0.0258    3021.24x
    32   4096   4096       7.3431     527.4963       0.1749      71.84x
--------------------------------------------------------------------------------