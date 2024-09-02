## Week 7.29-8.4
1. DSE
    1.1 Make up utils & pruning prec data
    1.2 [DONE] Run a toy dse test
2. Pruning
   2.1 Use forward hooks
   2.2 Find new datasets
3. RTL Implementation 
   3.1 [DONE] Finish sparse_matmul & test
   3.2 Understand fixed_attention


## Week 8.5-8.11

1. Pruning
   1.1 [DONE] Use parametrization registration for standard weight pruning.
   1.2 [DONE] Deploy a formal downstream test (QNLI)?
   1.3 [DONE] Finish pruning sensitivity test based on the downstream model.
      1.3.1 [TODO] Support downstream model finetuning?
2. RTL
   2.1 Fix bugs from sparse_matmul synthesis
   2.2 Compare resource utils & timing of sparse & dense matmul for the same task
   2.3 Understand fixed_attention dataflows
3. DSE



## Week 8.26-9.2
0. 9.2任务：
   * 彻底解决dse**并push**
   * 将高于80%的cfg跑dse
   * 研究海报
   * （optional）看random的并行？
1. DSE
   1.1 Implement random search integrated with DSE
      * finish `pass_accuracy_check()`
          * Can it be done in parallel as well (i.e. add them to subprocesses)
      * catch the stdout of each subprocess and redirect them to logs under each report dir (i.e. make the screen silent)
      * Implement the standard search procedure:
          * Once got 64 trials of sparsity cfgs, dispatch them: each time 5 trials
      * 如何设置subprocess的返回值？对每一次dse，最终目标是返回给定资源下的最优吞吐量
   1.2 [DONE] Optimise rsc & throughput (II) evaluation in mase-dse.git -> models.py
      * N.B.: global `block_num` vs. local `block_num`
   1.3 [DONE] Make `col` parallelism tunable?
   1.4 **IMPORTANT**
      * make `max_x_row` and all related row num divisible by `block_num`?
      * [DONE] optimise DSP evaluation: lut += 300*dsp
      * vary `max_x_row` and see what happens
   1.3 Finish pruning sensitivity analysis
   1.4 (Optional) introduce quantisation to mase-pruning-test.git
2. RTL
   2.1 Finish all sparse_matmul_xcomp synthesis tasks
   2.2 Pass fixed_attention tb embedded with sparse_matmul
   2.3 Fix bugs in the emitted bert3-hdl-prj
      * [FIXED in the main branch] `fixed_softer_max_1d.sv`: POW2_WIDTH & POW2_FRAC_WIDTH
      * `fixed_self_attention_head.sv`: line 185 & 192, should be _0
      * `lut.sv`: line 17 & 37: $readmem got an empty file path



# Report Ideas
1. To compare DSE implementation strategies
   * Ordinary 3D DSE (design space too large)
   * 2D DSE (Thr + Rsc) constrained by Acc
   * TPE
   * Can make a list containing no. iterations for each strategy, implying why TPE is more efficient
2. To justify why block-wise sparsity 
   * Accuracy: prove that block-wise pruned BERT can have the same accuracy with element-wise pruned BERT
   * Resource/Latency: prove that a finer granularity pruning scheme causes a larger fanout / more logic resources / higher latency
   * Cache miss
3. To justify why DSE with sparsity is needed
   * BERT layers/modules have different sensitivity to sparsity
   * Show my type1/type2 pruning sensitivity analysis data.
4. To compare different search algorithms
   * TPE
   * Random Search
   * Grid Search (inefficient)
5. Implementation Details
   * Global (sparse_matmul) block_num vs. local (sparse_simple_matmul) block_num
6. Limitation/Future Works
   * Clock gating to save power
      * dynamic weight sparsity checking (in real scenarios dense blocks might be larger or smaller than the statically set value)
      * dynamic input activation zero checking (can be fine granularity)
   * Post-pruning finetuning
   * 
  
  