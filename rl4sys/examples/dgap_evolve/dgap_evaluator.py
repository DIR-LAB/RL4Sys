import os
import time
import numpy as np
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

from rl4sys.examples.dgap_evolve.dgap_evol import VCSR


class DGAPEvaluator:
  def __init__(self, base_file: str, dynamic_file: str, nv: int, ne: int):
    self.base_file = base_file
    self.dynamic_file = dynamic_file
    self.nv = nv
    self.ne = ne
    ts = int(time.time())
    self.writer = SummaryWriter(log_dir=os.path.join('./logs/rl4sys-evaluator', f"{ts}"))

  def _run_once(self, use_rl: bool, rl_mode: str = 'continuous') -> Dict[str, Any]:
    env = VCSR(self.nv, self.ne, use_rl=use_rl, rl_mode=rl_mode)
    t0 = time.time()
    env.load_basegraph(self.base_file)
    env.print_pma_meta()
    start_writes = env.num_write_insert + env.num_write_rebal + env.num_write_resize
    t1 = time.time()
    env.load_dynamicgraph(self.dynamic_file)
    t2 = time.time()
    total_time = t2 - t0
    dyn_time = t2 - t1
    total_writes = env.num_write_insert + env.num_write_rebal + env.num_write_resize
    total_reads = env.num_read_insert + env.num_read_rebal + env.num_read_resize
    rebalance_count = int(np.sum(np.array(env.rebalance_counter)))
    result = {
      'total_time': total_time,
      'dynamic_time': dyn_time,
      'total_writes': int(total_writes),
      'total_reads': int(total_reads),
      'rebalance_count': int(rebalance_count),
      'final_edges': int(env.num_edges),
      'elem_capacity': int(env.elem_capacity),
    }
    return result

  def compare(self, rl_mode: str = 'continuous', tag_prefix: str = 'eval', repeats: int = 1) -> Dict[str, Any]:
    agg = {
      'baseline': {'total_time': [], 'dynamic_time': [], 'total_writes': [], 'total_reads': [], 'rebalance_count': []},
      'rl': {'total_time': [], 'dynamic_time': [], 'total_writes': [], 'total_reads': [], 'rebalance_count': []}
    }
    for r in range(repeats):
      base_res = self._run_once(use_rl=False)
      rl_res = self._run_once(use_rl=True, rl_mode=rl_mode)
      for k in agg['baseline'].keys():
        agg['baseline'][k].append(base_res[k])
        agg['rl'][k].append(rl_res[k])
      # Write per-repeat scalars
      self.writer.add_scalar(f'{tag_prefix}/baseline_total_time', base_res['total_time'], r)
      self.writer.add_scalar(f'{tag_prefix}/rl_total_time', rl_res['total_time'], r)
      self.writer.add_scalar(f'{tag_prefix}/baseline_total_writes', base_res['total_writes'], r)
      self.writer.add_scalar(f'{tag_prefix}/rl_total_writes', rl_res['total_writes'], r)
      self.writer.add_scalar(f'{tag_prefix}/baseline_total_reads', base_res['total_reads'], r)
      self.writer.add_scalar(f'{tag_prefix}/rl_total_reads', rl_res['total_reads'], r)
      self.writer.add_scalar(f'{tag_prefix}/baseline_rebalance_count', base_res['rebalance_count'], r)
      self.writer.add_scalar(f'{tag_prefix}/rl_rebalance_count', rl_res['rebalance_count'], r)

    summary = {
      'baseline': {k: float(np.mean(v)) for k, v in agg['baseline'].items()},
      'rl': {k: float(np.mean(v)) for k, v in agg['rl'].items()}
    }
    # Log summary
    self.writer.add_text(f'{tag_prefix}/summary', str(summary))
    return summary


