import os
import time
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

class Vertex:
  def __init__(self, index, degree):
    self.index = np.int64(index)
    self.degree = np.int32(degree)

class PPOBuffer:
  def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95, continuous: bool = True, device: str = "cpu"):
    self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
    self.act_buf = np.zeros((size, act_dim), dtype=np.float32 if continuous else np.int64)
    self.adv_buf = np.zeros(size, dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.ret_buf = np.zeros(size, dtype=np.float32)
    self.logp_buf = np.zeros(size, dtype=np.float32)
    self.val_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.gamma, self.lam = gamma, lam
    self.ptr, self.path_start_idx, self.max_size = 0, 0, size
    self.device = device
    self.continuous = continuous

  def store(self, obs, act, rew, val, logp, done):
    assert self.ptr < self.max_size, "Buffer overflow"
    self.obs_buf[self.ptr] = obs
    if self.continuous:
      self.act_buf[self.ptr] = act
    else:
      self.act_buf[self.ptr] = int(act)
    self.rew_buf[self.ptr] = rew
    self.val_buf[self.ptr] = val
    self.logp_buf[self.ptr] = logp
    self.done_buf[self.ptr] = float(done)
    self.ptr += 1

  def finish_path(self, last_val: float = 0.0):
    path_slice = slice(self.path_start_idx, self.ptr)
    rews = np.append(self.rew_buf[path_slice], last_val)
    vals = np.append(self.val_buf[path_slice], last_val)
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    adv = np.zeros_like(deltas)
    gae = 0
    for t in reversed(range(len(deltas))):
      gae = deltas[t] + self.gamma * self.lam * gae * (1.0 - self.done_buf[self.path_start_idx + t])
      adv[t] = gae
    self.adv_buf[path_slice] = adv
    self.ret_buf[path_slice] = adv + self.val_buf[path_slice]
    self.path_start_idx = self.ptr

  def get(self):
    assert self.ptr == self.max_size, "Buffer must be full before get()"
    self.ptr, self.path_start_idx = 0, 0
    adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf) + 1e-8
    self.adv_buf = (self.adv_buf - adv_mean) / adv_std
    return (
      torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device),
      torch.as_tensor(self.act_buf, dtype=torch.float32 if self.continuous else torch.long, device=self.device),
      torch.as_tensor(self.adv_buf, dtype=torch.float32, device=self.device),
      torch.as_tensor(self.ret_buf, dtype=torch.float32, device=self.device),
      torch.as_tensor(self.logp_buf, dtype=torch.float32, device=self.device)
    )


class ActorCritic(nn.Module):
  def __init__(self, obs_dim: int, act_dim: int, continuous: bool = True):
    super().__init__()
    self.continuous = continuous
    hidden = 128
    # Policy
    self.pi_body = nn.Sequential(
      nn.Linear(obs_dim, hidden), nn.ReLU(),
      nn.Linear(hidden, hidden), nn.ReLU()
    )
    if continuous:
      self.mu = nn.Linear(hidden, act_dim)
      self.log_std = nn.Parameter(torch.zeros(act_dim))
    else:
      self.logits = nn.Linear(hidden, act_dim)
    # Value
    self.v = nn.Sequential(
      nn.Linear(obs_dim, hidden), nn.ReLU(),
      nn.Linear(hidden, hidden), nn.ReLU(),
      nn.Linear(hidden, 1)
    )

  def step(self, obs: torch.Tensor):
    with torch.no_grad():
      pi_h = self.pi_body(obs)
      if self.continuous:
        mu = self.mu(pi_h)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)
        act = torch.tanh(dist.sample())  # in (-1,1)
        logp = dist.log_prob(act).sum(dim=-1)
        act_clipped = act
      else:
        logits = self.logits(pi_h)
        dist = Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act)
        act_clipped = act
      val = self.v(obs).squeeze(-1)
    return act_clipped, logp, val

  def act_logp(self, obs: torch.Tensor, act: torch.Tensor):
    pi_h = self.pi_body(obs)
    if self.continuous:
      mu = self.mu(pi_h)
      std = torch.exp(self.log_std).expand_as(mu)
      dist = Normal(mu, std)
      logp = dist.log_prob(act).sum(dim=-1)
    else:
      logits = self.logits(pi_h)
      dist = Categorical(logits=logits)
      logp = dist.log_prob(act)
    return logp

  def value(self, obs: torch.Tensor):
    return self.v(obs).squeeze(-1)


class PPOAgent:
  def __init__(self, obs_dim: int, act_dim: int, continuous: bool = True, pi_lr: float = 3e-4, vf_lr: float = 1e-3, steps_per_update: int = 4096, train_iters: int = 10, clip_ratio: float = 0.2, target_kl: float = 0.02, device: str = None):
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.continuous = continuous
    self.ac = ActorCritic(obs_dim, act_dim, continuous=continuous).to(self.device)
    self.pi_opt = torch.optim.Adam(self.ac.parameters(), lr=pi_lr)
    self.vf_opt = torch.optim.Adam(self.ac.v.parameters(), lr=vf_lr)
    self.clip_ratio = clip_ratio
    self.target_kl = target_kl
    self.steps_per_update = steps_per_update
    self.train_iters = train_iters
    self.buf = PPOBuffer(obs_dim, act_dim, steps_per_update, continuous=continuous, device=self.device)

  def policy_step(self, obs_np: np.ndarray):
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
    act_t, logp_t, v_t = self.ac.step(obs_t)
    act = act_t.squeeze(0).detach().cpu().numpy() if self.continuous else int(act_t.squeeze(0).detach().cpu().item())
    logp = float(logp_t.squeeze(0).detach().cpu().item())
    val = float(v_t.squeeze(0).detach().cpu().item())
    return act, logp, val

  def store(self, obs, act, rew, val, logp, done):
    self.buf.store(obs, act, rew, val, logp, done)

  def finish_trajectory(self, last_val: float = 0.0):
    self.buf.finish_path(last_val=last_val)

  def maybe_update(self):
    if self.buf.ptr < self.buf.max_size:
      return False
    obs, act, adv, ret, logp_old = self.buf.get()
    for _ in range(self.train_iters):
      # Policy loss
      if self.continuous:
        logp = self.ac.act_logp(obs, act)
      else:
        logp = self.ac.act_logp(obs, act)
      ratio = torch.exp(logp - logp_old)
      clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
      loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
      approx_kl = (logp_old - logp).mean().item()
      self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()
      if approx_kl > 1.5 * self.target_kl:
        break
    # Value loss
    v = self.ac.value(obs)
    loss_v = ((v - ret) ** 2).mean()
    self.vf_opt.zero_grad(); loss_v.backward(); self.vf_opt.step()
    return True


class VCSR:
  # Height-based (as opposed to depth-based) tree thresholds
  # Upper density thresholds
  up_h = np.float64(0.75) # root
  up_0 = np.float64(1.00) # leaves
  # Lower density thresholds
  low_h = np.float64(0.50) # root
  low_0 = np.float64(0.25) # leaves

  max_sparseness = np.int8(1.0 / low_0)
  largest_empty_segment = np.int8(1.0 * max_sparseness)

  def __init__(self, n_vertices, n_edges, use_rl: bool = False, rl_mode: str = 'continuous'):
    self.num_vertices = n_vertices
    self.num_edges = n_edges

    # pma tree related metadata
    self.compute_capacity()

    self.segment_edges_actual = [np.int64(0)] * (self.segment_count * 2)
    self.segment_edges_total = [np.int64(0)] * (self.segment_count * 2)

    self.tree_height = self.floor_log2(self.segment_count)
    self.delta_up = (self.up_0 - self.up_h) / self.tree_height
    self.delta_low = (self.low_h - self.low_0) / self.tree_height

    # vertex array
    self.vertices_ = []
    for i in range(self.num_vertices):
      vt = Vertex(0, 0)
      self.vertices_.append(vt)

    # read/write metadata counter
    self.num_write_insert = 0
    self.num_write_rebal = 0
    self.num_write_resize = 0
    self.num_read_insert = 0
    self.num_read_rebal = 0
    self.num_read_resize = 0
    self.num_resize = 0
    self.rebalance_counter = [np.int64(0)] * (self.segment_count * 2)
    self.rebalance_reads = [np.int64(0)] * (self.segment_count * 2)
    self.rebalance_writes = [np.int64(0)] * (self.segment_count * 2)

    # TensorBoard logging to mirror RL version
    self._writer = SummaryWriter(log_dir=os.path.join('./logs/rl4sys-noml-info', f"{int(time.time())}"))
    self._global_step = 0
    self._last_total_writes = 0

    # RL integration (for OpenEvolve experiments)
    self.use_rl = bool(use_rl)
    self.rl_mode = rl_mode  # 'continuous' or 'discrete'
    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    # Observation: [left_density, right_density, left_deg_ratio, right_deg_ratio, parent_gaps_norm, level_norm]
    self._obs_dim = 6
    if self.use_rl:
      if self.rl_mode == 'continuous':
        self._act_dim = 1
        self._continuous = True
      else:
        # Discrete bins for split ratio in [0.1 .. 0.9]
        self._act_dim = 9
        self._continuous = False
      self.agent = PPOAgent(self._obs_dim, self._act_dim, continuous=self._continuous, steps_per_update=2048, device=self._device)
      self._rl_decision_queue = []  # items: dict(level, before_writes)


  def update_rebalance_metadata(self, height, cupdate, rupdate, wupdate):
    self.rebalance_counter[height] += np.int64(cupdate)
    self.rebalance_reads[height] += np.int64(rupdate)
    self.rebalance_writes[height] += np.int64(wupdate)


  def print_pma_meta(self):
    print("num_vertices: {}, num_edges: {}, avg_degree: {}, elem_capacity: {}".format(self.num_vertices, self.num_edges, self.avg_degree, self.elem_capacity))
    print("segment_size: {}, segment_count: {}, tree_height: {}".format(self.segment_size, self.segment_count, self.tree_height))
    print("total-edges: {}, actual-edges: {}".format(self.segment_edges_total[1], self.segment_edges_actual[1]))


  def print_pma_counter(self):
    print("PMA Statistics Start   ----------------")

    print("\n# write while insertion: {}".format(self.num_write_insert))
    print("# write while rebalance (degree): {}".format(self.num_write_rebal))
    print("# write while resize: {}".format(self.num_write_resize))
    print("# total-writes: {}".format(self.num_write_insert + self.num_write_rebal + self.num_write_resize))

    print("\n# read while insertion: {}".format(self.num_read_insert))
    print("# read while rebalance (degree): {}".format(self.num_read_rebal))
    print("# read while resize: {}".format(self.num_read_resize))

    print("\n# of resize: {}".format(self.num_resize))

    print("\nrebalance (weighted) counter by PMA height: ")
    for i in range(self.segment_count * 2):
      if self.rebalance_counter[i] != np.int64(0):
        print("at height {} number of rebalanced occurred: {}".format(i, self.rebalance_counter[i]))
    # for key, value in sorted(self.rebalance_counter.items(), key=lambda x: x[0]):
    #   print("at height {} number of rebalanced occurred: {}".format(key, value))
    # for (auto it = rebalance_counter.begin(); it != rebalance_counter.end(); it++)
    # {
    #   print("at height " << it->first << " number of rebalanced occurred: " << it->second << endl;
    # }

    print("\nReads happen during rebalance (weighted) by PMA height: ")
    for i in range(self.segment_count * 2):
      if self.rebalance_reads[i] != np.int64(0):
        print("at height {} number of Reads occurred: {}".format(i, self.rebalance_reads[i]))
    # for key, value in sorted(self.rebalance_reads.items(), key=lambda x: x[0]):
    #   print("at height {} number of Reads occurred: {}".format(key, value))
    # for (auto it = rebalance_reads.begin(); it != rebalance_reads.end(); it++)
    # {
    #   print("at height " << it->first << " number of Reads occurred: " << it->second << endl;
    # }

    print("\nWrites happen during rebalance (weighted) by PMA height: ")
    for i in range(self.segment_count * 2):
      if self.rebalance_writes[i] != np.int64(0):
        print("at height {} number of Writes occurred: {}".format(i, self.rebalance_writes[i]))
    # for key, value in sorted(self.rebalance_writes.items(), key=lambda x: x[0]):
    #   print("at height {} number of Writes occurred: {}".format(key, value))
    # for (auto it = rebalance_writes.begin(); it != rebalance_writes.end(); it++)
    # {
    #   print("at height " << it->first << " number of Writes occurred: " << it->second << endl;
    # }

    print("\nPMA Statistics End   ----------------")


  def compute_capacity(self):
    # Ideal segment size
    self.segment_size = np.int32(self.ceil_log2(self.num_vertices))
    # Ideal number of segments
    self.segment_count = np.int64(self.ceil_div(self.num_vertices, self.segment_size))

    # The number of segments has to be a power of 2, though.
    self.segment_count = np.int64(self.hyperfloor(self.segment_count))
    # Update the segment size accordingly
    self.segment_size = np.int32(self.ceil_div(self.num_vertices, self.segment_count))

    # correcting the number of vertices
    self.num_vertices = self.segment_count * self.segment_size
    self.avg_degree = self.ceil_div(self.num_edges, self.num_vertices)

    self.elem_capacity = np.int64(self.num_edges * self.max_sparseness)


  def load_basegraph(self, input_file):
    with open(input_file) as file:
      for line in file:
        u, v = line.split()

        # type casting to int
        u = int(u)
        v = int(v)

        self.vertices_[u].degree += np.int32(1)
        seg_id = int(self.get_segment_id(u))
        while seg_id > 0:
          self.segment_edges_actual[seg_id] += np.int64(1)
          seg_id //= 2

    pos = int(0)
    for i in range(self.num_vertices):
      self.vertices_[u].index = np.int64(pos)
      pos += self.vertices_[u].degree

    self.spread_weighted(0, self.num_vertices)


  # def load_dynamicgraph(self, input_file, output_file):
  def load_dynamicgraph(self, input_file):
    with open(input_file) as file:
      for line in file:
        u, v = line.split()

        # type casting to int
        u = int(u)
        v = int(v)

        # self.insert(u, v, output_file)
        self.insert(u, v)

  # def insert(self, src, dst, output_file):
  def insert(self, src, dst):
    # find an empty slot to the right of the vertex
    # move the data from loc to the free slot
    current_segment = self.get_segment_id(src)
    loc = self.vertices_[src].index + self.vertices_[src].degree
    right_free_slot = -1
    left_free_slot = -1
    left_vertex = src
    right_vertex = src
    left_vertex_boundary = int(src)
    right_vertex_boundary = int(src)

    if self.segment_edges_total[current_segment] > self.segment_edges_actual[current_segment]:
      left_vertex_boundary = int(int(src / self.segment_size) * self.segment_size)
      right_vertex_boundary = min(left_vertex_boundary + self.segment_size, self.num_vertices - 1)
    else:
      curr_seg_size = self.segment_size
      j = current_segment
      while j > 0:
        if self.segment_edges_total[j] > self.segment_edges_actual[j]:
          break
        j //= 2
        curr_seg_size *= 2
      left_vertex_boundary = int((src // curr_seg_size) * curr_seg_size)
      right_vertex_boundary = min(left_vertex_boundary + curr_seg_size, self.num_vertices - 1)

    # search right side for a free slot
    for i in range(src, right_vertex_boundary):
      if self.vertices_[i + 1].index > (self.vertices_[i].index + self.vertices_[i].degree):
        right_free_slot = self.vertices_[i].index + self.vertices_[i].degree  # we get a free slot here
        right_vertex = i
        break

    # in the last segment where we skipped the last vertex
    if (right_free_slot == -1
            and right_vertex_boundary == (self.num_vertices - 1)
            and self.elem_capacity > (
                    1 + self.vertices_[self.num_vertices - 1].index + self.vertices_[self.num_vertices - 1].degree)):
      right_free_slot = self.vertices_[self.num_vertices - 1].index + self.vertices_[self.num_vertices - 1].degree
      right_vertex = self.num_vertices - 1

    # if no space on the right side, search the left side
    if right_free_slot == -1:
      for i in range(src, left_vertex_boundary, -1):
        if self.vertices_[i].index > (self.vertices_[i - 1].index + self.vertices_[i - 1].degree):
          left_free_slot = self.vertices_[i].index - 1  # we get a free slot here
          left_vertex = i
          break

    # print("left_free_slot: {}, right_free_slot: {}".format(left_free_slot, right_free_slot))
    # print("left_free_vertex: {}, right_free_vertex: {}".format(left_vertex, right_vertex))

    # found free slot on the right
    if right_free_slot != -1:
      # move elements to the right to get the free slot
      if right_free_slot >= loc:
        # move edges towards right_free_slot position
        # in the original VCSR implementation, we actually move edges here
        # for i in range(right_free_slot, loc, -1):
        self.num_write_insert += (right_free_slot - loc)
        self.num_read_insert += (right_free_slot - loc)

        # if right_free_slot > loc:
        #   output_file.write(
        #     f"{src} {(right_vertex - src)} {self.vertices_[src].degree} {(right_free_slot - loc)} {self.vertices_[src + 1].degree} {self.vertices_[src - 1].degree}\n")

        # update vertex metadata
        for i in range(src + 1, right_vertex + 1):
          self.vertices_[i].index += np.int64(1)
          self.num_write_insert += 1
          self.num_read_insert += 1

        # update the segment_edges_total for the source-vertex and right-vertex's segment if it lies in different segment
        if current_segment != self.get_segment_id(right_vertex):
          self.update_segment_edge_total(src, 1)
          self.update_segment_edge_total(right_vertex, -1)

      # add new edge at loc
      self.num_write_insert += 1
      self.vertices_[src].degree += np.int32(1)
      self.num_write_insert += 1
      self.num_read_insert += 1

    elif left_free_slot != -1:
      if left_free_slot < loc:
        # move edges towards left_free_slot position
        # in the original VCSR implementation, we actually move edges here
        # for i in (left_free_slot, loc - 1):
        self.num_write_insert += (loc - 1 - left_free_slot)
        self.num_read_insert += (loc - 1 - left_free_slot)


        # output_file.write(
        #     f"{src} {(left_vertex - src)} {self.vertices_[src].degree} {(loc - 1 - left_free_slot)} {self.vertices_[src + 1].degree} {self.vertices_[src - 1].degree}\n")

        # update vertex metadata
        for i in range(left_vertex, src + 1):
          self.vertices_[i].index -= np.int64(1)
          self.num_write_insert += 1
          self.num_read_insert += 1

        # update the segment_edges_total for the source-vertex and left-vertex's segment if it lies in different segment
        assert left_vertex > 0, f"left-vertex should be larger than 0, got: {left_vertex}"
        if current_segment != self.get_segment_id(left_vertex - 1):
          self.update_segment_edge_total(src, 1)
          self.update_segment_edge_total(left_vertex - 1, -1)

      # add new edge at (loc - 1)
      self.num_write_insert += 1
      self.vertices_[src].degree += np.int32(1)
      self.num_write_insert += 1
      self.num_read_insert += 1
    else:
      assert 0 == 1, f"Should not happen num-edges: {self.num_edges}, src: {src}, dst: {dst}, left_free_slot: {left_free_slot}, right_free_slot: {right_free_slot}, segment_edges_total: {self.segment_edges_total[current_segment]}, segment_edges_actual: {self.segment_edges_actual[current_segment]}"

    # we insert a new edge, increasing the degree of the whole subtree
    j = current_segment
    while j > 0:
      self.segment_edges_actual[j] += np.int64(1)
      j //= 2

    self.num_edges += 1
    # check whether we need to call for a rebalance
    self.rebalance_wrapper(src)

  def rebalance_wrapper(self, src):
    height = np.int32(0)
    window = self.get_segment_id(src)
    density = self.segment_edges_actual[window] / self.segment_edges_total[window]

    up_height = self.up_0 - (height * self.delta_up)
    low_height = self.low_0 + (height * self.delta_low)

    # print("window: {}, segment_edges_actual: {}, segment_edges_total: {}, density: {}".format(window, self.segment_edges_actual[window], self.segment_edges_total[window], density))
    # print("Window: {}, density: {}, up_height: {}, low_height: {}".format(window, density, up_height, low_height))

    while window > 0 and (density >= up_height):
      # Repeatedly check window containing an increasing amount of segments
      # Now that we recorded the number of elements and occupancy in segment_edges_total and segment_edges_actual respectively
      # so, we are going to check if the current window can fulfill the density thresholds.
      # density = gap / segment-size

      # Go one level up in our conceptual PMA tree
      window //= 2
      height += 1

      up_height = self.up_0 - (height * self.delta_up)
      low_height = self.low_0 + (height * self.delta_low)

      density = self.segment_edges_actual[window] / self.segment_edges_total[window]
      # print(">>> Window: {}, density: {}, up_height: {}, low_height: {}".format(window, density, up_height, low_height))

    # print("height: {}".format(height))
    if height == 0:
      # rebalance is not required in the single pma leaf
      return

    left_index = src
    right_index = src
    if density < up_height:
      # Found a window within threshold
      window_size = self.segment_size * (1 << height)
      left_index = (src // window_size) * window_size
      right_index = min(left_index + window_size, self.num_vertices)

      # do degree-based distribution of gaps
      num_write_rebal_prev = self.num_write_rebal
      num_read_rebal_prev = self.num_read_rebal
      if self.use_rl:
        self.rebalance_rl(left_index, right_index, window)
      else:
        self.rebalance_weighted(left_index, right_index, window)
      self.update_rebalance_metadata(height, 1, (self.num_read_rebal - num_read_rebal_prev), (self.num_write_rebal - num_write_rebal_prev))
    else:
      # Rebalance not possible without increasing the underlying array size.
      # need to resize the size of "edges_" array
      self.resize()


  def resize(self):
    self.num_resize += 1
    print("elem_capacity: {}".format(self.elem_capacity))
    self.elem_capacity *= np.int64(2)

    gaps = self.elem_capacity - self.num_edges
    if self.use_rl:
      new_indices = self.calculate_positions_rl(0, self.num_vertices, 1)
    else:
      new_indices = self.calculate_positions(0, self.num_vertices, gaps, self.num_edges)

    # update the vertex metadata
    # in the original VCSR implementation, we actually move edges here
    for curr_vertex in range(self.num_vertices - 1, -1, -1):
      self.vertices_[curr_vertex].index = np.int64(new_indices[curr_vertex])

    self.num_read_resize += (self.num_vertices + self.num_edges)
    self.num_write_resize += (self.num_vertices + self.num_edges)

    self.recount_segment_total_full()
    # self.segment_sanity_check()
    # self.edge_list_boundary_sanity_checker()

    # TODO Check if this is correct
    # ================================================================================================
    # Mirror RL logging on resize (treat as root-level event): reward_step, graph_loaded_pct, total_edge_count
    try:
      total_writes = int(self.num_write_insert + self.num_write_rebal)
      reward = -(total_writes - int(self._last_total_writes))
      adjusted_reward = float(reward) / 1.0

      total_capacity_root = float(self.segment_edges_total[1]) if self.segment_edges_total[1] > 0 else 0.0
      percent_loaded = (float(self.segment_edges_actual[1]) / total_capacity_root) if total_capacity_root > 0.0 else 0.0

      self._writer.add_scalar('charts/reward_step', adjusted_reward, self._global_step)
      self._writer.add_scalar('charts/graph_loaded_pct', float(percent_loaded), self._global_step)
      self._writer.add_scalar('charts/total_edge_count', int(self.num_edges), self._global_step)

      self._last_total_writes = total_writes
      self._global_step += 1
    except Exception:
      pass
    # ================================================================================================

  def spread_weighted(self, start_vertex, end_vertex):
    assert start_vertex == 0, f"start-vertex is expected to be 0, got: {start_vertex}"
    gaps = self.elem_capacity - self.num_edges
    new_positions = self.calculate_positions(start_vertex, end_vertex, gaps, self.num_edges)

    read_index = np.dtype(np.int64)
    write_index = np.dtype(np.int64)
    curr_degree = np.dtype(np.int64)

    # update vertex metadata based on the calculated new_positions
    # in the original VCSR implementation, we actually move edges here
    for curr_vertex in range(end_vertex-1, start_vertex, -1):
      self.vertices_[curr_vertex].index = np.int64(new_positions[curr_vertex])

    # free(new_positions)
    # new_positions = nullptr
    self.recount_segment_total_full()

    # self.segment_sanity_check()
    # self.edge_list_boundary_sanity_checker()


  # allocate gaps based on degree
  def calculate_positions(self, start_vertex, end_vertex, gaps, total_degree):
    size = end_vertex - start_vertex
    new_index = [np.int64(0)] * size
    total_degree += size

    if gaps <= 0:
      print("Gaps: {}, size: {}, start-vertex: {}, end-vertex: {}".format(gaps, size, start_vertex, end_vertex))
      # print("actual-edge: {}, total-edge: {}")

    index_d = np.float64(self.vertices_[start_vertex].index)
    step = np.float64(gaps) / total_degree # gaps possible per-edge
    for i in range(start_vertex, end_vertex):
      new_index[i-start_vertex] = int(index_d)
      if i > start_vertex:
        # printf("v[%d] with degree %d gets actual space %ld\n", i-1, vertices_[i-1].degree, (new_index[i-start_vertex]-new_index[i-start_vertex-1]));
        assert new_index[i-start_vertex] >= new_index[(i-1)-start_vertex] + self.vertices_[i-1].degree, f"Edge-list can not be overlapped with the neighboring vertex! Gaps: {gaps}, total-degree: {total_degree}, step-size: {step}"

      # index_d += (vertices_[i].degree + (step * vertices_[i].degree))
      index_d += (self.vertices_[i].degree + (step * (self.vertices_[i].degree + 1)))

    return new_index


  def rebalance_weighted(self, start_vertex, end_vertex, pma_idx):
    start_vertex = int(start_vertex)
    end_vertex = int(end_vertex)
    pma_idx = int(pma_idx)

    from_idx = self.vertices_[start_vertex].index
    if end_vertex >= self.num_vertices:
      to_idx = self.elem_capacity
    else:
      to_idx = self.vertices_[end_vertex].index

    assert (to_idx > from_idx), f"Invalid range found while doing weighted rebalance"
    capacity = to_idx - from_idx

    # assert (self.segment_edges_total[pma_idx] == capacity), f"Segment capacity: {capacity} is not matched with segment_edges_total: {self.segment_edges_total[pma_idx]}, to-idx: {to_idx}, from-idx: {from_idx}."
    gaps = self.segment_edges_total[pma_idx] - self.segment_edges_actual[pma_idx]

    # calculate the future positions of the vertices_[i].index
    size = end_vertex - start_vertex

    if pma_idx == 1:
      print("[rebalance_weighted] total-edges: {}, actual-edges: {}".format(self.segment_edges_total[1], self.segment_edges_actual[1]))

    new_index = self.calculate_positions(start_vertex, end_vertex, gaps, self.segment_edges_actual[pma_idx])
    if end_vertex >= self.num_vertices:
      index_boundary = self.elem_capacity
    else:
      index_boundary = self.vertices_[end_vertex].index

    assert (new_index[size - 1] + self.vertices_[end_vertex - 1].degree <= index_boundary), f"Rebalance (weighted) index calculation is wrong! new_index[size - 1]: {new_index[size - 1]}, index_boundary: {index_boundary}, calculated: {new_index[size - 1] + self.vertices_[end_vertex - 1].degree}, total-edges: {self.segment_edges_total[pma_idx]}, actual-edges: {self.segment_edges_actual[pma_idx]}, start-vertex: {start_vertex}, end-vertex: {end_vertex}"

    ii = 0
    jj = 0
    curr_vertex = start_vertex + 1
    next_to_start = 0
    read_index = 0
    last_read_index = 0
    write_index = 0

    while curr_vertex < end_vertex:
      for ii in range(curr_vertex, end_vertex):
        if new_index[ii-start_vertex] <= self.vertices_[ii].index:
          break
      if ii == end_vertex:
        ii -= 1
      next_to_start = ii + 1
      if new_index[ii-start_vertex] <= self.vertices_[ii].index:
        # now it is guaranteed that, ii's new-starting index is less than or equal to it's old-starting index
        jj = ii
        read_index = self.vertices_[jj].index
        last_read_index = read_index + self.vertices_[jj].degree
        write_index = new_index[jj - start_vertex]
        self.num_read_rebal += 2

        self.num_write_rebal += (last_read_index - read_index)
        self.num_read_rebal += (last_read_index - read_index)
        # update the index to the new position
        # in the original VCSR implementation, we actually move edges here
        # while read_index < last_read_index:
        self.vertices_[jj].index = np.int64(new_index[jj - start_vertex])
        self.num_write_rebal += 1
        ii -= 1

      # from current_vertex to ii, the new-starting index is greater than to it's old-starting index
      for jj in range(ii, curr_vertex-1, -1):
        read_index = self.vertices_[jj].index + self.vertices_[jj].degree - 1
        last_read_index = self.vertices_[jj].index
        write_index = new_index[jj-start_vertex] + self.vertices_[jj].degree - 1
        self.num_read_rebal += 2

        self.num_write_rebal += (read_index - last_read_index + 1)
        self.num_read_rebal += (read_index - last_read_index + 1)
        # update the index to the new position
        # in the original VCSR implementation, we actually move edges here
        # while read_index >= last_read_index:
        self.vertices_[jj].index = np.int64(new_index[jj-start_vertex])
        self.num_write_rebal += 1

      # move current_vertex to the next position of ii
      curr_vertex = next_to_start

    # free(new_index)
    # new_index = nullptr
    self.recount_segment_total_in_range(start_vertex, end_vertex)
    # self.segment_sanity_check()
    # self.edge_list_boundary_sanity_checker()

    # TODO Check if this is correct
    # ================================================================================================
    # Mirror RL logging: reward_step (delta writes, level-weighted), graph_loaded_pct, total_edge_count
    try:
      # Compute tree level from pma_idx (root=1 => level=0)
      level = 0
      tmp = int(pma_idx)
      while tmp > 1:
        tmp //= 2
        level += 1
      div_factor = max(1, level + 1)

      total_writes = int(self.num_write_insert + self.num_write_rebal)
      reward = -(total_writes - int(self._last_total_writes))
      adjusted_reward = float(reward) / float(div_factor)

      total_capacity_root = float(self.segment_edges_total[1]) if self.segment_edges_total[1] > 0 else 0.0
      percent_loaded = (float(self.segment_edges_actual[1]) / total_capacity_root) if total_capacity_root > 0.0 else 0.0

      self._writer.add_scalar('charts/reward_step', adjusted_reward, self._global_step)
      self._writer.add_scalar('charts/graph_loaded_pct', float(percent_loaded), self._global_step)
      self._writer.add_scalar('charts/total_edge_count', int(self.num_edges), self._global_step)

      self._last_total_writes = total_writes
      self._global_step += 1
    except Exception:
      pass
    # ================================================================================================

  def update_segment_edge_total(self, vid, count):
    sid = self.get_segment_id(vid)
    # assert count >= 0, f"count should not be a negative number, received: {count}"
    while sid > 0:
      self.segment_edges_total[sid] += np.int64(count)
      sid //= 2


  def recount_segment_total_full(self):
    # count the size of each segment in the tree
    # memset(segment_edges_total, 0, sizeof(int64_t)*segment_count*2)
    for i in range (self.segment_count*2):
      self.segment_edges_total[i] = np.int64(0)

    for i in range(self.segment_count):
      # next_starter = (i == (segment_count - 1)) ? (elem_capacity) : vertices_[(i+1)*segment_size].index
      if i == self.segment_count - 1:
        next_starter = self.elem_capacity
      else:
        next_starter = self.vertices_[(i+1) * self.segment_size].index

      segment_total_p = next_starter - self.vertices_[i*self.segment_size].index
      # assert segment_total_p >= 0, f"segment_total_p should not be a negative number, here: {segment_total_p}"
      j = i + self.segment_count # tree leaves
      while j > 0:
        self.segment_edges_total[j] += np.int64(segment_total_p)
        j //= 2

  def get_unique_filename(self, filename):
    counter = 0
    new_filename = filename
    while os.path.exists(new_filename):
      counter += 1
      new_filename = f"{filename}_{counter}"
    return new_filename

  def recount_segment_total_in_range(self, start_vertex, end_vertex):
    start_seg = self.get_segment_id(start_vertex) - self.segment_count
    end_seg = self.get_segment_id(end_vertex) - self.segment_count
    for i in range(start_seg, end_seg):
      if i == self.segment_count - 1:
        next_starter = self.elem_capacity
      else:
        next_starter = self.vertices_[(i+1) * self.segment_size].index

      segment_total_p = next_starter - self.vertices_[i*self.segment_size].index
      j = i + self.segment_count # tree leaves
      segment_total_p -= self.segment_edges_total[j]  # getting the absolute difference
      # assert segment_total_p >= 0, f"segment_total_p should not be a negative number, here: {segment_total_p}"
      while j > 0:
        self.segment_edges_total[j] += np.int64(segment_total_p)
        j //= 2


  def segment_sanity_check(self):
    for i in range(0, self.segment_count):
      j = i + self.segment_count  # tree leaves
      while j > 0:
        assert self.segment_edges_total[j] >= self.segment_edges_actual[j], f"Rebalancing (adaptive) segment boundary invalidate segment capacity!"
        j //= 2


  def edge_list_boundary_sanity_checker(self):
    for curr_vertex in range(1, self.num_vertices):
      if self.vertices_[curr_vertex - 1].index + self.vertices_[curr_vertex - 1].degree > self.vertices_[curr_vertex].index:
        print("**** Invalid edge-list boundary found at vertex-id: {} index: {}, degree: {}, next vertex start at: {}".format(curr_vertex - 1, self.vertices_[curr_vertex - 1].index, self.vertices_[curr_vertex - 1].degree, self.vertices_[curr_vertex].index))
      assert self.vertices_[curr_vertex - 1].index + self.vertices_[curr_vertex - 1].degree <= self.vertices_[curr_vertex].index, f"Invalid edge-list boundary found!"
    assert self.vertices_[self.num_vertices - 1].index + self.vertices_[self.num_vertices - 1].degree <= self.elem_capacity, f"Invalid edge-list boundary found!"


  def get_segment_id(self, vid):
    return int((vid // self.segment_size) + self.segment_count)


  @staticmethod
  def clzll(x):
    leading = 64
    v = x
    # v = v.astype("uint64")
    while v:
      v = (v >> np.uint64(1))
      leading -= 1
    return leading


  @staticmethod
  def last_bit_set(x):
    # return (np.dtype(np.uint64).itemsize * 8 - __builtin_clzll(x)) # Linux
    return (np.dtype(np.uint64).itemsize * 8 - VCSR.clzll(np.uint64(x)))


  @staticmethod
  def floor_log2(x):
    return (VCSR.last_bit_set(x) - 1)


  @staticmethod
  def ceil_log2(x):
    assert x > 0, f"number greater than 0 expected, got: {x}"
    return (VCSR.last_bit_set(x - 1))


  @staticmethod
  def floor_div(x, y):
    return (x / y)


  @staticmethod
  def ceil_div(x, y):
    if(x == 0):
      return 0
    return (1 + ((x - 1) / y))


  @staticmethod
  def hyperfloor(x):
    return (np.uint64(1) << np.uint64(VCSR.floor_log2(x)))


  @staticmethod
  def hyperceil(x):
    return (np.uint64(1) << np.uint64(VCSR.ceil_log2(x)))

  # ================= RL-augmented methods =================
  def _make_obs(self, seg_id: int, parent_gaps: float, level: int) -> np.ndarray:
    left = seg_id * 2
    right = seg_id * 2 + 1
    left_child_total = float(self.segment_edges_total[left]) if self.segment_edges_total[left] > 0 else 0.0
    right_child_total = float(self.segment_edges_total[right]) if self.segment_edges_total[right] > 0 else 0.0
    left_child_degree = float(self.segment_edges_actual[left])
    right_child_degree = float(self.segment_edges_actual[right])
    left_density = (left_child_degree / left_child_total) if left_child_total > 0.0 else 0.0
    right_density = (right_child_degree / right_child_total) if right_child_total > 0.0 else 0.0
    sum_degree = left_child_degree + right_child_degree
    left_ratio = (left_child_degree / sum_degree) if sum_degree > 0.0 else 0.0
    right_ratio = (right_child_degree / sum_degree) if sum_degree > 0.0 else 0.0
    parent_total = float(self.segment_edges_total[seg_id]) if self.segment_edges_total[seg_id] > 0 else 0.0
    parent_gaps_norm = float(parent_gaps) / parent_total if parent_total > 0.0 else 0.0
    level_norm = float(level) / float(self.tree_height if self.tree_height > 0 else 1)
    return np.array([left_density, right_density, left_ratio, right_ratio, parent_gaps_norm, level_norm], dtype=np.float32)

  def calculate_positions_rl(self, start_vertex, end_vertex, pma_idx):
    size = end_vertex - start_vertex
    new_index = [np.int64(0)] * size
    gaps_total = self.segment_edges_total[pma_idx] - self.segment_edges_actual[pma_idx]
    seg_gaps = {pma_idx: float(gaps_total)}
    seg_queue = [pma_idx]
    self._rl_decision_queue = []
    while seg_queue:
      seg_id = seg_queue.pop(0)
      parent_gaps = seg_gaps.get(seg_id, 0.0)
      if seg_id >= self.segment_count:
        continue
      # compute tree level
      level = 0
      tmp = int(seg_id)
      while tmp > 1:
        tmp //= 2
        level += 1
      obs = self._make_obs(seg_id, parent_gaps, level)
      act, logp, val = self.agent.policy_step(obs)
      if self._continuous:
        # action in (-1,1) -> map to [0.1, 0.9]
        ratio = 0.5 * (act[0] + 1.0)
        ratio = float(np.clip(0.1 + ratio * 0.8, 0.1, 0.9))
      else:
        # discrete 0..8 -> 0.1..0.9
        idx = int(np.clip(act, 0, 8))
        ratio = 0.1 + 0.1 * idx
      # snapshot baseline writes
      before_writes = int(self.num_write_insert + self.num_write_rebal)
      self._rl_decision_queue.append({
        'obs': obs,
        'act': np.array([ratio], dtype=np.float32) if self._continuous else idx,
        'val': val,
        'logp': logp,
        'level': level,
        'before_writes': before_writes
      })
      left_child = seg_id * 2
      right_child = seg_id * 2 + 1
      seg_gaps[left_child] = float(parent_gaps) * ratio
      seg_gaps[right_child] = float(parent_gaps) - float(seg_gaps[left_child])
      if left_child < self.segment_count:
        seg_queue.append(left_child)
        seg_queue.append(right_child)

    start_seg = self.get_segment_id(start_vertex)
    end_seg = self.get_segment_id(end_vertex)
    num_segs = end_seg - start_seg
    new_seg_index = [np.int64(0)] * num_segs
    seg_index_d = self.vertices_[start_vertex].index
    for seg_id in range(start_seg, end_seg):
      new_seg_index[seg_id - start_seg] = np.int64(seg_index_d)
      seg_index_d += (self.segment_edges_actual[seg_id] + seg_gaps.get(seg_id, 0.0))

    for seg_id in range(start_seg, end_seg):
      current_seg_start_vertex = self.get_segment_id(self.get_segment_start_vertex(seg_id))
      current_seg_start_vertex = self.get_segment_start_vertex(seg_id)
      current_seg_end_vertex = min(current_seg_start_vertex + self.segment_size, self.num_vertices)
      index_d = new_seg_index[seg_id - start_seg]
      denom = self.segment_edges_actual[seg_id]
      step = (float(seg_gaps.get(seg_id, 0.0)) / float(denom)) if denom > 0 else 0.0
      for i in range(current_seg_start_vertex, current_seg_end_vertex):
        new_index[i - start_vertex] = int(index_d)
        if i > start_vertex:
          assert new_index[i-start_vertex] >= new_index[(i-1)-start_vertex] + self.vertices_[i-1].degree, "Edge-list overlap in RL index calc"
        index_d += (self.vertices_[i].degree + (step * self.vertices_[i].degree))
    return new_index

  def rebalance_rl(self, start_vertex, end_vertex, pma_idx):
    start_vertex = int(start_vertex)
    end_vertex = int(end_vertex)
    pma_idx = int(pma_idx)
    from_idx = self.vertices_[start_vertex].index
    to_idx = self.elem_capacity if end_vertex >= self.num_vertices else self.vertices_[end_vertex].index
    assert (to_idx > from_idx), "Invalid range for RL rebalance"
    new_index = self.calculate_positions_rl(start_vertex, end_vertex, pma_idx)
    index_boundary = self.elem_capacity if end_vertex >= self.num_vertices else self.vertices_[end_vertex].index
    size = end_vertex - start_vertex
    assert (new_index[size - 1] + self.vertices_[end_vertex - 1].degree <= index_boundary), "RL rebalance index calc error"

    ii = 0
    jj = 0
    curr_vertex = start_vertex + 1
    next_to_start = 0
    read_index = 0
    last_read_index = 0
    write_index = 0
    while curr_vertex < end_vertex:
      for ii in range(curr_vertex, end_vertex):
        if new_index[ii-start_vertex] <= self.vertices_[ii].index:
          break
      if ii == end_vertex:
        ii -= 1
      next_to_start = ii + 1
      if new_index[ii-start_vertex] <= self.vertices_[ii].index:
        jj = ii
        read_index = self.vertices_[jj].index
        last_read_index = read_index + self.vertices_[jj].degree
        write_index = new_index[jj - start_vertex]
        self.num_read_rebal += 2
        delta1 = int(last_read_index) - int(read_index)
        self.num_write_rebal += delta1
        self.num_read_rebal += delta1
        self.vertices_[jj].index = np.int64(new_index[jj - start_vertex])
        self.num_write_rebal += 1
        ii -= 1
      for jj in range(ii, curr_vertex-1, -1):
        read_index = self.vertices_[jj].index + self.vertices_[jj].degree - 1
        last_read_index = self.vertices_[jj].index
        write_index = new_index[jj-start_vertex] + self.vertices_[jj].degree - 1
        self.num_read_rebal += 2
        delta2 = int(read_index) - int(last_read_index) + 1
        self.num_write_rebal += delta2
        self.num_read_rebal += delta2
        self.vertices_[jj].index = np.int64(new_index[jj-start_vertex])
        self.num_write_rebal += 1
      curr_vertex = next_to_start

    self.recount_segment_total_in_range(start_vertex, end_vertex)

    # Assign rewards for each RL decision (delta writes, level-weighted)
    if self.use_rl and self._rl_decision_queue:
      total_writes_now = int(self.num_write_insert + self.num_write_rebal)
      for item in self._rl_decision_queue:
        delta = total_writes_now - int(item['before_writes'])
        lvl = int(item['level'])
        div = max(1, lvl + 1)
        rew = -float(delta) / float(div)
        self.agent.store(item['obs'], item['act'], rew, item['val'], item['logp'], False)
        # TensorBoard mirrors (step-wise)
        try:
          total_capacity_root = float(self.segment_edges_total[1]) if self.segment_edges_total[1] > 0 else 0.0
          percent_loaded = (float(self.segment_edges_actual[1]) / total_capacity_root) if total_capacity_root > 0.0 else 0.0
          self._writer.add_scalar('charts/reward_step', rew, self._global_step)
          self._writer.add_scalar('charts/graph_loaded_pct', float(percent_loaded), self._global_step)
          self._writer.add_scalar('charts/total_edge_count', int(self.num_edges), self._global_step)
          self._global_step += 1
        except Exception:
          pass
      self.agent.finish_trajectory(last_val=0.0)
      self.agent.maybe_update()



# Test: Amazon
# vcsr = VCSR(403394, 488681)
# # print(', '.join("%s: %s" % item for item in vcsr.items()))
# # dir(vcsr)
# # vcsr.__dict__
# vcsr.load_basegraph("./graph-datasets/amazon0601.base.el")
# vcsr.print_pma_meta()
# # vcsr.insert(0, 1)
# vcsr.load_dynamicgraph("./graph-datasets/amazon0601.dynamic.el")
# vcsr.print_pma_meta()
# vcsr.print_pma_counter()

# Test: Stackoverflow
# # initializing VCSR with @param num_vertices, num_edges
# vcsr_sk = VCSR(6024271, 5772481)
# # loading basegraph into VCSR
# vcsr_sk.load_basegraph("/mnt/cci-files/dataset-dgap/stackoverflow_10/sx-unique-undir.base.el")
# vcsr_sk.print_pma_meta()
# # inserting dynamic graps into VCSR
# vcsr_sk.load_dynamicgraph("/mnt/cci-files/dataset-dgap/stackoverflow_10/sx-unique-undir.dynamic.el")
# vcsr_sk.print_pma_meta()
# vcsr_sk.print_pma_counter()

# # Test: Enron
# # initializing VCSR with @param num_vertices, num_edges
# vcsr_enron = VCSR(87274, 59492)
# # loading basegraph into VCSR
# vcsr_enron.load_basegraph("/mnt/cci-files/dataset-dgap/enron_10/enron-unique-undir.base.el")
# vcsr_enron.print_pma_meta()
# # inserting dynamic graps into VCSR
# vcsr_enron.load_dynamicgraph("/mnt/cci-files/dataset-dgap/enron_10/enron-unique-undir.dynamic.el")
# vcsr_enron.print_pma_meta()
# vcsr_enron.print_pma_counter()

#########
######### test on scaled graph (10% of total graph)
#########

# # Test: Amazon
# vcsr = VCSR(403394, 48868)
# # print(', '.join("%s: %s" % item for item in vcsr.items()))
# # dir(vcsr)
# # vcsr.__dict__
# vcsr.load_basegraph("/mnt/cci-files/dataset-learnedcsr/amazon/amazon0601.base.el")
# vcsr.print_pma_meta()
# # vcsr.insert(0, 1)
# vcsr.load_dynamicgraph("/mnt/cci-files/dataset-learnedcsr/amazon/amazon0601.dynamic.el")
# vcsr.print_pma_meta()
# vcsr.print_pma_counter()

def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description="Process command line arguments.")

  # Add the command-line arguments
  parser.add_argument("--nv", default=88581, type=int, help="Number of vertices")
  parser.add_argument("--ne", default=37598, type=int, help="Number of edges")
  parser.add_argument("--base_file", default="rl4sys/examples/dgap/sx-mathoverflow-unique-undir.base.el", type=str, help="Base file")
  parser.add_argument("--dynamic_file", default="rl4sys/examples/dgap/sx-mathoverflow-unique-undir.dynamic.el", type=str, help="Dynamic file")

  # Parse the command-line arguments
  args = parser.parse_args()

  # Extract the argument values
  nv = args.nv
  ne = args.ne
  base_file = args.base_file
  dynamic_file = args.dynamic_file

  # process_items(nv, ne, base_file, dynamic_file)
  vcsr = VCSR(nv, ne)
  # unique_filename = vcsr.get_unique_filename("your_filename.txt")
  # loading basegraph into VCSR
  vcsr.load_basegraph(base_file)
  vcsr.print_pma_meta()
  # inserting dynamic graps into VCSR
  start = time.time()
  # with open(unique_filename, 'a') as file:
  #   vcsr.load_dynamicgraph(dynamic_file, file)
  for _i in range(1):
    vcsr.load_dynamicgraph(dynamic_file)
  end = time.time()
  print("D-Graph Build Time: {} seconds.".format(end - start))
  vcsr.print_pma_meta()
  vcsr.print_pma_counter()


if __name__ == "__main__":
  main()

# Test: Amazon
# python3 py-vcsr-noml.py --nv 403394 --ne 488681 --base_file /mnt/cci-files/dataset-dgap/amazon0601_10-90/amazon0601.base.el --dynamic_file /mnt/cci-files/dataset-dgap/amazon0601_10-90/amazon0601.dynamic.el

# Test: Stackoverflow
# python3 py-vcsr-noml.py --nv 6024271 --ne 5772481 --base_file /mnt/cci-files/dataset-dgap/stackoverflow_10/sx-unique-undir.base.el --dynamic_file /mnt/cci-files/dataset-dgap/stackoverflow_10/sx-unique-undir.dynamic.el

# Test: Enron
# python3 py-vcsr-noml.py --nv 87274 --ne 59492 --base_file /mnt/cci-files/dataset-dgap/enron_10/enron-unique-undir.base.el --dynamic_file /mnt/cci-files/dataset-dgap/enron_10/enron-unique-undir.dynamic.el

# Test: Amazon 10%
# python3 py-vcsr-noml.py --nv 403394 --ne 48868 --base_file /mnt/cci-files/dataset-learnedcsr/amazon/10/amazon0601.base.el --dynamic_file /mnt/cci-files/dataset-learnedcsr/amazon/10/amazon0601.dynamic.el
