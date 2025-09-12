import os
import time
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple

# Define the structure for queue items
QueueItem = namedtuple("QueueItem", ["num_edge", "reward_counter"])

class Vertex:
  def __init__(self, index, degree):
    self.index = np.int64(index)
    self.degree = np.int32(degree)

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

  def __init__(self, n_vertices, n_edges):
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

    self.pma_root = 1  # root of the pma tree
    self.reward_tracker = deque()
    self.feedback_threshold = 1000


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
        if self.reward_tracker and (self.num_edges - self.reward_tracker[0].num_edge) >= self.feedback_threshold:
          while self.reward_tracker and (self.num_edges - self.reward_tracker[0].num_edge) >= self.feedback_threshold:
            # memory access based reward
            total_writes = self.num_write_insert + self.num_write_rebal #+ self.num_write_resize
            reward = -(total_writes - self.reward_tracker[0].reward_counter)
            # percent of graph loaded (occupancy of PMA root)
            total_capacity_root = float(self.segment_edges_total[self.pma_root])
            percent_loaded = (float(self.segment_edges_actual[self.pma_root]) / total_capacity_root) if total_capacity_root > 0.0 else 0.0

            #self.reward_tracker[0].rl4sys_action.update_reward(reward)
            traj_step = self.reward_tracker.popleft()
            #traj_step.rl4sys_action.data["total_edge_count"] = self.num_edges
            self._writer.add_scalar('charts/reward_step', reward, self._global_step)
            self._writer.add_scalar('charts/graph_loaded_pct', float(percent_loaded), self._global_step)
            self._writer.add_scalar('charts/total_edge_count', int(self.num_edges), self._global_step)
            # Log total memory access (reads + writes, including resize)
            total_reads_all = self.num_read_insert + self.num_read_rebal + self.num_read_resize
            total_writes_all = self.num_write_insert + self.num_write_rebal + self.num_write_resize
            self._writer.add_scalar('charts/memory_acess_log', float(total_reads_all + total_writes_all), self._global_step)
            self._writer.add_scalar('charts/memory_reads', float(total_reads_all), self._global_step)
            self._writer.add_scalar('charts/memory_writes', float(total_writes_all), self._global_step)
            self._global_step += 1

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

    total_writes = self.num_write_insert + self.num_write_rebal
    self.reward_tracker.append(QueueItem(self.num_edges, total_writes))

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

    total_writes = self.num_write_insert + self.num_write_rebal
    self.reward_tracker.append(QueueItem(self.num_edges, total_writes))

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
