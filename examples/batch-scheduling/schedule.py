import numpy as np
import argparse
import math
import random

from copy import deepcopy
import os
import sys

import re

from agent import RL4SysAgent
from training_server import TrainingServer


class Job:
    """
    1. Job Number -- a counter field, starting from 1.
    2. Submit Time -- in seconds. The earliest time the log refers to is zero, and is usually the submittal time of the first job. The lines in the log are sorted by ascending submittal times. It makes sense for jobs to also be numbered in this order.
    3. Wait Time -- in seconds. The difference between the job's submit time and the time at which it actually began to run. Naturally, this is only relevant to real logs, not to models.
    4. Run Time -- in seconds. The wall clock time the job was running (end time minus start time).
    We decided to use ``wait time'' and ``run time'' instead of the equivalent ``start time'' and ``end time'' because they are directly attributable to the Scheduler and application, and are more suitable for models where only the run time is relevant.
    Note that when values are rounded to an integral number of seconds (as often happens in logs) a run time of 0 is possible and means the job ran for less than 0.5 seconds. On the other hand it is permissable to use floating point values for time fields.
    5. Number of Allocated Processors -- an integer. In most cases this is also the number of processors the job uses; if the job does not use all of them, we typically don't know about it.
    6. Average CPU Time Used -- both user and system, in seconds. This is the average over all processors of the CPU time used, and may therefore be smaller than the wall clock runtime. If a log contains the total CPU time used by all the processors, it is divided by the number of allocated processors to derive the average.
    7. Used Memory -- in kilobytes. This is again the average per processor.
    8. Requested Number of Processors.
    9. Requested Time. This can be either runtime (measured in wallclock seconds), or average CPU time per processor (also in seconds) -- the exact meaning is determined by a header comment. In many logs this field is used for the user runtime estimate (or upper bound) used in backfilling. If a log contains a request for total CPU time, it is divided by the number of requested processors.
    10. Requested Memory (again kilobytes per processor).
    11. Status 1 if the job was completed, 0 if it failed, and 5 if cancelled. If information about chekcpointing or swapping is included, other values are also possible. See usage note below. This field is meaningless for models, so would be -1.
    12. User ID -- a natural number, between one and the number of different users.
    13. Group ID -- a natural number, between one and the number of different groups. Some systems control resource usage by groups rather than by individual users.
    14. Executable (Application) Number -- a natural number, between one and the number of different applications appearing in the workload. in some logs, this might represent a script file used to run jobs rather than the executable directly; this should be noted in a header comment.
    15. Queue Number -- a natural number, between one and the number of different queues in the system. The nature of the system's queues should be explained in a header comment. This field is where batch and interactive jobs should be differentiated: we suggest the convention of denoting interactive jobs by 0.
    16. Partition Number -- a natural number, between one and the number of different partitions in the systems. The nature of the system's partitions should be explained in a header comment. For example, it is possible to use partition numbers to identify which machine in a cluster was used.
    17. Preceding Job Number -- this is the number of a previous job in the workload, such that the current job can only start after the termination of this preceding job. Together with the next field, this allows the workload to include feedback as described below.
    18. Think Time from Preceding Job -- this is the number of seconds that should elapse between the termination of the preceding job and the submittal of this one.
    """

    def __init__(self, line="0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"):
        line = line.strip()
        s_array = re.split("\\s+", line)
        self.job_id = int(s_array[0])
        self.submit_time = int(s_array[1])
        self.wait_time = int(s_array[2])
        self.run_time = int(s_array[3])
        self.number_of_allocated_processors = int(s_array[4])
        self.average_cpu_time_used = float(s_array[5])
        self.used_memory = int(s_array[6])

        # "requested number of processors" and "number of allocated processors" are typically mixed.
        # I do not know their difference clearly. But it seems to me using a larger one will be sufficient.
        self.request_number_of_processors = int(s_array[7])
        self.number_of_allocated_processors = max(
            self.number_of_allocated_processors, self.request_number_of_processors)
        self.request_number_of_processors = self.number_of_allocated_processors

        self.request_number_of_nodes = -1

        # if we use the job's request time field
        # for model, request_time might be empty. In this case, we set request_time to the run_time
        self.request_time = int(s_array[8])
        if self.request_time == -1:
            self.request_time = self.run_time

        # if we use the run time as the most accurate request time
        # self.request_time = self.run_time + 60
        # if we gradually increase the accuracy of job's request time
        # with a percentage wrong estimation and round to a fixed time: 1,2,3,... hours.
        # this.requestTime = (int) (this.runTime + this.runTime * 0.4);
        # int roundsTo = 60 * 60; //round up to hours
        # this.requestTime = (this.requestTime / roundsTo + 1) * roundsTo;

        self.request_memory = int(s_array[9])
        self.status = int(s_array[10])
        self.user_id = int(s_array[11])
        self.group_id = int(s_array[12])
        self.executable_number = int(s_array[13])
        self.queue_number = int(s_array[14])

        try:
            self.partition_number = int(s_array[15])
        except ValueError:
            self.partition_number = 0

        self.proceeding_job_number = int(s_array[16])
        self.think_time_from_proceeding_job = int(s_array[17])

        self.random_id = self.submit_time

        self.scheduled_time = -1

        self.allocated_machines = None

        self.slurm_in_queue_time = 0
        self.slurm_age = 0
        self.slurm_job_size = 0.0
        self.slurm_fair = 0.0
        self.slurm_partition = 0
        self.slurm_qos = 0
        self.slurm_tres_cpu = 0.0

    def __eq__(self, other):
        return self.job_id == other.job_id

    def __lt__(self, other):
        return self.job_id < other.job_id

    def __hash__(self):
        return hash(self.job_id)

    def __str__(self):
        return "J["+str(self.job_id)+"]-["+str(self.request_number_of_processors)+"]-["+str(self.submit_time)+"]-["+str(self.request_time)+"]"

    def __feature__(self):
        return [self.submit_time, self.request_number_of_processors, self.request_time,
                self.user_id, self.group_id, self.executable_number, self.queue_number]


class Workloads:

    def __init__(self, path):
        self.all_jobs = []
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_job_id = 0

        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_job_id = 0
        self.max_nodes = 0
        self.max_procs = 0

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    continue

                j = Job(line)
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.request_memory > self.max_requested_memory:
                    self.max_requested_memory = j.request_memory
                if j.user_id > self.max_user_id:
                    self.max_user_id = j.user_id
                if j.group_id > self.max_group_id:
                    self.max_group_id = j.group_id
                if j.executable_number > self.max_executable_number:
                    self.max_executable_number = j.executable_number

                # filter those illegal data whose runtime < 0
                if j.run_time < 0:
                    j.run_time = 10
                if j.run_time > 0:
                    self.all_jobs.append(j)

                    if j.request_number_of_processors > self.max:
                        self.max = j.request_number_of_processors

        # if max_procs = 0, it means node/proc are the same.
        if self.max_procs == 0:
            self.max_procs = self.max_nodes

        print("Max Allocated Processors:", str(self.max), "; max node:", self.max_nodes,
              "; max procs:", self.max_procs,
              "; max execution time:", self.max_exec_time)

        self.all_jobs.sort(key=lambda job: job.job_id)

    def size(self):
        return len(self.all_jobs)

    def reset(self):
        for job in self.all_jobs:
            job.scheduled_time = -1

    def __getitem__(self, item):
        return self.all_jobs[item]


class Machine:
    def __init__(self, id):
        self.id = id
        self.running_job_id = -1
        self.is_free = True
        self.job_history = []

    def taken_by_job(self, job_id):
        if self.is_free:
            self.running_job_id = job_id
            self.is_free = False
            self.job_history.append(job_id)
            return True
        else:
            return False

    def release(self):
        if self.is_free:
            return -1
        else:
            self.is_free = True
            self.running_job_id = -1
            return 1

    def reset(self):
        self.is_free = True
        self.running_job_id = -1
        self.job_history = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return "M["+str(self.id)+"] "


class Cluster:
    def __init__(self, cluster_name, node_num, num_procs_per_node):
        self.name = cluster_name
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []

        for i in range(self.total_node):
            self.all_nodes.append(Machine(i))

    def feature(self):
        return [self.free_node]

    def can_allocated(self, job):
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes > self.free_node:
            return False
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes <= self.free_node:
            return True

        request_node = int(math.ceil(
            float(job.request_number_of_processors)/float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True

    def allocate(self, job_id, request_num_procs):
        allocated_nodes = []
        request_node = int(
            math.ceil(float(request_num_procs) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = 0

        for m in self.all_nodes:
            if allocated == request_node:
                return allocated_nodes
            if m.taken_by_job(job_id):
                allocated += 1
                self.used_node += 1
                self.free_node -= 1
                allocated_nodes.append(m)

        if allocated == request_node:
            return allocated_nodes

        print("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)

        for m in releases:
            m.release()

    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node
        for m in self.all_nodes:
            m.reset()


# HPC Env
MAX_QUEUE_SIZE = 128
MLP_SIZE = 256

MAX_WAIT_TIME = 12 * 60 * 60  # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60  # assume maximal runtime is 12 hours

# each job has three features: wait_time, requested_node, runtime, machine states,
JOB_FEATURES = 8
DEBUG = False

JOB_SEQUENCE_SIZE = 256
SKIP_TIME = 360  # skip 60 seconds


class BatchSchedSim():
    def __init__(self, workload_file, seed):
        print("Initialize Batch Job Scheduler Simulator from dataset:", workload_file)

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = Workloads(workload_file)
        self.cluster = Cluster(
            "Cluster", self.loads.max_nodes, self.loads.max_procs/self.loads.max_nodes)

        self.bsld_algo_dict = {}
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.enable_preworkloads = False
        self.pre_workloads = []

        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = 0
    
        if seed <= 0:
            print(f"Seed must be a non-negative integer or omitted, not {seed}")
        
        seed_seq = np.random.SeedSequence(seed)
        np_seed = seed_seq.entropy
        self.np_random = np.random.Generator(np.random.PCG64(seed_seq))

        self.rlagent = RL4SysAgent()

    def f1_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        return (np.log10(request_time if request_time > 0 else 0.1) * request_processors + 870 * np.log10(submit_time if submit_time > 0 else 0.1))

    def f2_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))

    def f3_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f3: r * n + 6860000 * log10(s)
        return (request_time * request_processors + 6860000 * np.log10(submit_time))

    def f4_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f4: r * sqrt(n) + 530000 * log10(s)
        return (request_time * np.sqrt(request_processors) + 530000 * np.log10(submit_time))

    def sjf_score(self, job):
        # run_time = job.run_time
        request_time = job.request_time
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return (request_time, submit_time)

    def smallest_score(self, job):
        request_processors = job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return (request_processors, submit_time)

    def wfp_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time
        return -np.power(float(waiting_time)/request_time, 3)*request_processors

    def uni_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time

        return -(waiting_time+1e-15)/(np.log2(request_processors+1e-15)*request_time)

    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time

    # @profile
    def reset(self):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = self.start + self.loads.size()
        self.num_job_in_batch = self.loads.size()
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.pre_workloads = []

    def schedule_whole_trace(self):
        # start from the beginning of the trace
        self.start = 0
        self.start_idx_last_reset = self.start
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        # record the job scheduling sequences.
        self.scheduled_logs = {}
        self.rl_scheduled_jobs = []
        rl_working = False
        MAX_RL_RUNS = 128
        rl_runs = 0
        rew = 0

        # main scheduling loop
        while True:
            job_for_scheduling = None

            # added for enabling RL
            if rl_working or random.random() < 0.01:
                rl_working = True
                rl_runs += 1
                flatten_obs_old = self.build_observation()
                rl4sys_action = self.rlagent.request_for_action(flatten_obs_old, rew)
                assert rl4sys_action != None
                job_id_in_queue = rl4sys_action.act[0]
                job_for_scheduling = self.pairs[job_id_in_queue][0]

                self.rl_scheduled_jobs.append(job_for_scheduling)
                if rl_runs > MAX_RL_RUNS:
                    rl_runs = 0
                    rl_working = False
                    rl_total = self.calculate_scheduling_score(self.rl_scheduled_jobs)
                    rew = -rl_total
                    rl4sys_action.flag_last_action(rew)
                rew = 0

            else:
                # greedy scheduler
                self.job_queue.sort(key=lambda j: self.sjf_score(j))
                job_for_scheduling = self.job_queue[0]


            # ready to schedule "job_for_scheduling" job
            if not self.cluster.can_allocated(job_for_scheduling):
                
                earliest_start_time = self.current_timestamp

                if self.backfil:  # recalculate the earliest start time of this job
                    self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.request_time))
                    # calculate when will be the earliest start time of "job_for_scheduling"
                    avail_procs = self.cluster.free_node * self.cluster.num_procs_per_node
                    for running_job in self.running_jobs:
                        avail_procs += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
                        earliest_start_time = running_job.scheduled_time + running_job.request_time
                        if avail_procs >= job_for_scheduling.request_number_of_processors:
                            break
                
                while not self.cluster.can_allocated(job_for_scheduling):
                    if self.backfil: # backfilling as many jobs as possible in FCFS way based on the earliest-start-time
                        self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
                        job_queue_iter_copy = list(self.job_queue)
                        for _j in job_queue_iter_copy:
                            if (self.current_timestamp + _j.request_time) < earliest_start_time and self.cluster.can_allocated(_j):
                                # this job can fit based on time and resource needs
                                assert _j.scheduled_time == -1
                                _j.scheduled_time = self.current_timestamp
                                _j.allocated_machines = self.cluster.allocate(_j.job_id, _j.request_number_f_processors)
                                self.running_jobs.append(_j)
                                self.job_queue.remove(_j)
                        
                    # Move to the next timestamp
                    assert self.running_jobs

                    self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                    next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                    next_resource_release_machines = self.running_jobs[0].allocated_machines

                    if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                        self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                        self.job_queue.append(self.loads[self.next_arriving_job_idx])
                        self.next_arriving_job_idx += 1
                    else:
                        self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                        self.cluster.release(next_resource_release_machines)
                        self.running_jobs.pop(0)  # remove the first running job


            # this job should never be scheduled before.
            assert job_for_scheduling.scheduled_time == -1
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling.job_id,job_for_scheduling.request_number_of_processors)
            self.running_jobs.append(job_for_scheduling)
            self.job_queue.remove(job_for_scheduling)

            not_empty = self.moveforward_for_job()
            if not not_empty:
                break


    def calculate_scheduling_score(self, scheduled_jobs):
        sum = 0
        for job in scheduled_jobs:
            sum += self.job_score(job)
        avg = sum / len(scheduled_jobs)
        return avg


    def post_process_score(self, scheduled_logs):
        if self.job_score_type == 0:
            # bsld
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 1:
            # wait time
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 2:
            # turnaround time
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        elif self.job_score_type == 3:
            total_cpu_hour = (
                self.current_timestamp - self.loads[self.start].submit_time)*self.loads.max_procs
            for i in scheduled_logs:
                scheduled_logs[i] /= total_cpu_hour
        elif self.job_score_type == 4:
            for i in scheduled_logs:
                scheduled_logs[i] /= self.num_job_in_batch
        else:
            raise NotImplementedError


    def build_critic_observation(self):
        vector = np.zeros(JOB_SEQUENCE_SIZE * 3, dtype=float)
        earlist_job = self.loads[self.start_idx_last_reset]
        earlist_submit_time = earlist_job.submit_time
        pairs = []
        for i in range(self.start_idx_last_reset, self.last_job_in_batch+1):
            job = self.loads[i]
            submit_time = job.submit_time - earlist_submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time

            normalized_submit_time = min(
                float(submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
            normalized_run_time = min(
                float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
            normalized_request_nodes = min(
                float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

            pairs.append(
                [normalized_submit_time, normalized_run_time, normalized_request_nodes])

        for i in range(JOB_SEQUENCE_SIZE):
            vector[i*3:(i+1)*3] = pairs[i]

        return vector

    def build_observation(self):
        vector = np.zeros((MAX_QUEUE_SIZE) * JOB_FEATURES, dtype=float)
        self.job_queue.sort(key=lambda job: self.fcfs_score(job))
        self.visible_jobs = []
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.job_queue):
                self.visible_jobs.append(self.job_queue[i])
            else:
                break
        self.visible_jobs.sort(key=lambda j: self.fcfs_score(j))
        if self.shuffle:
            random.shuffle(self.visible_jobs)

        # @ddai: optimize the observable jobs
        self.visible_jobs = []
        if len(self.job_queue) <= MAX_QUEUE_SIZE:
            for i in range(0, len(self.job_queue)):
                self.visible_jobs.append(self.job_queue[i])
        else:
            visible_f1 = []
            f1_index = 0
            self.job_queue.sort(key=lambda job: self.f1_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f1.append(self.job_queue[i])

            visible_f2 = []
            f2_index = 0
            self.job_queue.sort(key=lambda job: self.f2_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_f2.append(self.job_queue[i])

            visible_sjf = []
            sjf_index = 0
            self.job_queue.sort(key=lambda job: self.sjf_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_sjf.append(self.job_queue[i])

            visible_small = []
            small_index = 0
            self.job_queue.sort(key=lambda job: self.smallest_score(job))
            for i in range(0, MAX_QUEUE_SIZE):
                visible_small.append(self.job_queue[i])

            visible_random = []
            random_index = 0
            shuffled = list(self.job_queue)
            random.shuffle(shuffled)
            for i in range(0, MAX_QUEUE_SIZE):
                visible_random.append(shuffled[i])

            index = 0

            while index < MAX_QUEUE_SIZE:
                f1_job = visible_f1[f1_index]
                f1_index += 1
                f2_job = visible_f2[f2_index]
                f2_index += 1
                sjf_job = visible_sjf[sjf_index]
                sjf_index += 1
                small_job = visible_small[small_index]
                small_index += 1
                random_job = visible_sjf[random_index]
                random_index += 1
                # if (not f1_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                #    self.visible_jobs.append(f1_job)
                #    index += 1
                # if (not f2_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                #    self.visible_jobs.append(f2_job)
                #    index += 1
                if (not sjf_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(sjf_job)
                    index += 1
                if (not small_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(small_job)
                    index += 1
                if (not random_job in self.visible_jobs) and index < MAX_QUEUE_SIZE:
                    self.visible_jobs.append(random_job)
                    index += 1

        '''
        @ddai: OPTIMIZE_OBSV. This time, we calculate the earliest start time of each job and expose that to the RL agent.
        if it is 0, then the job can start now, if it is near 1, that means it will have to wait for a really long time to start.
        The earliest start time is calculated based on current resources and the running jobs. It assumes no more jobs will be scheduled.

        # calculate the free resources at each outstanding ts
        free_processors_pair = []
        free_processors = (self.cluster.free_node * self.cluster.num_procs_per_node)
        free_processors_pair.append((free_processors, 0))

        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        for rj in self.running_jobs:
            free_processors += rj.request_number_of_processors
            free_processors_pair.append((free_processors, (rj.scheduled_time + rj.run_time - self.current_timestamp)))
        '''

        self.pairs = []
        add_skip = False
        for i in range(0, MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs) and i < (MAX_QUEUE_SIZE):
                job = self.visible_jobs[i]
                submit_time = job.submit_time
                request_processors = job.request_number_of_processors
                request_time = job.request_time
                # run_time = job.run_time
                wait_time = self.current_timestamp - submit_time

                # make sure that larger value is better.
                normalized_wait_time = min(
                    float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                normalized_run_time = min(
                    float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
                normalized_request_nodes = min(
                    float(request_processors) / float(self.loads.max_procs),  1.0 - 1e-5)

                '''
                @ddai: part 2 of OPTIMIZE_OBSV
                earliest_start_time = 1
                for fp, ts in free_processors_pair:
                    if request_processors < fp:
                        earliest_start_time = ts
                        break
                normalized_earliest_start_time = min(float(earliest_start_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
                '''

                # add extra parameters, include "Requested Memory", "User Id", "Groupd Id", "Exectuable Id", if its value does not exist in the trace (-1), we set it to 1 by default.
                if job.request_memory == -1:
                    normalized_request_memory = 1
                else:
                    normalized_request_memory = min(
                        float(job.request_memory)/float(self.loads.max_requested_memory), 1.0 - 1e-5)

                if job.user_id == -1:
                    normalized_user_id = 1
                else:
                    normalized_user_id = min(
                        float(job.user_id)/float(self.loads.max_user_id), 1.0-1e-5)

                if job.group_id == -1:
                    normalized_group_id = 1
                else:
                    normalized_group_id = min(
                        float(job.group_id)/float(self.loads.max_group_id), 1.0-1e-5)

                if job.executable_number == -1:
                    normalized_executable_id = 1
                else:
                    normalized_executable_id = min(
                        float(job.executable_number)/float(self.loads.max_executable_number), 1.0-1e-5)

                if self.cluster.can_allocated(job):
                    can_schedule_now = 1.0 - 1e-5
                else:
                    can_schedule_now = 1e-5
                self.pairs.append([job, normalized_wait_time, normalized_run_time, normalized_request_nodes,
                                  normalized_request_memory, normalized_user_id, normalized_group_id, normalized_executable_id, can_schedule_now])

            elif self.skip and not add_skip:  # the next job is skip
                add_skip = True
                if self.pivot_job:
                    self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1, 1])
                else:
                    self.pairs.append([None, 1, 1, 1, 1, 1, 1, 1, 0])
            else:
                self.pairs.append([None, 0, 1, 1, 1, 1, 1, 1, 0])

        for i in range(0, MAX_QUEUE_SIZE):
            vector[i*JOB_FEATURES:(i+1)*JOB_FEATURES] = self.pairs[i][1:]

        return vector

    # @profile
    def moveforward_for_job(self):
        if self.job_queue:
            return True

        # if we need to add job, but can not add any more, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            if not self.running_jobs:  # there are no running jobs
                # always add jobs if no resource can be released.
                next_resource_release_time = sys.maxsize
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (
                    running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (
                    self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(
                    self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True     # job added
            else:
                self.current_timestamp = max(
                    self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def job_score(self, job_for_scheduling):
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        if self.job_score_type == 0:
            # bsld
            _tmp = max(1.0, (float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                             /
                             max(job_for_scheduling.run_time, 10)))
        elif self.job_score_type == 1:
            # wait time
            _tmp = float(job_for_scheduling.scheduled_time -
                         job_for_scheduling.submit_time)
        elif self.job_score_type == 2:
            # turnaround time
            _tmp = float(job_for_scheduling.scheduled_time -
                         job_for_scheduling.submit_time + job_for_scheduling.run_time)
        elif self.job_score_type == 3:
            # utilization
            _tmp = -float(job_for_scheduling.run_time *
                          job_for_scheduling.request_number_of_processors)
        elif self.job_score_type == 4:
            # sld
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)\
                / job_for_scheduling.run_time
        else:
            raise NotImplementedError

            # Weight larger jobs.
        # _tmp = _tmp * (job_for_scheduling.run_time * job_for_scheduling.request_number_of_processors)
        return _tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str,
                        default='./data/lublin_256.swf')  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    sim = BatchSchedSim(workload_file=workload_file, seed=0)
    
    rl_training_server = TrainingServer()
    rl_training_server.start_loop()
        
    iter = 0
    # iterate multiple rounds to train the models
    while iter < 100:
        sim.reset()
        sim.schedule_whole_trace()
        iter += 1
