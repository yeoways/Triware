from .tf_rew import Simulator, get_op_costs


# 该类继承了tf_sim.py中的模拟器
class ImportantOpsRewarder(Simulator):

    def __init__(self, mg, op_perf, step_stats, devices):

        # 获取每一步骤的耗时
        cost_d, _ = get_op_costs(step_stats)

        out_d = {}
        for op in op_perf:
            out_d[op.node] = op.op_memory.output_memory  # 该层输出的内存大小

        for dev_stats in step_stats.dev_stats:
            for node_stats in dev_stats.node_stats:
                node = node_stats.node_name
                for output in node_stats.output:
                    allocation = output.tensor_description.allocation_description
                    num_bytes = allocation.requested_bytes
                    out_d[node] = [num_bytes]
                    break

        # 将底层模拟器进行初始化，输入包括原始的tf元数据，耗时，输出大小以及设备
        Simulator.__init__(self, mg, cost_d, out_d, devices)

    def simulate(self, pl, sim_mem_usage=False, sim_com_usage=False):

        for k, v in pl.items():
            pl[k] = self.devices[int(v)]

        r, f = Simulator.simulate(self, pl)

        self.f = f

        start_t = {}
        for node in self.metagraph.graph_def.node:
            n = node.name
            start_t[n] = f[n].start_time

        if sim_mem_usage:

            mem_q = []

            for n, t in start_t.items():

                mem = sum(self.output_dict[n])
                if mem == 0:
                    continue

                dev = self.devices.index(f[n].device)

                mem_q.append((t, '+', mem, dev))

                t_out_done = t
                for c in f[n].children:
                    t_out_done = max(t_out_done,
                                     int(f[c].start_time) + int(f[c].compute_cost) - 1)

                mem_q.append((t_out_done, '-', -mem, dev))

            mem_q.sort()

            mem_utils = [0] * len(self.devices)
            peak_utils = [0] * len(self.devices)

            for (t, _, mem, dev) in mem_q:
                mem_utils[dev] += mem

                if mem_utils[dev] > peak_utils[dev]:
                    peak_utils[dev] = mem_utils[dev]
            if sim_com_usage:
                comm_utils = [0] * len(self.devices)
                for k, v in self.parent_map.items():
                    device_c = pl[k]
                    for p in v.keys():
                        if device_c != pl[p]:
                            comm_utils[self.devices.index(device_c)] += sum(self.output_dict[p])
                return r/1e6, peak_utils, mem_utils, comm_utils
            # TODO
            # return r, peak_utils, mem_utils
            return r/1e6, peak_utils, mem_utils

        return r


