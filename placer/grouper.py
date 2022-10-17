# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Grouper module."""
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from collections import deque
import itertools
import operator
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
from placer import placer_utils as utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('grouper', 'aware', ['tf', 'coplace', 'aware'],
                         'Grouping algorithm')


class Grouper(object):
    """Default grouper that does nothing."""

    def __call__(self, op_graph):
        raise NotImplementedError()


def _len_and_str(string):
    """Returns a tuple of string length and string."""
    return (len(string), string)


def _update_colocation_group(op_graph, ungrouped_mapping, group_to_group, colocation_group_map):
    """Updates colocation groups of operators in op_graph with new mapping."""
    # pick the shortest group name among groups as new group name
    group_dict = {group: min(group_set, key=_len_and_str)
                  for group, group_set in colocation_group_map.items()}

    # print merged groups
    reverse_mapping = {}
    for prev_name, new_name in group_dict.items():
        if new_name in reverse_mapping:
            reverse_mapping[new_name].append(prev_name)
        else:
            reverse_mapping[new_name] = [prev_name]
    for new_name, prev_names in reverse_mapping.items():
        _LOGGER.debug('Change group: %s -> %s', sorted(prev_names), new_name)

    # update colocation group
    for op_name, op_data in op_graph.nodes.items():
        if isinstance(op_data['colocation_group'], list):
            new_group = None
            for colocation_group in op_data['colocation_group']:
                ret = group_dict.get(colocation_group, colocation_group)
                if new_group is None:
                    new_group = ret
                else:
                    assert new_group == ret, 'node=%s, cur=%s, new=%s' % (
                        op_data['name'], new_group, ret)
        else:
            prev_group_name = op_data['colocation_group']
            new_group = group_dict.get(prev_group_name, prev_group_name)

        if isinstance(op_data['colocation_group'], list):
            old_key = op_data['colocation_group'][0]
        else:
            old_key = op_data['colocation_group']
        op_data['colocation_group'] = new_group

        change_keys = []
        for key, value in group_to_group.items():
            if value == group_to_group[old_key]:
                change_keys.append(key)
        for key in change_keys:
            group_to_group[key] = new_group
        ungrouped_mapping[op_name] = new_group

    func = lambda z: dict([(x, y) for y, x in z.items()])
    text1 = func(func(ungrouped_mapping)).values()
    text2 = func(func(group_to_group)).values()
    print("目前预测节点数量：" + str(len(text1)) + "\t" + str(len(text2)))


def process_colocation_group(op_graph):
    """Process a list of colocations groups into a single colocation group."""

    # This maps a colocation group name to a set of other group names
    colocation_group_map = utils.ColocationGroupMap()

    for _, op_data in op_graph.nodes.items():
        colocation_group = op_data['colocation_group']
        for op1, op2 in itertools.combinations(colocation_group, 2):
            colocation_group_map.colocate(op1, op2)

    _update_colocation_group(op_graph, colocation_group_map)


def aware_process_colocation_group(op_graph, ungrouped_mapping, group_to_group):
    """Process a list of colocations groups into a single colocation group."""

    # This maps a colocation group name to a set of other group names
    colocation_group_map = utils.AwareColocationGroupMap()

    for _, op_data in op_graph.nodes.items():
        colocation_group = op_data['colocation_group']
        for op1, op2 in itertools.combinations(colocation_group, 2):
            colocation_group_map.colocate(op1, op2)

    _update_colocation_group(op_graph, ungrouped_mapping, group_to_group, colocation_group_map)

class TFColocationGrouper(Grouper):
    """Generate a new graph by using TensorFlow colocation group information."""

    def __call__(self, op_graph):
        # use the existing colocation group information
        process_colocation_group(op_graph)
        return op_graph


class CoplacementGrouper(Grouper):
    """Generate a new graph by using heuristic at the paper."""

    def __init__(self, log_colocation_group=False, ignore_control_edges=False):
        super(CoplacementGrouper, self).__init__()
        self._log_colocation_group = log_colocation_group
        self._ignore_control_edges = ignore_control_edges

    @staticmethod
    def _run_colocation_step(op_graph, ignore_control_edges):
        """Check whether there are operators that can be co-located.

        When the output of an operator is consumed only by another operator,
        assign the same colocation group for them

        Returns:
          True if there are opeartors that can be co-located.
          False, otherwise.
        """
        colocation_candidates = utils.ColocationGroupMap()

        for op_id, op_data in op_graph.nodes.items():
            # TODO: should we consider tensor-wise? not operator wise?
            out_edges = list(op_graph.out_edges(op_id))
            if len(out_edges) != 1:
                continue

            next_op_id = out_edges[0][1]

            # pass control edges because this does not have data transfer
            edge_data = op_graph.get_edge_data(op_id, next_op_id)
            if ignore_control_edges and edge_data["is_control"]:
                continue

            next_op_data = op_graph.nodes[next_op_id]

            op_group = op_data['colocation_group']
            next_op_group = next_op_data['colocation_group']

            if op_group != next_op_group:
                # these two can be colocated
                _LOGGER.debug('Possible colocation ops. %s[%s] -> %s[%s]',
                              op_data['name'],
                              op_group,
                              next_op_data['name'],
                              next_op_group)
                colocation_candidates.colocate(op_group, next_op_group)

        if len(colocation_candidates) > 0:
            _update_colocation_group(op_graph, colocation_candidates)
            return True

        return False

    def __call__(self, op_graph):
        process_colocation_group(op_graph)
        # first use default colocation group information
        if self._log_colocation_group:
            with open('tf_colocation_groups.log', 'w') as f:
                utils.print_colocation_group(
                    op_graph, print_cb=lambda v: f.write(v + '\n'))
        while self._run_colocation_step(op_graph, self._ignore_control_edges):
            pass
        if self._log_colocation_group:
            with open('coplaced_groups.log', 'w') as f:
                utils.print_colocation_group(
                    op_graph, print_cb=lambda v: f.write(v + '\n'))
        return op_graph


class CoplacementGrouperAware(Grouper):
    """Generate a new graph by using heuristic at the paper."""

    def __init__(self, log_colocation_group=False, ignore_control_edges=False):
        super(CoplacementGrouperAware, self).__init__()
        self._log_colocation_group = log_colocation_group
        self._ignore_control_edges = ignore_control_edges

    @staticmethod
    def _generate_fused_op_graph(
            op_graph, fusion_check_disjoint_paths, ungrouped_mapping, allow_cycle=False):
        """Generates a fused op graph.

        This first identifies ops that can be fused.
        When ops are in the same colocation group and they are directly
        connected, we fuse two ops into a single op.
        """

        def _assign_new_ids(fused_op_graph, op_name_old, op_name_new):
            """Returns a new graph that of which nodes have unique sequential ids.
            """
            new_fused_op_graph = nx.DiGraph()
            fused_id_map = {}  # maps ids in op_graph to new ids in fused_op_graph
            num_fused_ops = 0

            for new_id, (old_id, data) in enumerate(fused_op_graph.nodes.items()):
                # update id information
                data["old_name"] = old_id
                if old_id == op_name_old:
                    new_id = op_name_new
                data["name"] = new_id
                new_fused_op_graph.add_node(new_id, **data)
                fused_id_map[old_id] = new_id

                # log fused op information
                if "aggregated_nodes" in data:
                    num_fused_ops += len(data["aggregated_nodes"])
                    fused_ops_list = [
                        "%s" % (fused_op_data["name"])
                        for fused_op_data in data["aggregated_nodes"]]

            for new_id, (u, v, data) in enumerate(fused_op_graph.edges(data=True)):
                data["old_name"] = data["name"]
                data["name"] = new_id
                new_fused_op_graph.add_edge(
                    fused_id_map[u], fused_id_map[v], **data)

            return new_fused_op_graph

        def _add_fused_edge(fused_op_graph, from_op_id, to_op_id, edge_data):
            """Adds an edge to the fused op graph.
            Returns:
                True if a new edge is added. False, otherwise.
            """
            if fused_op_graph.has_edge(from_op_id, to_op_id):
                # update existing edge
                prev_edge = fused_op_graph[from_op_id][to_op_id]
                prev_edge_tensors = prev_edge['tensor']
                tensors_to_add = [
                    tensor_data_ for tensor_data_ in edge_data['tensor']
                    if tensor_data_ not in prev_edge_tensors]
                for tensor_data_ in tensors_to_add:
                    # prev_edge['weight'] += tensor_data_['weight']
                    prev_edge['tensor'].append(tensor_data_)
                return False

            fused_op_graph.add_edge(from_op_id, to_op_id, **edge_data)
            return True

        # pylint: disable=too-many-locals,too-many-branches
        def _create_colocation_group_to_ops_map(op_graph):
            """Generate a dict that maps a colocation group to its op id list."""
            retval = {}

            for op_id, op_data in op_graph.nodes().items():
                # assume there is only one group
                group = op_data['colocation_group']
                if group in retval:
                    retval[group].append(op_id)
                else:
                    retval[group] = [op_id]

            return retval

        print("Allow cycle in operator fusion: %s", str(allow_cycle))

        fused_op_graph = copy.deepcopy(op_graph)

        group_to_ops = _create_colocation_group_to_ops_map(
            fused_op_graph)

        for ops in group_to_ops.values():
            internal_edges = deque(
                [(u, v) for u, v in fused_op_graph.edges(ops)
                 if u in ops and v in ops])

            while len(internal_edges) > 0:
                op1, op2 = internal_edges.popleft()

                # internal edge might be connected to a fused op
                if not fused_op_graph.has_edge(op1, op2):
                    continue

                # check whether there is another path from op1 to op2
                if not allow_cycle and (fused_op_graph.out_degree(op1) > 1
                                        and fused_op_graph.in_degree(op2) > 1):
                    if fusion_check_disjoint_paths:
                        # CAVEATS: finding disjoint paths may take long time
                        paths = list(
                            nx.node_disjoint_paths(fused_op_graph, op1, op2))
                        if len(paths) > 1:
                            # ops cannot be fused since it will create a cycle
                            continue
                    else:
                        # if _add_test(op1, op2, op_graph):
                        # skip this fusion due to potential cycle generation
                        continue

                # fuse op2 into op1
                op1_data = fused_op_graph.nodes[op1]
                op2_data = fused_op_graph.nodes[op2]

                print("%s[%d] is fused into %s[%d]",
                      op2_data["name"], op2, op1_data["name"], op1)

                op1_data["cost"] += op2_data["cost"]
                op1_data["mem"] += op2_data["mem"]
                # use max since each op runs at a time, not simultaneously
                # op1_data["temporary_memory"] = max(
                #     op1_data["temporary_memory"], op2_data["temporary_memory"])

                # add op2's edges to op1
                for in_edge in fused_op_graph.in_edges(op2, data=True):
                    from_op, _, edge_data = in_edge
                    if from_op == op1:
                        continue
                    if _add_fused_edge(
                            fused_op_graph, from_op, op1, edge_data):
                        # if the new edge is a candidate,
                        # check it by adding it to internal_edges
                        if from_op in ops:
                            internal_edges.append((from_op, op1))

                for out_edge in fused_op_graph.out_edges(op2, data=True):
                    _, to_op, edge_data = out_edge
                    if to_op == op1:
                        continue
                    if _add_fused_edge(
                            fused_op_graph, op1, to_op, edge_data):
                        # if the new edge is a candidate,
                        # check it by adding it to internal_edges
                        if to_op in ops:
                            internal_edges.append((op1, to_op))

                # op2 might be a fused op. merge information
                new_fused_ops = op1_data.get("aggregated_nodes", [])
                new_fused_ops += op2_data.pop("aggregated_nodes", [])
                new_fused_ops += [op2_data]
                op1_data["aggregated_nodes"] = new_fused_ops

                fused_op_graph.remove_node(op2)

                def get_keys(d, value):
                    return [k for k, v in d.items() if v == value]

                for key in get_keys(ungrouped_mapping, op2):
                    ungrouped_mapping[key] = op1
                ops.remove(op2)
                # update output memory
                # CAVEATS: output port number is no longer valid.
                output_tensors = {}
                for out_edge in fused_op_graph.out_edges(op1, data=True):
                    out_edge_data = out_edge[-1]
                    for tensor_data in out_edge_data['tensor']:
                        output_tensors[tensor_data['name']] = \
                            tensor_data['num_bytes']
                op1_data['out_memory'] = list(output_tensors.values())
                op1_data['out_size'] = sum(list(output_tensors.values()))

        # need to assign new node ids and edge ids to be compatible with m_sct.
        # m_sct expects op_graph that has consecutive node ids and edge ids.
        return fused_op_graph

    @staticmethod
    def _create_colocation_graph(op_graph):
        def _should_ignore(op1_node, op2_node):
            # Not adding an edge that connects to the optimizer through
            # control dependency to break the cycle.
            # TODO: Can the name rule "control_dependency" -> "Apply*"
            # be applied to any graph?
            op1_name = op1_node["name"]
            op2_name = op2_node["name"]
            if "control_dependency" in op1_name and "Apply" in op2_name:
                # Ignore this edge to break the cycle.
                _LOGGER.info("Ignore an edge from %s[%s] to %s[%s].",
                             op1_name,
                             op1_node["colocation_group"],
                             op2_name,
                             op2_node["colocation_group"])
                return True
            else:
                return False

        new_graph = nx.DiGraph()
        new_index = {}

        for op_name in op_graph:
            op_node = op_graph.nodes[op_name]
            colocation_group = op_node["colocation_group"]
            group_name = str(colocation_group).split("@")[1]
            if colocation_group in new_index:
                target_node = new_graph.nodes[new_index[colocation_group]]
                # Update the existing node
                # sum up the computation time by assuming that a device runs
                # operators sequentially.
                # op_graph.add_node(str(tf_op.name),
                #                   name=str(tf_op.name),
                #                   cost=duration,
                #                   weight=duration,
                #                   mem=sum(output_memory),
                #                   colocation_group=get_raw_colocation_group(tf_op),
                #                   device=str(tf_op.device),
                #                   out_size=sum(output_memory),
                #                   aggregated_nodes=[str(tf_op.name)],
                #                   output_memory=output_memory)
                target_node["name"] = group_name
                target_node["cost"] += op_node["cost"]
                target_node["mem"] += op_node["mem"]
                target_node["out_size"] += op_node["out_size"]
                target_node["aggregated_nodes"].append(op_name)
                # 需要修改（先废除）
                target_node["output_memory"] += op_node["output_memory"]
                _LOGGER.debug("Node updated. %s", str(target_node))
            else:
                new_graph.add_node(group_name,
                                   name=group_name,
                                   cost=op_node["cost"],
                                   mem=op_node["mem"],
                                   colocation_group=colocation_group,
                                   device=op_node["device"],
                                   out_size=op_node["out_size"],
                                   aggregated_nodes=op_node["aggregated_nodes"],
                                   contraction=op_node["contraction"],
                                   output_memory=op_node["output_memory"])
                new_index[colocation_group] = group_name
        # add edges
        for op1_name, op2_name in op_graph.edges:
            edge = op_graph[op1_name][op2_name]
            op1_node = op_graph.nodes[op1_name]
            op2_node = op_graph.nodes[op2_name]
            group1 = op1_node["colocation_group"]
            group2 = op2_node["colocation_group"]
            group1_name = new_index[group1]
            group2_name = new_index[group2]
            if group1_name == group2_name:
                continue

            if new_graph.has_edge(group1_name, group2_name):
                target_edge = new_graph[group1_name][group2_name]
                target_edge["num_bytes"] += edge["num_bytes"]
            else:
                if not FLAGS.consider_all_edges_in_grouping:
                    if _should_ignore(op1_node, op2_node, ):
                        continue
                new_graph.add_edge(group1_name, group2_name, **edge)

        return new_graph, new_index

    def neighbor_merge_pruner(self, G, ungrouped_mapping, final_size):
        def generate_edge_table(G):
            table = []
            for edge in G.edges.data():
                p_node = edge[0]
                c_node = edge[1]
                edge_id = edge[2]['id']
                num_bytes = edge[2]['num_bytes']
                table.append({'id': edge_id, 'num_bytes': num_bytes, 'p_node': p_node, 'c_node': c_node})
            return sorted(table, key=operator.itemgetter('num_bytes'), reverse=True)

        edge = generate_edge_table(G)[0]
        i = 0
        while True:
            if self._run_one_step_merge(G, True, ungrouped_mapping, edge['p_node'], edge['c_node'], False):
                edge = generate_edge_table(G)[i]
                i = 0
            else:
                i = i + 1
                edge = generate_edge_table(G)[i]
            if len(G.nodes()) <= final_size:
                break

    @staticmethod
    def _run_one_step_merge(
            op_graph, fusion_check_disjoint_paths, ungrouped_mapping, op1, op2, allow_cycle=False):
        def _add_test(op1, op2, op_graph_):
            op_graph = copy.deepcopy(op_graph_)
            pre = set(op_graph.predecessors(op1)) | set(op_graph.predecessors(op2))
            if op1 in pre:
                pre.remove(op1)
            if op2 in pre:
                pre.remove(op2)
            suc = set(op_graph.successors(op1)) | set(op_graph.successors(op2))
            if op1 in suc:
                suc.remove(op1)
            if op2 in suc:
                suc.remove(op2)
            op_graph.add_node("test")
            op_graph.add_edges_from([(p, "test") for p in pre])
            op_graph.add_edges_from([("test", s) for s in suc])
            op_graph.remove_nodes_from([op1, op2])
            flag = nx.is_directed_acyclic_graph(op_graph)
            op_graph.remove_nodes_from(["test"])
            return flag

        def _add_fused_edge(fused_op_graph, from_op_id, to_op_id, edge_data):
            if fused_op_graph.has_edge(from_op_id, to_op_id):
                # update existing edge
                prev_edge = fused_op_graph[from_op_id][to_op_id]
                prev_edge['num_bytes'] = prev_edge['num_bytes'] + edge_data['num_bytes']
                return False
            else:
                fused_op_graph.add_edge(from_op_id, to_op_id, **edge_data)
                return True

        assert op_graph.has_edge(op1, op2)
        op1_group = op_graph.nodes[op1]['colocation_group']
        op2_group = op_graph.nodes[op2]['colocation_group']
        assert op1_group != op2_group

        if not allow_cycle and (op_graph.out_degree(op1) > 1
                                and op_graph.in_degree(op2) > 1):
            if fusion_check_disjoint_paths:
                # CAVEATS: finding disjoint paths may take long time
                paths = list(
                    nx.node_disjoint_paths(op_graph, op1, op2))
                if len(paths) > 1:
                    # ops cannot be fused since it will create a cycle
                    print("ops cannot be fused since it will create a cycle")
                    return False
            else:
                return False

        # fuse op2 into op1
        op1_data = op_graph.nodes[op1]
        op2_data = op_graph.nodes[op2]

        print("%s[%d] is fused into %s[%d]",
              op2_data["name"], op2, op1_data["name"], op1)

        op1_data["cost"] += op2_data["cost"]
        op1_data["mem"] += op2_data["mem"]
        op1_data["out_size"] += op2_data["out_size"]
        op1_data["output_memory"].append(op2_data["output_memory"])

        # add op2's edges to op1
        for in_edge in op_graph.in_edges(op2, data=True):
            from_op, _, edge_data = in_edge
            if from_op == op1:
                continue
            _add_fused_edge(
                op_graph, from_op, op1, edge_data)

        for out_edge in op_graph.out_edges(op2, data=True):
            _, to_op, edge_data = out_edge
            if to_op == op1:
                continue
            _add_fused_edge(
                op_graph, op1, to_op, edge_data)

        # op2 might be a fused op. merge information
        new_fused_ops = op1_data.get("aggregated_nodes", [])
        new_fused_ops += op2_data.pop("aggregated_nodes", [])
        new_fused_ops += [op2_data]
        op1_data["aggregated_nodes"] = new_fused_ops

        op_graph.remove_node(op2)

        def get_keys(d, value):
            return [k for k, v in d.items() if v == value]

        for key in get_keys(ungrouped_mapping, op2_group):
            ungrouped_mapping[key] = op1_group
        # update output memory
        # CAVEATS: output port number is no longer valid.
        # output_tensors = {}
        op1_output_memory = []
        for out_edge in op_graph.out_edges(op1, data=True):
            op1_output_memory.append(out_edge[-1]['num_bytes'])
        op1_data['output_memory'] = op1_output_memory
        func = lambda z: dict([(x, y) for y, x in z.items()])
        text = func(func(ungrouped_mapping)).values()
        print("节点归并进程：" + str(len(text)) + "\t" + str(len(op_graph.nodes)))
        return True

    def _find_cycle_and_save_figures(self, colocation_op_graph):
        """Finds a cycle in the colocation op graph.

        If a cycle exists, save the cycle in the graph and
        also corresponding ops as figures. Then, raise ValueError.
        Otherwise, just return.
        """
        try:
            nodes_in_cycle = [u for u, _ in nx.find_cycle(colocation_op_graph)]
            _LOGGER.info("Cycle: %s",
                         str([colocation_op_graph.nodes[u]["name"]
                              for u in nodes_in_cycle]))

            if FLAGS.resolve_cycle:
                op1_name = nodes_in_cycle[0]
                op2_name = nodes_in_cycle[1]
                _LOGGER.info(
                    "Removing %s -> %s to remove a cycle",
                    colocation_op_graph.nodes[op1_name]["name"],
                    colocation_op_graph.nodes[op2_name]["name"])
                colocation_op_graph.remove_edge(op1_name, op2_name)
                self._find_cycle_and_save_figures(colocation_op_graph)
            else:

                op_names_in_cycle = set()
                for colocation_op_name in nodes_in_cycle:
                    colocation_op = colocation_op_graph.nodes[colocation_op_name]
                    for op_name in colocation_op["op_names"]:
                        op_names_in_cycle.add(op_name)
                raise ValueError("Cycle exists in the placement graph")

        except nx.NetworkXNoCycle:
            _LOGGER.info("No cycle exists")

    @staticmethod
    def _run_colocation_step(op_graph, ungrouped_mapping, group_to_group, ignore_control_edges):
        colocation_candidates = utils.ColocationGroupMap()
        func = lambda z: dict([(x, y) for y, x in z.items()])
        text = func(func(ungrouped_mapping)).values()
        i = len(text)
        for op_id, op_data in op_graph.nodes.items():
            # TODO: should we consider tensor-wise? not operator wise?
            out_edges = list(op_graph.out_edges(op_id))
            if len(out_edges) != 1:
                continue

            next_op_id = out_edges[0][1]

            # pass control edges because this does not have data transfer
            edge_data = op_graph.get_edge_data(op_id, next_op_id)
            if ignore_control_edges and edge_data["is_control"]:
                continue

            next_op_data = op_graph.nodes[next_op_id]

            op_group = op_data['colocation_group']
            next_op_group = next_op_data['colocation_group']
            # and (op_data['out_size'] < 1000 or next_op_data['out_size'] < 1000)
            if op_group != next_op_group:
                # these two can be colocated
                _LOGGER.debug('Possible colocation ops. %s[%s] -> %s[%s]',
                              op_data['name'],
                              op_group,
                              next_op_data['name'],
                              next_op_group)
                colocation_candidates.colocate(op_group, next_op_group)
                i = i - 1
        if len(colocation_candidates) > 0:
            _update_colocation_group(op_graph, ungrouped_mapping, group_to_group, colocation_candidates)
            return True

        return False

    @staticmethod
    def _run_merge_nodes(op_graph, ungrouped_mapping, group_cost, ignore_control_edges):
        def get_keys(d, value):
            return [k for k, v in d.items() if v == value]
        colocation_candidates = utils.ColocationGroupMap()
        out_cost = group_cost[0]
        key_min = min(out_cost.keys(), key=(lambda k: out_cost[k]))
        candidates = get_keys(ungrouped_mapping, key_min)

        func = lambda z: dict([(x, y) for y, x in z.items()])
        text = func(func(ungrouped_mapping)).values()

        i = 0
        for op_id, op_data in op_graph.nodes.items():
            out_edges = list(op_graph.out_edges(op_id))
            if out_edges and out_cost[op_data['colocation_group']] < 100 and len(text) - i > 200:
                next_op_id = out_edges[0][1]
            else:
                continue
            # pass control edges because this does not have data transfer
            edge_data = op_graph.get_edge_data(op_id, next_op_id)
            if ignore_control_edges and edge_data["is_control"]:
                continue

            next_op_data = op_graph.nodes[next_op_id]

            op_group = op_data['colocation_group']
            next_op_group = next_op_data['colocation_group']

            if op_group != next_op_group:
                # these two can be colocated
                _LOGGER.debug('Possible colocation ops. %s[%s] -> %s[%s]',
                              op_data['name'],
                              op_group,
                              next_op_data['name'],
                              next_op_group)
                colocation_candidates.colocate(op_group, next_op_group)
                out_cost[op_group] = out_cost[next_op_group] = out_cost[op_group] + out_cost[next_op_group]
                i = i + 1

        if len(colocation_candidates) > 0:
            _update_colocation_group(op_graph, ungrouped_mapping, colocation_candidates)
            return True

        return False

    @staticmethod
    def _create_group_cost(op_graph):
        group_out_cost = {}
        group_com_cost = {}
        for op_name in op_graph:
            op_node = op_graph.nodes[op_name]
            colocation_group = op_node["colocation_group"]
            if colocation_group not in group_out_cost:
                group_out_cost[colocation_group] = op_node['out_size']
                group_com_cost[colocation_group] = op_node['cost']
            else:
                group_out_cost[colocation_group] += op_node['out_size']
                group_com_cost[colocation_group] += op_node['cost']

        return [group_out_cost, group_com_cost]

    @staticmethod
    def _debug_graph(G):
        Gg = G
        pos = nx.spring_layout(Gg)  # 螺旋状的图片
        nodesToPrint = Gg.nodes
        nc = nx.draw_networkx_nodes(Gg, pos,
                                    nodelist=nodesToPrint,
                                    with_labels=True,
                                    node_size=2,
                                    cmap='plasma')
        # TODO remove for remote play 这句话报错，所以先对其注释
        plt.axis('off')
        nx.draw_networkx_edges(Gg, pos, width=1.0, alpha=0.2)
        plt.show()
        plt.clf()

    def __call__(self, op_graph, ungrouped_mapping, group_to_group):
        aware_process_colocation_group(op_graph, ungrouped_mapping, group_to_group)
        # if self._log_colocation_group:
        #     with open('tf_colocation_groups.log', 'w') as f:
        #         utils.print_colocation_group(
        #             op_graph, print_cb=lambda v: f.write(v + '\n'))
        # while self._run_colocation_step(op_graph, ungrouped_mapping, group_to_group, self._ignore_control_edges):
        #     pass
        # if self._log_colocation_group:
        #     with open('coplaced_groups.log', 'w') as f:
        #         utils.print_colocation_group(
        #             op_graph, print_cb=lambda v: f.write(v + '\n'))
        new_graph, new_index = self._create_colocation_graph(op_graph)

        # debug
        self.neighbor_merge_pruner(new_graph, ungrouped_mapping, 200)

        self._debug_graph(new_graph)

        func = lambda z: dict([(x, y) for y, x in z.items()])
        text1 = func(func(ungrouped_mapping)).values()
        text2 = func(func(group_to_group)).values()
        print("目前预测节点数量：" + str(len(text1)) + "\t" + str(len(text2)))
        kong_list = []
        target_group = ''
        for key, value in ungrouped_mapping.items():
            if value == '':
                kong_list.append(key)
            else:
                target_group = value
        for kong in kong_list:
            ungrouped_mapping[kong] = target_group
        for key, value in group_to_group.items():
            if value == '':
                group_to_group[key] = target_group
        return new_graph


_GROUPER_CLASS_MAP = {
    'none': Grouper,
    'tf': TFColocationGrouper,
    'coplace': CoplacementGrouper,
    "aware": CoplacementGrouperAware
}


def get_grouper(grouper=None):
    """Generates and returns a grouper instance."""
    grouper = grouper or FLAGS.grouper
    _LOGGER.info('Grouper: %s', grouper)
    grouper_class = _GROUPER_CLASS_MAP[grouper]
    return grouper_class()
