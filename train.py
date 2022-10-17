from __future__ import absolute_import, division, print_function
import sys
import collections
import os
import pickle
from tensorflow.python.framework import errors
import time
import codecs
import json
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from tensorflow.python.platform import tf_logging
import numpy as np
import tensorflow as tf
from future.utils import bytes_to_native_str
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem
from tensorflow.core.protobuf import meta_graph_pb2
from cv.networks import nets_factory
from nlp import model_factory
from placer import placer_lib, cost as cost_lib
from third_party.ColorRL import graph_placer as grappler_graph_placer
from utils import logger
from sim.tf_sim.tf_pl_rewarder import ImportantOpsRewarder
import random
# row_graph
tf.app.flags.DEFINE_boolean(
    'log_device_placement', False, 'Logging device placement.')

tf.app.flags.DEFINE_string(
    'eval_graph_dest', './cifarnet.pkl', 'Logging device placement.')

tf.app.flags.DEFINE_string(
    'dest_graph_output', './cifarnet.pkl', 'Logging device placement.')

tf.app.flags.DEFINE_boolean(
    'colocate_grads_with_ops', True, 'Colocate gradient with ops.')

tf.app.flags.DEFINE_enum(
    'optimizer', 'adam',
    ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'sgd', 'rmsprop'],
    'The name of the optimizer')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v1_152', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
# 初始值 32

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_string(
    'logdir', '', 'Path to log dir.')

tf.app.flags.DEFINE_string(
    'cost_path', '/tmp/cost.pkl', 'Path to the cost file.')

tf.app.flags.DEFINE_string(
    'place_path', './tmp/best_place.json', 'Path to the place file.')

tf.app.flags.DEFINE_boolean(
    'costgen', False, 'Generate cost dict.')

tf.app.flags.DEFINE_boolean(
    'only_forward', True, 'Consider only forward ops.')

tf.app.flags.DEFINE_float('memory_fraction', 1.0, 'GPU memory fraction')

tf.app.flags.DEFINE_string(
    'comm_cost_coeffs', '0.0001754,134',
    'Comma-separated linear communication cost function coefficients')

tf.app.flags.DEFINE_float(
    'comm_cost_factor', 1.0, 'Communication cost function factor.')

tf.app.flags.DEFINE_float(
    'cost_factor', 1.0, 'Factor that applies to all costs')

###### Image classifier ######
tf.app.flags.DEFINE_enum(
    'data_format', 'NHWC', ['NHWC', 'NCHW'], 'Image data format')

tf.app.flags.DEFINE_enum(
    'exe_pattern', 'aware_forward', ['aware_forward', 'eval', 'ColorRL', 'aware_back'], 'Image data format')

##### NMT ######
tf.app.flags.DEFINE_integer('vocab_size', 5000, 'Vocabulary size.')
tf.app.flags.DEFINE_integer('max_seq_length', 30, 'Max. sequence length.')
tf.app.flags.DEFINE_integer('rnn_units', 1024, 'RNN units.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'RNN # layers.')
tf.app.flags.DEFINE_enum(
    'rnn_unit_type', 'lstm', ['lstm', 'gru'], 'RNN unit type.')
tf.app.flags.DEFINE_enum(
    'encoder_type', 'bi', ['bi', 'uni', 'gnmt'], 'Encoder type.')
tf.app.flags.DEFINE_boolean(
    'residual', False, 'Add residual connections to RNN.')
tf.app.flags.DEFINE_integer('num_gpus', 80, 'Number of gpus for NMT.')
tf.app.flags.DEFINE_boolean('disable_nmt_colocation', False,
                            'Disable the NMT ops colocation.')

##### Grappler ######
tf.app.flags.DEFINE_boolean('ColorRL', False, 'Use Grappler.')
tf.app.flags.DEFINE_integer(
    'grappler_time', 3600, 'Allotted time in seconds for Grappler.')

_LOGGER = logger.get_logger(__file__)


def _configure_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, name='Momentum')
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError(
            'Optimizer [%s] was not recognized' % optimizer_name)
    return optimizer


def _get_gpu_devices(sess_config):
    with tf.Session(config=sess_config) as sess:
        return [
            {"name": device.name,
             "memory_size": device.memory_limit_bytes,
             "type": device.device_type}
            for device in sess.list_devices()
            if device.device_type == 'GPU']

_CLASSIFICATION_NUM = {
    'cifarnet': 10,
    'inception_v3': 10,
    'resnet_v1_50': 10,
    'resnet_v1_152': 10,
    'resnet_v1_101': 10,
    'resnet_v1_200': 10,
    'nasnet': 10,
    'nasnet_cifar': 10,
    'vgg_19': 10,
}

ModelSpec = collections.namedtuple('ModelSpec', ['cls', 'image_size'])


def build_cv_model(inputs, model_name, data_format):
    _LOGGER.info('数据编码: %s', data_format)

    images, labels = inputs

    num_classes = _CLASSIFICATION_NUM[model_name]
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)
    logits, _ = network_fn(images)
    with tf.variable_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_sum(losses) / tf.to_float(images.shape[0])
    return loss


def build_nlp_networks(inputs, model_name, **kwargs):
    """Builds NMT with the given specs."""
    # pylint: disable=too-many-locals
    # log NMT spec.
    _LOGGER.info(', '.join(['{}={}'.format(*item) for item in kwargs.items()]))

    src_input, target_input, target_output = inputs

    vocab_size = kwargs.pop('vocab_size')

    # replicate vocab size
    kwargs['src_vocab_size'] = vocab_size
    kwargs['tgt_vocab_size'] = vocab_size

    model_fn = model_factory.get_model_fn(model_name, **kwargs)
    _, loss = model_fn(src_input, target_input, target_output)

    return loss


def build_deep_learning_network(inputs, model_name, data_format, **kwargs):
    """Builds a model with the given specs."""
    if model_name in _CLASSIFICATION_NUM:
        return build_cv_model(inputs, model_name, data_format)

    return build_nlp_networks(inputs, model_name, **kwargs)


# 在真实分布式环境下执行一次计算图所需要的时间
def measure_graph(target_op, warmup_count=5, num_measurement=10,
                  profile_every_n_steps=None, logdir=None, config=None):
    with tf.Session(config=config) as sess:
        if logdir:
            writer = tf.summary.FileWriter(logdir=logdir,
                                           graph=tf.get_default_graph())
        else:
            writer = None

        sess.run(tf.global_variables_initializer())

        warmup_start_time = time.time()

        for _ in range(warmup_count):
            sess.run(target_op)

        warmup_end_time = time.time()
        _LOGGER.info('准备时间: %s',
                     str(warmup_end_time - warmup_start_time))
        runtimes = []
        run_metadata_list = []
        for step in range(1, num_measurement + 1):
            if profile_every_n_steps and step % profile_every_n_steps == 0:
                _LOGGER.info('执行进度 %d...', step)
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                sess.run(target_op,
                         options=run_options,
                         run_metadata=run_metadata)
                if writer:
                    writer.add_run_metadata(
                        run_metadata, 'step-{}'.format(step))
                    # pylint: disable=invalid-name
                    metadata_out_path = os.path.join(
                        logdir, 'run_metadata-{}.pbtxt'.format(step))
                    with open(metadata_out_path, 'wb') as f:
                        f.write(run_metadata.SerializeToString())

                run_metadata_list.append(run_metadata)
            else:
                start_time = time.time()
                sess.run(target_op)
                end_time = time.time()
                runtimes.append(end_time - start_time)

        _LOGGER.info('执行时间: %s',
                     str(time.time() - warmup_end_time))

        avg_step_time = np.average(runtimes)

        _LOGGER.info('计算图执行统计数据. #samples=%d, median=%s, mean=%s',
                     len(runtimes),
                     np.median(runtimes),
                     np.average(runtimes))

        return avg_step_time, run_metadata_list


def generate_cost_dict(target_op, warmup_count=5, num_measurement=10,
                       profile_every_n_steps=5, sess_config=None, logdir=None):
    avg_step_time, run_metadata_list = measure_graph(
        target_op,
        warmup_count=warmup_count,
        num_measurement=num_measurement,
        profile_every_n_steps=profile_every_n_steps,
        logdir=logdir,
        config=sess_config)
    cost_dict = cost_lib.build_cost_dict(run_metadata_list)
    return avg_step_time, cost_dict


def run_random_placement(tf_graph, devices, ignore_colocation=True):
    """Places the operators in tf.Graph over the devices randomly."""
    _LOGGER.info('Run the random placement. #devices=%d, ignore_colocation=%s',
                 len(devices), str(ignore_colocation))
    graph = tf.get_default_graph()
    stats = {}
    for tf_op in graph.get_operations():
        device_name = devices[random.randrange(len(devices))]['name']
        stats[device_name] = stats.get(device_name, 0) + 1
        # pylint: disable=protected-access
        tf_op._set_device(device_name)


def generate_cost(target_op, cost_path, sess_config=None, logdir=None):
    if not cost_path:
        raise ValueError('生成cost的位置不正确.')

    graphdef = tf.get_default_graph().as_graph_def()

    start_time = time.time()
    step_time, cost_dict = generate_cost_dict(
        target_op, sess_config=sess_config, logdir=logdir)

    _LOGGER.info('Original runtime: %f', step_time)

    cost_dir_path = os.path.dirname(cost_path)
    if cost_dir_path:
        os.makedirs(cost_dir_path, exist_ok=True)
    # pylint: disable=invalid-name
    with open(cost_path, 'wb') as f:
        _LOGGER.info('Saving to %s...', cost_path)
        cost_data = {'graphdef': graphdef,
                     'cost_dict': cost_dict}
        pickle.dump(cost_data, f)

    _LOGGER.info('Profile run costs: %s', str(time.time() - start_time))


def run_placement(target_op, cost_path, comm_cost_coeffs, cost_factor,
                  logdir=None, sess_config=None):
    if not cost_path:
        raise ValueError('cost_path is required.')

    # pylint: disable=invalid-name
    with open(cost_path, 'rb') as f:
        cost_data = pickle.load(f)

    graph = tf.get_default_graph()

    assert cost_data['graphdef'] == graph.as_graph_def()

    devices = _get_gpu_devices(sess_config)

    cost_dict = cost_data['cost_dict']

    # adjust costs for sensitivity experiments.
    if cost_factor != 1.0:
        cost_dict, comm_cost_coeffs = cost_lib.adjust_costs(
            cost_factor, cost_dict, comm_cost_coeffs)

    start_time = time.time()
    placer = placer_lib.get_placer(
        graph,
        devices=devices,
        cost_dict=cost_dict,
        comm_cost_coeffs=comm_cost_coeffs)
    placer.run()
    _LOGGER.info('Entire placement time: %s', str(time.time() - start_time))


def _build_cv_inputs(model_name, batch_size, data_format):
    num_classes = _CLASSIFICATION_NUM[model_name]
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)

    if data_format == 'NHWC':
        input_shape = (batch_size,
                       network_fn.default_image_size,
                       network_fn.default_image_size,
                       3)
    else:
        input_shape = (batch_size,
                       3,
                       network_fn.default_image_size,
                       network_fn.default_image_size)

    images = np.ones(input_shape, dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)

    element = (images, labels)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def _build_nlp_inputs(batch_size, max_seq_length):
    input_shape = (batch_size, max_seq_length)

    src_input = np.ones(input_shape, dtype=np.int32)
    target_input = np.ones(input_shape, dtype=np.int32)
    target_output = np.ones(input_shape, dtype=np.int32)

    element = (src_input, target_input, target_output)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def build_inputs(model_name, batch_size, data_format, max_seq_length):
    if model_name in _CLASSIFICATION_NUM:
        return _build_cv_inputs(
            model_name, batch_size, data_format)
    return _build_nlp_inputs(batch_size, max_seq_length)


def build_train_graph(loss, optimizer_name, learning_rate,
                      colocate_grads_with_ops):
    optimizer = _configure_optimizer(optimizer_name, learning_rate)
    grads_and_vars = optimizer.compute_gradients(
        loss, colocate_gradients_with_ops=colocate_grads_with_ops)
    global_step = tf.train.create_global_step()
    return optimizer.apply_gradients(grads_and_vars,
                                     global_step=global_step)


def run_colorl(target_op, allotted_time, logdir, sess_config):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=sess_config):
        pass

    graph = tf.get_default_graph()

    cluster = gcluster.Cluster()
    metagraph = tf.train.export_meta_graph(graph=graph,
                                           clear_extraneous_savers=True)

    placed_metagraph_list = grappler_graph_placer.PlaceGraph(
        metagraph,
        cluster=cluster,
        allotted_time=allotted_time,
        verbose=True,
        sess_config=sess_config,
        gpu_only=True)

    # _LOGGER.info('# found metagraph: %d', len(placed_metagraph_list))

    if len(placed_metagraph_list) == 0:
        _LOGGER.info('没有找到合适的调度策略')
        return

    if logdir:
        metagraph_dir = os.path.join(logdir, 'metagraph')
        os.makedirs(metagraph_dir, exist_ok=True)
        for i, metagraph in enumerate(placed_metagraph_list):
            metagraph_path = os.path.join(
                metagraph_dir, 'metagraph-%d.pbtxt' % i)
            with open(metagraph_path, 'wb') as f:
                f.write(metagraph.SerializeToString())
    placed_metagraph = placed_metagraph_list[-1]

    for node in placed_metagraph.graph_def.node:
        tf_op = graph.get_operation_by_name(node.name)
        tf_op._set_device(node.device)

    step_time = measure_graph(
        target_op, warmup_count=10, num_measurement=21,
        profile_every_n_steps=21, logdir=logdir,
        config=sess_config)[0]

    _LOGGER.info('平均执行时间: {}'.format(step_time))


def generate_aware_graph(target_op, dest_graph_output, cost_path, allotted_time, comm_cost_coeffs, logdir, verbose, sess_config):
    graph = tf.get_default_graph()
    devices = _get_gpu_devices(sess_config)
    run_random_placement(graph, devices, ignore_colocation=False)
    with tf.Session(config=sess_config):
        pass
    cluster = gcluster.Cluster(disable_detailed_stats=False, disable_timeline=False)
    metagraph = tf.train.export_meta_graph(graph=graph,
                                           clear_extraneous_savers=True)
    metagraph_list = []
    if cluster is None:
        cluster = gcluster.Cluster(disable_detailed_stats=False, disable_timeline=False)
    metagraph_copy = meta_graph_pb2.MetaGraphDef()
    metagraph_copy.CopyFrom(metagraph)
    item = gitem.Item(metagraph, ignore_colocation=False)
    try:
        op_perf, original_run_time, step_stats = cluster.MeasureCosts(item)
        metagraph_list.append(metagraph)
        if verbose:
            tf_logging.info("原始的执行时间: " +
                            str(original_run_time))
    except errors.OpError as e:
        if verbose:
            tf_logging.info("原始图无法生成: " + str(e))

    placer = placer_lib.get_placer(
        graph,
        metagraph=metagraph,
        op_perf=op_perf,
        step_stats=step_stats,
        devices=devices,
        only_important_ops=False,
        comm_cost_coeffs=comm_cost_coeffs)

    G = placer.get_fused_op_graph()
    ungrouped_mapping = placer.get_ungrouped_mapping()
    group_to_group = placer.get_group_to_group()
    node_list = ungrouped_mapping.keys()
    for key in node_list:
        ungrouped_mapping[key] = ungrouped_mapping[key].split("@")[1]
    func = lambda z: dict([(x, y) for y, x in z.items()])
    text = func(func(ungrouped_mapping)).values()
    print("目前预测节点数量：" + str(len(text)) + "\t" + str(len(G.nodes)))

    with open(dest_graph_output, 'wb') as file:
        pickle.dump({"optim_mg": metagraph,
                     "op_perf": op_perf,
                     "step_stats": step_stats,
                     "G": G,
                     'group_to_group': group_to_group,
                     "ungrouped_mapping": ungrouped_mapping}, file)
    print("前向图生成结束！！")
    sys.exit()


def generate_aware_backward_graph(target_op, dest_graph_output, cost_path, allotted_time, comm_cost_coeffs, logdir, verbose, sess_config):

    def get_raw_colocation_group(tf_op):
        """Returns a raw string-typed co-location group of the given TF op."""
        return [bytes_to_native_str(colocation_group)
                for colocation_group in tf_op.colocation_groups()]

    if not dest_graph_output:
        raise ValueError('需要指定cost的位置.')

    with open(dest_graph_output, 'rb') as f:
        row_data = pickle.load(f)

    if not cost_path:
        raise ValueError('需要指定cost的位置.')

    with open(cost_path, 'rb') as f:
        cost_data = pickle.load(f)

    graph = tf.get_default_graph()

    assert cost_data['graphdef'] == graph.as_graph_def()

    devices = _get_gpu_devices(sess_config)

    cost_dict = cost_data['cost_dict']

    with tf.Session(config=sess_config):
        pass
    cluster = gcluster.Cluster(disable_detailed_stats=False, disable_timeline=False)
    metagraph = tf.train.export_meta_graph(graph=graph,
                                           clear_extraneous_savers=True)
    metagraph_list = []
    if cluster is None:
        cluster = gcluster.Cluster(disable_detailed_stats=False, disable_timeline=False)
    metagraph_copy = meta_graph_pb2.MetaGraphDef()
    metagraph_copy.CopyFrom(metagraph)
    item = gitem.Item(metagraph, ignore_colocation=False)
    try:
        op_perf, original_run_time, step_stats = cluster.MeasureCosts(item)
        metagraph_list.append(metagraph)
        if verbose:
            tf_logging.info("原始的执行时间: " +
                            str(original_run_time))
    except errors.OpError as e:
        if verbose:
            tf_logging.info("原始图无法生成: " + str(e))

    ungrouped_mapping = row_data["ungrouped_mapping"]
    group_to_group = row_data["group_to_group"]
    single_node = {}
    func = lambda z: dict([(x, y) for y, x in z.items()])
    text1 = func(func(group_to_group)).values()
    for i, tf_op in enumerate(graph.get_operations()):
        if tf_op.name in ungrouped_mapping:
            continue
        else:
            group = get_raw_colocation_group(tf_op)[0]
            if group in group_to_group.keys():
                ungrouped_mapping[tf_op.name] = group_to_group[group]
            else:
                single_node[tf_op.name] = group
    for key, value in single_node.items():
        ungrouped_mapping[key] = list(text1)[random.randint(0, len(text1)-1)]

    text2 = func(func(ungrouped_mapping)).values()
    assert i + 1 == len(ungrouped_mapping) and len(text2) == len(row_data['G'].nodes)

    # G = placer.get_fused_op_graph()
    # ungrouped_mapping = placer.get_ungrouped_mapping()
    # func = lambda z: dict([(x, y) for y, x in z.items()])
    # text = func(func(ungrouped_mapping)).values()
    # print("目前预测节点数量：" + str(len(text)) + "\t" + str(len(G.nodes)))
    #
    with open('./input.pkl', 'wb') as file:
        pickle.dump({"optim_mg": metagraph,
                     "op_perf": op_perf,
                     "step_stats": step_stats,
                     "G": row_data['G'],
                     "ungrouped_mapping": ungrouped_mapping}, file)
    print("反向图生成结束！！")
    sys.exit()


def schedule_ops_and_exe(targe_op, pl, digraph, sess_config=None):
    ungrouped_mapping = digraph['ungrouped_mapping']
    graph = tf.get_default_graph()
    devices = _get_gpu_devices(sess_config)
    available_devices = [device["name"] for device in devices]
    ungrouped_pl = {}
    for op in digraph['optim_mg'].graph_def.node:
        # if op.name in ungrouped_mapping.keys():
        #     if ungrouped_mapping[op.name] in pl.keys():
        #         ungrouped_pl[op.name] = pl[ungrouped_mapping[op.name]]
        #     else:
        #         ungrouped_pl[op.name] = 0
        # else:
        #     ungrouped_pl[op.name] = 0
        ungrouped_pl[op.name] = 0

    ios = ImportantOpsRewarder(digraph['optim_mg'], digraph['op_perf'], digraph['step_stats'], available_devices)
    run_time, peak_utils, _, comm_utils = ios.simulate(ungrouped_pl, sim_mem_usage=True, sim_com_usage=True)
    _LOGGER.debug('执行时间: %s', str(run_time))
    _LOGGER.debug('内存: %s', str(peak_utils))
    _LOGGER.debug('通信开销: %s', str(comm_utils))
    for tf_op in graph.get_operations():
        if ungrouped_mapping[tf_op.name] in pl.keys():
            tf_op._set_device(devices[pl[ungrouped_mapping[tf_op.name]]]["name"])


def parse_comm_cost_coeffs(coeffs_str, factor=1.0):
    comm_cost_coeffs = coeffs_str.split(',')
    assert len(comm_cost_coeffs) == 2

    comm_cost_coeffs[0] = float(comm_cost_coeffs[0])
    comm_cost_coeffs[1] = int(comm_cost_coeffs[1])

    if factor != 1.0:
        _LOGGER.info('通信代价影响因子: %s', str(factor))
        comm_cost_coeffs = tuple(
            [value * factor for value in comm_cost_coeffs])

    return comm_cost_coeffs


def main(unparsed_args):
    if len(unparsed_args) > 1:
        raise RuntimeError('Unparsed args: {}'.format(unparsed_args[1:]))

    FLAGS = tf.app.flags.FLAGS

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement)

    if FLAGS.memory_fraction != 1.0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = \
            FLAGS.memory_fraction
    sess_config.graph_options.optimizer_options.opt_level = -1
    _LOGGER.debug('Session config: %s', str(sess_config))

    inputs = build_inputs(
        model_name=FLAGS.model_name,
        batch_size=FLAGS.batch_size,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        max_seq_length=FLAGS.max_seq_length,
    )

    loss = build_deep_learning_network(
        inputs=inputs,
        model_name=FLAGS.model_name,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        vocab_size=FLAGS.vocab_size,
        rnn_units=FLAGS.rnn_units,
        num_layers=FLAGS.num_layers,
        rnn_unit_type=FLAGS.rnn_unit_type,
        encoder_type=FLAGS.encoder_type,
        residual=FLAGS.residual,
        num_gpus=FLAGS.num_gpus,
        colocation=not FLAGS.disable_nmt_colocation)

    only_forward = FLAGS.only_forward
    _LOGGER.info('只考虑前向传播的算子: %s', str(only_forward))
    colocate_grads_with_ops = FLAGS.colocate_grads_with_ops
    _LOGGER.info('处理反向传播的梯度算子: %s' % str(colocate_grads_with_ops))

    comm_cost_coeffs = parse_comm_cost_coeffs(
        FLAGS.comm_cost_coeffs, FLAGS.comm_cost_factor)

    if only_forward:
        assert colocate_grads_with_ops

    tf.add_to_collection(tf.GraphKeys.TRAIN_OP, loss)

    target_op = loss

    if FLAGS.costgen:
        if not only_forward:
            train_op = build_train_graph(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op
        generate_cost(target_op,
                      cost_path=FLAGS.cost_path,
                      sess_config=sess_config,
                      logdir=FLAGS.logdir)
        # 生成包括backwards的性能数据

    else:
        if not only_forward:
            train_op = build_train_graph(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        if FLAGS.exe_pattern == "ColorRL":
            run_colorl(
                target_op,
                allotted_time=FLAGS.grappler_time,
                logdir=FLAGS.logdir,
                sess_config=sess_config)
            return
        elif FLAGS.exe_pattern == "aware_forward":
            generate_aware_graph(target_op,
                                 dest_graph_output=FLAGS.dest_graph_output,
                                 cost_path=FLAGS.cost_path,
                                 allotted_time=FLAGS.grappler_time,
                                 comm_cost_coeffs=comm_cost_coeffs,
                                 logdir=FLAGS.logdir, verbose=True, sess_config=sess_config)
        elif FLAGS.exe_pattern == 'eval':
            with open(FLAGS.place_path, 'r') as f:
                pl = json.load(f)
            with open(FLAGS.eval_graph_dest, 'rb') as f:
                digraph = pickle.load(f)
            schedule_ops_and_exe(target_op, pl, digraph, sess_config=sess_config)
            step_time = measure_graph(
                target_op, warmup_count=10, num_measurement=51,
                profile_every_n_steps=51, logdir=FLAGS.logdir,
                config=sess_config)[0]
            _LOGGER.info('平均执行时间: {}'.format(step_time))
        elif FLAGS.exe_pattern == 'aware_back':
            assert FLAGS.colocate_grads_with_ops and not only_forward
            generate_aware_backward_graph(target_op,
                                 dest_graph_output=FLAGS.dest_graph_output,
                                 cost_path=FLAGS.cost_path,
                                 allotted_time=FLAGS.grappler_time,
                                 comm_cost_coeffs=comm_cost_coeffs,
                                 logdir=FLAGS.logdir, verbose=True, sess_config=sess_config)

        if only_forward:
            train_op = build_train_graph(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        step_time = measure_graph(
            target_op, warmup_count=10, num_measurement=51,
            profile_every_n_steps=51, logdir=FLAGS.logdir,
            config=sess_config)[0]

        _LOGGER.info('平均执行时间: {}'.format(step_time))


if __name__ == "__main__":
    tf.app.run(main)
