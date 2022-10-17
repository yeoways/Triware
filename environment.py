
import tensorflow as tf
if __name__ == '__main__':
    # import mosek
    # import sys
    # env = mosek.Env()
    # env.set_Stream(mosek.streamtype.log, lambda x: sys.stdout.write(x))
    # env.echointro(1)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    with tf.Session(config=sess_config) as sess:
        print(sess.list_devices())