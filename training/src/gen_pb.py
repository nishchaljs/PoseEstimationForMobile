import tensorflow as tf
import argparse
from networks import get_network
import os

from pprint import pprint

meta_path = '/home/nishchalj/Desktop/PoseEstimationForMobile/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-12_lr-0.001_gpus-3_192x192_experiments-mv2_hourglass/model-9000.meta' # Your .meta file
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--model', type=str, default='mv2_hourglass', help='')
parser.add_argument('--size', type=int, default=224)
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')
parser.add_argument('--output_node_names', type=str, default='GPU_0/hourglass_out_3')
parser.add_argument('--output_graph', type=str, default='./model9000.pb', help='output_freeze_path')

args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.size, args.size, 3], name="image")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess,tf.train.latest_checkpoint('/home/nishchalj/Desktop/PoseEstimationForMobile/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-12_lr-0.001_gpus-3_192x192_experiments-mv2_hourglass/'))
    #net = get_network(args.model, input_node, trainable=False)
    #saver = tf.train.Saver()
    #saver.restore(sess, args.checkpoint)
    input_graph_def = tf.get_default_graph().as_graph_def()
    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    writer = tf.summary.FileWriter("./output/", tf.get_default_graph())
    writer.close()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )

with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

