# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from networks import get_network
import os

from pprint import pprint

meta_path = '/home/nishchalj/Desktop/PoseEstimationForMobile/trained/mv2_cpm_tiny/models/mv2_cpm_batch-10_lr-0.001_gpus-3_192x192_experiments-mv2_cpm/model-26000.meta' # Your .meta file

#os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--model', type=str, default='mv2_cpm', help='')
parser.add_argument('--size', type=int, default=192)
parser.add_argument('--checkpoint', type=str, default='/home/nishchalj/Desktop/PoseEstimationForMobile/trained/mv2_cpm_tiny/models/mv2_cpm_batch-10_lr-0.001_gpus-3_192x192_experiments-mv2_cpm/model-26000.meta', help='checkpoint')
parser.add_argument('--output_node_names', type=str, default='GPU_0/Convolutional_Pose_Machine/stage_5_out')
parser.add_argument('--output_graph', type=str, default='./modelf.pb', help='output_freeze_path')

args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.size, args.size, 3], name="image")

with tf.Session() as sess:
    net = get_network(args.model, input_node, trainable=False)
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
   # saver = tf.train.import_meta_graph(meta_path)
   # saver.restore(sess,tf.train.latest_checkpoint('/home/nishchalj/Desktop/PoseEstimationForMobile/trained/mv2_cpm_tiny/models/mv2_cpm_batch-10_lr-0.001_gpus-3_192x192_experiments-mv2_cpm/'))
    input_graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )

with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
