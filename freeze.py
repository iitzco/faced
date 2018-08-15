import tensorflow as tf


def run(model_dir):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        ckpt_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
        saver.restore(sess, ckpt_path)

        output_node_names= ["img", "training", "prob", "x_center", "y_center", "w", "h"]
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), output_node_names)

        output_graph="graph.pb"
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            sess.close()


if __name__ == "__main__":
    run("../yolo-face-artifacts/run6/models/")
