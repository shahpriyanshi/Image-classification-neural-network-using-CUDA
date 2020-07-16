from keras import backend as K
import tensorflow as tf
import uff
K.set_learning_phase(0)

from keras.models import load_model
model = load_model('../model/classification_v3.h5')
print(model.outputs)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


print([out.op.name for out in model.outputs])
frozen_graph = freeze_session(tf.keras.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])
frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

tf.train.write_graph(frozen_graph, "../model", "image_classification_model.pb", as_text=False)

print([out.op.name for out in model.inputs])
print([out.op.name for out in model.outputs])


uff.from_tensorflow('../model/image_classification_model.pb', output_filename='image_classification_model.uff',
                    output_nodes=[out.op.name for out in model.outputs])