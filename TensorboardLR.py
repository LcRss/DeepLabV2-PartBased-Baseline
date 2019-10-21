from Utils import *
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as k


class TensorboardLR(Callback):
    def __init__(self,
                 log_dir='./log',
                 write_graph=True):
        self.write_graph = write_graph
        self.log_dir = log_dir

    def set_model(self, model):
        self.model = model
        self.sess = k.get_session()
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    # def on_epoch_end(self, epoch, logs={}):
    def on_train_batch_end(self, batch, logs={}):
        logs.update({'learning_rate': float(k.get_value(self.model.optimizer.lr))})
        index = tf.keras.backend.eval(self.model.optimizer.iterations)
        self._write_logs(logs, index)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)

        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
