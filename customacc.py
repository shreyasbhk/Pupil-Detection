import tensorflow as tf

class Metric(object):
    """ Base Metric Class.

    Metric class is meant to be used by TFLearn models class. It can be
    first initialized with desired parameters, and a model class will
    build it later using the given network output and targets.

    Attributes:
        tensor: `Tensor`. The metric tensor.

    """
    def __init__(self, name=None):
        self.name = name
        self.tensor = None
        self.built = False

    def build(self, predictions, targets, inputs):
        """ build.

        Build metric method, with common arguments to all Metrics.

        Arguments:
            prediction: `Tensor`. The network to perform prediction.
            targets: `Tensor`. The targets (labels).
            inputs: `Tensor`. The input data.

        """
        raise NotImplementedError

    def get_tensor(self):
        """ get_tensor.

        Get the metric tensor.

        Returns:
            The metric `Tensor`.

        """
        if not self.built:
            raise Exception("Metric class Tensor hasn't be built. 'build' "
                            "method must be invoked before using 'get_tensor'.")
        return self.tensor

class Distance(Metric):

    def __init__(self, name=None):
        super(Distance, self).__init__(name)
        self.name = "Distance" if not name else name

    def build(self, predictions, targets, inputs=None):
        """ Build standard error tensor. """
        self.built = True
        self.tensor = dist_op(predictions, targets)
        # Add a special name to that tensor, to be used by monitors
        self.tensor.m_name = self.name

def dist_op(predictions, targets):
    return tf.sqrt(tf.reduce_sum(tf.square(predictions-targets)))