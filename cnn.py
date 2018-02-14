import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from sklearn.metrics import confusion_matrix, roc_auc_score

model_version = 0
batch_size = 100
val_batch_size = 350
image_dimensions = (120, 160)
train_dataset_file = "../Data/train-"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
val_dataset_file = "../Data/val-"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
test_dataset_file = "../Data/test-"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"

class ConCaDNet(tfe.Network):
    """
    The ConCaDNet Model Class.
    """
    def __init__(self):
        super(ConCaDNet, self).__init__(name='')
        self.l1_1 = self.track_layer(tf.layers.Conv2D(48, 11, strides=1, padding="SAME", name="Conv_1_1",
                                                      activation=tf.nn.leaky_relu))
        self.l1_mp = self.track_layer(tf.layers.MaxPooling2D(3, strides=2, padding="SAME"))

        self.l2_1 = self.track_layer(tf.layers.Conv2D(128, 5, strides=1, padding="SAME", name="Conv_2_1",
                                                      activation=tf.nn.leaky_relu))
        self.l2_mp = self.track_layer(tf.layers.MaxPooling2D(3, strides=2, padding="SAME"))

        self.l3_1 = self.track_layer(tf.layers.Conv2D(192, 3, strides=1, padding="SAME", name="Conv_3_1",
                                                      activation=tf.nn.leaky_relu))
        self.l3_2 = self.track_layer(tf.layers.Conv2D(192, 3, strides=1, padding="SAME", name="Conv_3_2",
                                                      activation=tf.nn.leaky_relu))
        self.l3_3 = self.track_layer(tf.layers.Conv2D(128, 3, strides=1, padding="SAME", name="Conv_3_3",
                                                      activation=tf.nn.leaky_relu))
        self.l3_mp = self.track_layer(tf.layers.MaxPooling2D(3, strides=2, padding="SAME"))

        self.fc_out = self.track_layer(tf.layers.Dense(units=1))

    def call(self, inputs, training=False):
        x = ((inputs-tf.reduce_min(inputs))/tf.reduce_max(inputs))-0.5
        conv1 = self.l1_1(x)
        conv2 = self.l1_mp(conv1)
        conv3 = self.l2_1(conv2)
        conv4 = self.l2_mp(conv3)
        conv5 = self.l3_1(conv4)
        conv6 = self.l3_2(conv5)
        conv7 = self.l3_3(conv6)
        conv = self.l3_mp(conv7)
        conv = tf.layers.flatten(conv)
        conv = tf.nn.dropout(conv, keep_prob=0.5) if training else conv
        conv = self.fc_out(conv)
        return conv

def initialize_datasets():
    with tf.device("/cpu:0"):
        def parser_function(example_proto):
            features = {
                "image": tf.FixedLenFeature((), tf.string, default_value=""),
                "x-label": tf.FixedLenFeature((), tf.float32),
                "y-label": tf.FixedLenFeature((), tf.float32),
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            image = tf.reshape(tf.decode_raw(parsed_features["image"], tf.float32),
                               [image_dimensions[0], image_dimensions[1], 3])
            label = tf.reshape(tf.cast(parsed_features["x-label"], tf.float32), [1])
            return image, label
        dataset = tf.data.TFRecordDataset(train_dataset_file)
        dataset = dataset.map(parser_function)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(batch_size*3)
        dataset = dataset.batch(batch_size)

        val_dataset = tf.data.TFRecordDataset(val_dataset_file)
        val_dataset = val_dataset.map(parser_function)
        val_dataset = val_dataset.repeat(1)
        #val_dataset = val_dataset.shuffle(val_batch_size)
        val_dataset = val_dataset.batch(val_batch_size)

        test_dataset = tf.data.TFRecordDataset(test_dataset_file)
        test_dataset = test_dataset.map(parser_function)
        test_dataset = test_dataset.repeat(1)
        #test_dataset = test_dataset.shuffle(val_batch_size)
        test_dataset = test_dataset.batch(val_batch_size)
        return dataset, val_dataset, test_dataset

def loss(preds, labels):
    return tf.losses.sigmoid_cross_entropy(labels, preds)

def train_one_epoch(model, optimizer, epoch, run_number, log_interval=None):
    tf.train.get_or_create_global_step()
    def model_loss_auc(x, y):
        preds = model(x, display_image=False)
        loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return loss_value.numpy(), auc

    def model_loss(x, y):
        preds = model(x, display_image=False, training=True)
        loss_value = loss(preds, y)
        return loss_value

    with tf.device("/GPU:0"):
        tr_ds, v_ds, t_ds = initialize_datasets()
        for (batch, (x, y)) in enumerate(tfe.Iterator(tr_ds)):
            grads = tfe.implicit_gradients(model_loss)(x, y)
            optimizer.apply_gradients(grads)
            if batch%log_interval == 0:
                evaluate(model, model_loss_auc(x, y), [v_ds, t_ds], epoch, batch, run_number)
            if batch%(log_interval*2) == 0:
                global_step = tf.train.get_or_create_global_step()
                all_variables = (model.variables + optimizer.variables() + [global_step])
                saver = tfe.Saver(all_variables)
                _ = saver.save("../Models/"+str(model_version)+"/"+str(run_number)+"/"+str(epoch)+"-"+str(batch))

def save_training_progress(run_number, vars):
    with open("../Models/"+str(model_version)+ "/"+str(run_number)+"/Training_Progress.txt", "a+") as f:
        f.write(vars+"\n")

def evaluate(model, train_values, datasets, epoch, batch, run_number):
    def get_ranges(x):
        return model(x, return_ranges=True)

    def model_loss_auc(x, y, display_image):
        preds = model(x, display_image=display_image)
        loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return loss_value.numpy(), auc
    with tf.device("/GPU:0"):
        trl, tra = train_values
        val_dataset = datasets[0]
        test_dataset = datasets[1]
        val = tfe.Iterator(val_dataset)
        x, y, s, d = val.next()
        vl, va = model_loss_auc(x, y, display_image=False)
        test = tfe.Iterator(test_dataset)
        x, y, s, d = test.next()
        tl, ta = model_loss_auc(x, y, display_image=True)
        test = tfe.Iterator(test_dataset)
        x, y, s, d = test.next()
        ranges = get_ranges(x)
        for r in range(len(ranges)):
            print("Layer {} Ranges: {}".format(r, ranges["Layer "+str(r)]))
        print("Epoch: {}, Batch {}, Training Loss: {:.5f}, Training AUC: {:.2f}, " 
              "Validation Loss: {:.5f}, Validation AUC: {:.2f}, Testing Loss: {:.5f}, " 
              "Testing AUC: {:.2f}".format(epoch, batch, trl, tra*100, vl, va*100, tl, ta*100))
        save_training_progress(run_number, "Epoch: {}, Batch {}, Training Loss: {:.5f}, Training AUC: {:.2f}, "
                               "Validation Loss: {:.5f}, Validation AUC: {:.2f}, Testing Loss: {:.5f}, "
                               "Testing AUC: {:.2f}, Layer Ranges: {}"
                               .format(epoch, batch, trl, tra*100, vl, va*100, tl, ta*100, ranges))