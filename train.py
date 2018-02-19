import tensorflow as tf
import os
from tensorflow.contrib.eager.python import tfe
from cnn import ConCaDNet, train_one_epoch

tfe.enable_eager_execution()

model_version = 0
run_number = 0
learning_rate = 0.01
num_epochs = 10
model_dir = "../Models/"+str(model_version)+"/"+str(run_number)+"/"

def train_cnn_model(run_number):
    model = ConCaDNet()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for epoch in range(1, num_epochs):
        print("Epoch #"+str(epoch))
        tf.train.get_or_create_global_step()
        with tfe.restore_variables_on_create(tf.train.latest_checkpoint(model_dir)):
            train_one_epoch(model, optimizer, epoch, run_number, log_interval=5)

if __name__ == "__main__":
    for a in range(0, 1):
        train_cnn_model(run_number=a)