import pathlib
from flynet import FlyNet

data_dir = pathlib.Path().home() / 'flynet_data/training'
file_train = 'labels.h5'
file_valid = 'valid_labels.h5'
learning_rate = 1.0e-4
decay = 1.0e-6

net = FlyNet.Network()
net.set_learning_rate(learning_rate, decay)
net.set_annotated_data(str(data_dir), file_train, file_valid)
net.set_batch_size(500)
net.load_network()
net.set_N_epochs(50)
net.train_network()
plt.show()
