import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load in training data
history_file = sys.argv[1]
with open(history_file, 'rb') as f:
    data = pickle.load(f)

# Extract loss, val_loss and epoch number
loss = np.array(data['loss'])
log10_loss = np.log10(loss)
val_loss = np.array(data['val_loss'])
log10_val_loss = np.log10(val_loss)
epochs = np.arange(loss.shape[0])

# Save epoch#, log10(loss) and log10(val_loss) to .csv file
with open('flynet_training_loss.csv', 'w') as f:
    f.write('epoch#, log10(loss), log10(val_loss)\n')
    for x, y, z in zip(epochs, log10_loss, log10_val_loss):
        f.write(f'{x}, {y}, {z}\n')

# Plot epoch#, log10(loss) and log10(val_loss) 
fig, ax = plt.subplots(1,1, figsize=(20,4))
ax.plot(epochs, log10_loss, 'b')
ax.plot(epochs, log10_val_loss, 'g')
ax.set_xlabel('number of epochs')
ax.set_ylabel(r'$\mathrm{log}_{10}$ mean square error')
ax.grid(True)
fig.savefig('flynet_training_loss.png')
plt.show()









