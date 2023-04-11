import matplotlib.pyplot as plt
import pickle as pk

history = pk.load(open('res-history.pkl', 'rb'))


fig, ax = plt.subplots(2,1)
ax[0].plot(history['loss'], label='train')
ax[0].plot(history['val_loss'], label='test')
ax[1].plot(history['accuracy'], label='train')
ax[1].plot(history['val_accuracy'], label='test')

ax[0].set_title('Loss')
ax[1].set_title('Accuracy')

plt.legend()

plt.savefig('result.png')