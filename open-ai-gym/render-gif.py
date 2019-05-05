import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import animation
import pickle

file_name = "grad-sarsa-out/mc"

with open('{}-frames.pickle'.format(file_name), 'rb') as handle:
    valid_frames = pickle.load(handle)

    def animate(i):
        patch.set_data(valid_frames[i])

    plt.figure(figsize=(valid_frames[0].shape[1] / 72.0, valid_frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(valid_frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(valid_frames), interval=50)
    anim.save('{}.gif'.format(file_name), writer='imagemagick', fps=50)
