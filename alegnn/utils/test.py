import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Settings
video_file = "myvid.mp4"
clear_frames = True     # Should it clear the figure between each frame?
fps = 15

# Output video writer
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
writer = FFMpegWriter(fps=fps, metadata=metadata)

fig = plt.figure()
x = np.arange(0,10,0.1)
with writer.saving(fig, video_file, 100):
    for i in np.arange(1, 10, 0.1):
        y = np.sin(x+i)
        if clear_frames:
            fig.clear()
        ax, = plt.plot(x, y, 'r-', linestyle="solid", linewidth=1)
        writer.grab_frame()
