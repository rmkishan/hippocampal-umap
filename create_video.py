import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import os 
os.chdir("Data")

import mat73
import scipy.io as sio 
import numpy as np 
basename='e13_26m1_210913'
sessionID = 2
params = {
    'speed_lim':0,
    'supervised':False,
    'spk_bin':0.1,
    'sm_window':3,
    'seed':0,
    'trial_bin':5,
    'pos_bin':5,
    'method':'umap',
    'n_component':6,
    'iterations':2000,
    'num_shuffle': 0,
    'sub_cell': 'all',
    'show_plot': True,
}


# %% 
def load_behavior_data(basename,sessionID):

    # LOAD SPIK TIME DATA
    spike_fname = basename + '.spike_time.mat'
    try:
        spike_mat = sio.loadmat(spike_fname)
    except:
        spike_mat = mat73.loadmat(spike_fname)

    spike_time = spike_mat['spike_time'][0]
    # LOAD BEHAVIOR DATA
    beh_fname = basename + '.Behavior.mat'
    try:
        beh_mat = sio.loadmat(beh_fname)
    except:
        beh_mat = mat73.loadmat(beh_fname)

    beh_time = beh_mat['behavior']['timestamps'][0][0]
    pos_x =beh_mat['behavior']['position'][0][0]['x'][0][0]
    pos_y = beh_mat['behavior']['position'][0][0]['y'][0][0]
    pos_lin = beh_mat['behavior']['position'][0][0]['lin'][0][0]
    trial_data = beh_mat['behavior']['masks'][0][0]['TRIALS'][0][0]

    # LOAD SESSION INFO
    session_fname = basename + '.session.mat'
    try:
        session_mat = sio.loadmat( session_fname)
    except:
        session_mat = mat73.loadmat( session_fname)

    try:
        session_epochs = session_mat['session']['epochs'][0][0]
        startTime = session_epochs[0,sessionID-1]['startTime'][0][0][0][0]
        stopTime = session_epochs[0,sessionID-1]['stopTime'][0][0][0][0]
    except:
        session_epochs = session_mat['session']['epochs']
        startTime = session_epochs[sessionID-1]['startTime']
        stopTime = session_epochs[sessionID-1]['stopTime']

    # LOAD CELL TYPE
    cell_fname = basename + '.cell_type.mat'
    try:
        cell_mat = sio.loadmat(cell_fname)
    except:
        cell_mat = mat73.loadmat(cell_fname)

    cell_type = cell_mat['cell_type'][0]
    # pyramidal_cell = []
    # for id,ct in enumerate(cell_type):
    #     if ct == 'Pyramidal Cell':
    #         pyramidal_cell.append(id)

    data = {}
    data= {'spike_time' : spike_time,
         'beh_time':beh_time,
         'pos_x':pos_x,
         'pos_y':pos_y,
         'pos_lin':pos_lin,
         'trial_data':trial_data,
         'session_epochs':session_epochs,
         'startTime':startTime,
         'stopTime':stopTime,
         'cell_type':cell_type,
    }

    del spike_fname,session_fname, beh_fname, cell_fname
    return data

data = load_behavior_data(basename, sessionID)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def make_flipbook_raster(spike_time, beh_time, duration=10, save_path="raster_flipbook.mp4"):

    beh_time = beh_time.flatten()

    # start at beginning of behavior
    t_start = float(beh_time[0])
    t_end = t_start + duration

    # DISCRETE 1-second chunks (flipbook)
    frame_times = np.arange(t_start, t_end, 1.0)

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame_t):
        ax.clear()

        for i, spikes in enumerate(spike_time):
            spikes = np.array(spikes).flatten()
            spikes_window = spikes[(spikes >= frame_t) & (spikes < frame_t + 1.0)]

            ax.scatter(spikes_window - frame_t,
                       np.ones_like(spikes_window) * i,
                       s=2, color='black')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(spike_time))
        ax.set_title(f"{frame_t:.2f} – {frame_t+1:.2f} s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron")

    anim = FuncAnimation(fig, update, frames=frame_times)

    # ✅ Proper video writer (not incremental saving)
    writer = FFMpegWriter(fps=5)  # slow fps → flipbook feel
    anim.save(save_path, writer=writer)

    print(f"Saved video to {save_path}")


make_flipbook_raster(
    data['spike_time'],
    data['beh_time'],
    duration=20  # start small!
)
