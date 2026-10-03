# %% 
from attr import s
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
# %% 
def preprocess_behavior_data(data,params):

  spike_time = data['spike_time']
  beh_time = data['beh_time']
  pos_x = data['pos_x']
  pos_y = data['pos_y']
  pos_lin = data['pos_lin']
  trial_data = data['trial_data']
  session_epochs = data['session_epochs']
  startTime = data['startTime']
  stopTime = data['stopTime']
  cell_type = data['cell_type']

  speed_lim = params['speed_lim']
  spk_bin = params['spk_bin']
  sm_window = params['sm_window']
  seed = params['seed']
  trial_bin = params['trial_bin']
  pos_bin = params['pos_bin']
  num_shuffle = params['num_shuffle']
  sub_cell  = params['sub_cell']
  show_plot = params['show_plot']
  # 3.1 Get data to the behavior epoch

  beh_ts = (beh_time>startTime) &  (beh_time<stopTime)
  ts_beh =  beh_time[beh_ts]
  x_pos = pos_x[beh_ts]
  y_pos = pos_y[beh_ts]
  lin_pos = pos_lin[beh_ts]
  trial = trial_data[beh_ts]
  epoch_ind = sessionID*np.ones_like(x_pos)

  # 3.2.SPIKE COUNT
  num_cell = spike_time.shape[0]
  spk_ts = np.arange(ts_beh[0], ts_beh[-1], spk_bin)
  spk_count = np.zeros((num_cell,spk_ts.shape[0]-1))


  for spk in range(num_cell):
      # SPIKE COUNT
      hist_count = np.histogram(spike_time[spk],bins=spk_ts)
      spk_count[spk,:] = hist_count[0]

  spk_data = np.transpose(spk_count)
  spk_ts = spk_ts[1:]
  num_cell = spk_data.shape[1]
  
  return spk_data 

# %% 
import os 
os.chdir("Data")
# %% 
data = load_behavior_data(basename, sessionID)
# %% 

beh_start = data['beh_time'][0]
beh_end   = data['beh_time'][-1]

duration = beh_end - beh_start
duration = int(duration[0])
spike_times = data['spike_time']
rates = np.empty(spike_times.size)
max_number = 0 
for i, spike in enumerate(spike_times):

    spike = spike.flatten()
    n_spikes = np.sum(
        (spike > beh_start) &
        (spike < beh_end)
    )
    rates[i] = n_spikes / duration
    # max_number = max(spike.max(), max_number) 
    # rates[i] = spike.size 



# %% 
# %% 
plt.bar(np.arange(rates.size), rates )
# %% 
import matplotlib.pyplot as plt

spikes = data['spike_time']

# Convert each neuron's spikes into 1D arrays
spike_times = [s.flatten() for s in spikes]

plt.figure(figsize=(15, 8))

plt.eventplot(
    spike_times,
    colors='black',
    lineoffsets=range(len(spike_times)),
    linelengths=0.8,
    linewidths=0.5
)

plt.xlabel("Time")
plt.ylabel("Neuron Index")
plt.title("Neuron Spike Raster Plot")

plt.tight_layout()
plt.show()
# %% 
spike_time = preprocess_behavior_data(data, params) 
# %% 
sparsity = 1 - (np.count_nonzero(spike_time)/spike_time.size)
print(f"Sparsity {sparsity}")
#%% 
# Randomly sample 2 seconds and plot spike raster 

import matplotlib.pyplot as plt 
spikes = data['spike_time'][0].flatten()

def get_random_window(spikes, window=2.0):
    t_min = spikes.min()
    t_max = spikes.max()

    t_start = np.random.uniform(t_min, t_max - window)
    t_end = t_start + window

    return t_start, t_end

def plot_single_neuron_raster(spikes, t_start, t_end):
    # select spikes in window
    spikes_window = spikes[(spikes >= t_start) & (spikes < t_end)]

    plt.figure(figsize=(8, 2))

    # plot as raster (single neuron → y=0)
    plt.scatter(spikes_window - t_start,
                np.zeros_like(spikes_window),
                s=10, color='black')

    plt.xlim(0, t_end - t_start)
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.title(f"Neuron spikes ({t_start:.2f}–{t_end:.2f} s)")
    plt.show()

# %% 
spikes = data['spike_time'][200].flatten()
# %% 
t_start, t_end = get_random_window(spikes, window=2.0)
plot_single_neuron_raster(spikes, t_start, t_end)
# %% 
from collections import defaultdict

def group_cells_by_type(cell_type):
    cell_dict = defaultdict(list)

    for idx, ct in enumerate(cell_type):
        # extract string safely
        if isinstance(ct, np.ndarray):
            label = ct[0]
        else:
            label = str(ct)

        cell_dict[label].append(idx)

    return dict(cell_dict)

cell_groups = group_cells_by_type(data['cell_type'])
#%% 
cell_groups['Narrow Interneuron']
# %% 
def spike_ms_binned(data, params):
    spike_time = data['spike_time']
    spk_bin = 0.001
    ts_beh = data['beh_time'][30000:30300, 0]

    num_cell = spike_time.shape[0]
    spk_ts = np.arange(ts_beh[0], ts_beh[-1], spk_bin)
    spk_count = np.zeros((num_cell,spk_ts.shape[0]-1))

    for spk in range(num_cell):
        # SPIKE COUNT
        hist_count = np.histogram(spike_time[spk],bins=spk_ts)
        spk_count[spk,:] = hist_count[0]

    spk_data = np.transpose(spk_count)
    spk_ts = spk_ts[1:]
    num_cell = spk_data.shape[1]
    
    return spk_data 
