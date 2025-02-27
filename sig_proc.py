import numpy as np
from scipy.stats import norm, t
import math
import torch
import torch.nn.functional as F

def sig_name(signal):
    if signal < 10:
        return f'sig0{signal}'
    else:
        return f'sig{signal}'

def trial_splitter(data, trial_windows, signal_no):
    """
    Converts absolute spike times to relative spike times within trials
    :param data: Full data dict, with channel no.s as keys - each channel includes absolute spike times (in ms)- Col. 2
    :param trial_windows: Absolute trial start and end times (in s) - numpy array (#trials, 2)
    :param signal_no: Channel number - int
    :return: Spike times relative to start of trial (in ms)
    """

    d = data[sig_name(int(signal_no))][:, 1:4]
    d[:, 1] = (d[:, 1] * 1000)
    rel_trial_data = d
    rel_trial_data[:, -1] = 0

    trial_no = 1
    for trial_start, trial_end in trial_windows:
        trial_mask = (d[:, 1] > trial_start) & (d[:, 1] < trial_end)
        rel_trial_data[trial_mask, 1] = d[trial_mask, 1] - trial_start
        rel_trial_data[trial_mask, -1] = trial_no
        trial_no += 1
    keep_index = rel_trial_data[:, -1]>0
    return rel_trial_data[keep_index]

def gen_psth(spiketimes, binsize, window, neurons = 1):
    """
    Generate Peristimulus Time Histogram from spike times
    :param spiketimes: Times at which spike occurs (np array), zeroed at event start
    :param binsize: Time width of bins (in ms) - Usually 1
    :param window: Time steps for histogram (np array) - usually -1000:1000, step size 1
    :return: PSTH
    """
    T = int((window.max() - window.min())/binsize)
    if (window.max() - window.min())%binsize != 0:
        T += 1
    psth = np.zeros((T+1, 1+neurons))
    psth[:, 0] = np.arange(window[0], window[0]+binsize*T+.001, binsize)
    spxtimes = np.sort(spiketimes[:, 1])
    spxcounts = spiketimes[:, 0]


    for j in range(int(T)):
        temp = spxcounts[np.logical_and((window[0] + binsize * j) <= spxtimes,
                                        spxtimes < (window[0] + binsize * (j + 1)))]
        for neuron in range(neurons):
            psth[j+1, 1+neuron] = (temp==(neuron+1)).sum()
    return psth

def gen_sdf(psth, ftype, w, bin_size = 1, varargin = None, multi_unit = True):
    """
    Generate Spike Density Function by convolving the spike time histogram with a convolution function
    :param psth: Peristimulus Time Histogram (from psth function)
    :param ftype: Kernel Function; ['boxcar', 'Gauss', 'exp'] (typically Gauss)
    :param w: kernel width (usually 1ms)
    :param varargin: if there is another input (ignore)
    :return: sdf
    """
    f_map ={ # Options for kernel type
        'boxcar': boxcar,
        'Gauss': gauss,
        'exp': exp,
        'gpfa': gpfa
    }
    if multi_unit:
        sdf = psth
        sdf = torch.Tensor(sdf.T).unsqueeze(1)
    else:
        sdf = psth[:, -1]
        sdf = torch.Tensor(sdf).unsqueeze(-1).unsqueeze(-1)
    sdf, kernel = f_map[ftype](sdf, w, bin_size)
    return sdf, kernel

def gauss(sdf, w, bin_size=1):
    gauss_width = max([11, 6*w+1])
    kernel = norm.pdf(np.arange(math.floor(-gauss_width/2), math.floor(gauss_width/2)+.01, step=bin_size), loc=0, scale=w)
    kernel_tensor = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
    # sdf = F.conv1d(torch.Tensor(sdf.T).unsqueeze(1), weight=kernel_tensor, padding = 'same')
    sdf = F.conv1d(sdf, weight=kernel_tensor, padding = 'same')
    # sdf = np.convolve(sdf, kernel, 'same')
    max_pos = np.argmax(kernel)
    sdf = sdf.detach().cpu().numpy()
    temp = np.arange(start = -max_pos+1, stop = kernel.size-max_pos +.01, step = 1)
    kernel = np.vstack([temp, kernel])

    return sdf, kernel

def boxcar(sdf, w):
    if ~(w % 2):
        w += 1
    kernel = np.ones((w, 1))/w
    sdf = np.convolve(sdf, kernel, 'same')
    maxpos = np.argmax(kernel)
    kernel = np.arange(start =-(w-1)/2, stop = (w-1)/2, step = 1)

    return sdf, kernel

def exp(sdf, w):
    filtbase = np.arange(1, np.min([3000, 5*w]))
    filtfunc = np.exp(-filtbase/w)
    kernel   = filtfunc/np.sum(filtfunc)
    dummy = np.conv(sdf, kernel)
    max_pos = np.argmax(kernel)
    temp = np.arange(start=-max_pos + 1, stop=kernel.size - max_pos + .01, step=1)
    kernel = np.vstack([temp, kernel])
    return sdf, kernel

def gpfa(sdf, w, bin_size, s_n2=1e-3, tau=1):
    s_f2 = 1-s_n2
    T = sdf.shape[0]
    ts = torch.arange(start=1, end=T+1, step=1)
    channels = sdf.shape[-1]
    K = torch.zeros((channels, T, T))
    for i in range(channels):
        for idx, t in enumerate(ts):
            K1 = s_f2 * torch.exp(-(t-ts)**2/(2*tau**2))
            K2 = s_n2 * torch.Tensor(t == ts)
            K[i, idx] = K1 + K2
        print(K[i])
    print(K)
            # print(K1 + K2)

def t_test(sdf1, sdf2, q=0.025, paired = False):
    n1, n2 = sdf1.shape[1], sdf2.shape[1]
    m1, m2 = np.mean(sdf1, axis=1), np.mean(sdf2, axis=1)
    s1, s2 = np.std(sdf1, axis=1),  np.std(sdf2, axis=1)
    if paired:
        s_pop = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        s_mean = s_pop*np.sqrt(1/n1+1/n2)
        t_vals = (m1-m2)/s_mean
        df = np.ones_like(m1)*(n1+n2-2)
    else:
        v1, v2 = s1**2/n1, s2**2/n2
        # if v1.min()==0 or v2.min() ==0:
        #     print(f'V1 min: {v1.min()}, V2 min:{v2.min()}')
        t_vals = (m1-m2)/np.sqrt(v1 + v2)
        df = (v1+v2)**2/(1/(n1-1) *v1**2 + 1/(n2-1)*v2**2)
    modulation = np.zeros(len(m1), dtype=bool)
    for neuron_idx in range(len(m1)):
        # lower_tail = t.cdf(t_vals[neuron_idx], df[neuron_idx])
        sig_level = t.sf(abs(t_vals[neuron_idx]), df[neuron_idx])
        # modulation[neuron_idx] = (lower_tail < q) or (upper_tail <q)
        modulation[neuron_idx] = sig_level < q
    return modulation, t_vals
