from mne_lsl.lsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_streams,
)
import numpy as np
import matplotlib.pyplot as plt

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_streams()

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
# need to open stream to get info and data
inlet.open_stream()
# get stats about the data stream
sinfo = inlet.get_sinfo()
print(sinfo)

# arrays for each channel
ch1=[]
ch2=[]
timestamps=[]
for i in range(0,100):
    # get a new sample
    sample, timestamp = inlet.pull_sample()
    print(sample)

    #Sometimes the sample is none, tends to happen if connection is lost and re-established
    if type(sample) != None:
        ch1.append(sample[0])
        ch2.append(sample[1])
        timestamps.append(timestamp)
    else:
        print("aaaaaa none type :(")

# free up resources
inlet.close_stream()
del inlet

#plot what we collected
def singlechannelgraph(sf,chdata,chno):
# Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(timestamps, chdata, lw=1.5, color='k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    #plt.xlim([timestamps.min(), timestamps.max()])
    plt.title('Channel %d EEG data'%(chno))
    plt.show()

singlechannelgraph(250, ch1, 1)

## don't worry about the rest of this, it's just here for future reference
def singlechannelPSD(channeldata,sf):
    from scipy import signal

# Define window length (0.5 seconds)
    win = 0.5 * sf
    freqs, psd = signal.welch(channeldata, sf, nperseg=win)
    p = (np.fft.rfft(channeldata))

    f = np.linspace(0, 512/2, len(p))
   # print(freqs)
    #print(f)
    #plt.plot(f,p)
# Plot the power spectrum
#sns.set(font_scale=1.2, style='white')
#    plt.figure(figsize=(8, 4))
#    plt.plot(freqs, psd, color='k', lw=2)
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Power spectral density (V^2 / Hz)')
#    plt.ylim([0, psd.max() * 1.1])
#    plt.title("Welch's periodogram")
#    plt.xlim([0, freqs.max()])
    return freqs,psd,f,p
#sns.despine()
def Bandspecs_getidx_delta(lowb,highb,freqs,psd):
    low, high = lowb, highb

# Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)

# Plot the power spectral density and fill the delta area
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, psd, lw=2, color='k')
    plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (uV^2 / Hz)')
    plt.xlim([0, 40])
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    return idx_delta
#sns.despine()
from scipy.integrate import simps
# Frequency resolution

def deltapower(idx_delta1,freqs,psd1,f,p):
    freq_res = freqs[1] - freqs[0]  # = 1 / 0.5 = 2
    #print(f)
    fr_res = f[1] - f[0]
# Compute the absolute power by approximating the area under the curve
    delta_power = simps(p[idx_delta1], dx=fr_res)
    #print('Absolute delta power: %.3f uV^2' % delta_power)
    total_power = simps(p, dx=fr_res)
    delta_rel_power = delta_power / total_power

    #print('Relative delta power: %.3f' % delta_rel_power)
    return delta_power
def relpower(idx_delta,freqs,psd,f,p):
    freq_res = freqs[1] - freqs[0]  # = 1 / 0.5 = 2
    fr_res = f[1] - f[0]
# Compute the absolute power by approximating the area under the curve
    delta_power = simps(p[idx_delta], dx=fr_res)
    #print('Absolute delta power: %.3f uV^2' % delta_power)
    total_power = simps(p, dx=fr_res)
    delta_rel_power = delta_power / total_power
    #print('Relative delta power: %.3f' % delta_rel_power)
    return delta_rel_power
def _main(nc,sf):
    #nc=4
    #sf=512
    chan=[]
    for m in range(0,nc):
        chan.append(channel(m))
    for n in range(0,nc):
        singlechannelgraph(sf,chan[n],n)

    freqs_all=[]
    psd_all=[]
    f_all=[]
    p_all=[]
    for o in range(0,nc):
        freqs,psd,f,p=singlechannelPSD(chan[o],sf)
        freqs_all.append(freqs)
        psd_all.append(psd)
        f_all.append(f)
        p_all.append(p)
    idx_delta=[]
#for theta 4-7hz
    idx_delta.append(Bandspecs_getidx_delta(4,7,freqs_all[0],psd_all[0]))
#for alpha 8-13hz
    idx_delta.append(Bandspecs_getidx_delta(8,13,freqs_all[1],psd_all[1]))
#for beta 13-30hz
    idx_delta.append(Bandspecs_getidx_delta(13,30,freqs_all[2],psd_all[2]))
    abspower_sec=[]
    for p1 in range(0,nc):
        abspower_prim=[]
        for q in range(0,3):
            abspower_prim.append((deltapower(idx_delta[q],freqs_all[p1],psd_all[p1],f_all[p1],p_all[p1])).real)
        abspower_sec.append(abspower_prim)
    #print (len(abspower_sec),'x',int(len(abspower_prim)))
    relpower_sec=[]
    for p1 in range(0,nc):
        relpower_prim=[]
        for q in range(0,3):
            relpower_prim.append((relpower(idx_delta[q],freqs_all[p1],psd_all[p1],f_all[p1],p_all[p1])).real
                                )
        #print('insideq',relpower_prim)

        relpower_sec.append(relpower_prim)
    print(relpower_sec)
    #print('inside-----p')
    return relpower_sec