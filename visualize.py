import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import mido
import csv
import math

# DCS_LI_FullChoir_Take02_Stereo_STM.wav
# D:\CV\DagstuhlChoirSet_V1.2.3\audio_wav_22050_mono\DCS_LI_FullChoir_Solo05_A2_DYN.wav
# DCS_LI_FullChoir_Solo01_B2_DYN
# DCS_LI_FullChoir_Solo03_T2_DYN
# DCS_LI_FullChoir_Solo07_S1_DYN


def freq2pitch(freq):
    return int(round(math.log(freq / (440 / 32), 2) * 12 + 9))

def pitch2freq(pitch):
    return 440 / 32 * math.pow(2, (pitch - 9) / 12)


def read_wav_STFT(file_path):
    # 读取16bit整数wav
    fs, sig = wavfile.read(file_path)
    print(f"sampling rate = {fs} Hz, data shape {sig.shape}, channels = 1")

    # perform STFT
    fr_dis = 5
    freq_stamp, time_stamp, Zxx = signal.stft(sig, fs, window='hann', nperseg=fs//fr_dis)  # sig[0:10*fs]
    Zxx = np.abs(Zxx)
    print('STFT result shapes', freq_stamp.shape, time_stamp.shape, Zxx.shape)
    return freq_stamp, time_stamp, Zxx


def midi2note(file_path):   # deprecated
    mid = mido.MidiFile(file_path)
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)

OUTPUT_TIMESTEP = 0.011609977324263039

def read_groundtruth_csv(file_paths, clip_start, clip_end):
    '''format: onset, freq, duration'''
    # 音高取40-80
    time_stamp = np.arange(clip_start, clip_end + 1) * OUTPUT_TIMESTEP # align with output
    pitch_map = np.zeros((40, clip_end - clip_start), dtype=np.int8)

    if type(file_paths) is str:
        file_paths = [file_paths]
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')

            for line in reader:
                #print(line)
                onset, freq, duration = float(line[0]), float(line[1]), float(line[2])
                l = max(int(round(onset / OUTPUT_TIMESTEP)), clip_start)
                r = min(int(round((onset + duration) / OUTPUT_TIMESTEP)), clip_end)
                pitch = freq2pitch(np.abs(float(freq)))
                if l < r and 40 <= pitch and pitch < 80:
                    pitch_map[pitch - 40, l:r] = 1

    return time_stamp, pitch_map


def read_output_csv(file_paths, clip_start, clip_end):
    '''format: time\t freq1\t freq2\t freq3\t...'''
    # 音高取40-80
    time_stamp = np.empty((clip_end - clip_start + 1,))
    pitch_map = np.zeros((40, clip_end - clip_start), dtype=np.int8)

    if type(file_paths) is str:
        file_paths = [file_paths]
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter='\t')

            for i, line in enumerate(reader):
                #print(line)
                if i < clip_start: continue
                time_stamp[i - clip_start] = float(line[0])
                if i >= clip_end: break
                for freq_s in line[1:]:
                    pitch = freq2pitch(np.abs(float(freq_s)))
                    if 40 <= pitch and pitch < 80:
                        pitch_map[pitch - 40, i - clip_start] = 1

    return time_stamp, pitch_map


def visualize(name, title, freq_stamp, time_stamp, value, ylim, xlim=None, vmax=None, cmap='viridis'):
    '''cmap recommand: 'gnuplot' 'inferno' 'plasma' 'viridis' 'magma' '''
    f1 = plt.figure(1)
    #plt.pcolormesh(time_stamp, freq_stamp, Zxx*10000, cmap=CMAP, vmax=300, alpha=0.6)
    plt.pcolormesh(time_stamp, freq_stamp, value, vmax=vmax, cmap=cmap)
    #plt.colorbar()
    if xlim: plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(title)
    plt.tight_layout(h_pad=1.12)
    plt.savefig(name)
    #plt.show()
    f1.clear()


if __name__ == '__main__':
    name = 'DCS_LI_FullChoir_Take02'
    clip_l, clip_r = 8, 1064

    freq_stamp, time_stamp, Zxx = read_wav_STFT(name+'_Stereo_STM.wav')
    visualize('input.png', 'input', freq_stamp, time_stamp, Zxx*10000, (50, 800), (clip_l * OUTPUT_TIMESTEP, clip_r * OUTPUT_TIMESTEP), 200, 'gnuplot')

    csv_time_stamp, pitch_map = read_groundtruth_csv([f'{name}_{part}_DYN.csv' for part in ('A2','B2','S1','T2')], clip_l, clip_r)
    visualize('ground_truth.png', 'ground truth', np.arange(40, 81), csv_time_stamp, pitch_map, (40, 80))

    for i in range(1, 4):
        csv_time_stamp, pitch_map = read_output_csv(f'{name}_Stereo_STM_{i}.csv', clip_l, clip_r)
        model_name = ('Early/Deep', 'Early/Shallow', 'Late/Deep')[i-1]
        visualize(f'output{i}.png', model_name, np.arange(40, 81), csv_time_stamp, pitch_map, (40, 80))

    #for i in range(128):
    #    print(i, pitch2frequency(i))