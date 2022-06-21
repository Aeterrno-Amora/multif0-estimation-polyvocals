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


def read_wav_STFT(file_path):
    # 读取16bit整数wav
    fs, sig = wavfile.read(file_path)
    print(sig.shape)
    print("sampling rate = {} Hz, length = {} samples, channels = 1".format(fs, *sig.shape))
    print(sig)
    f1 = plt.figure(1)
    #plt.plot(sig)

    fr_dis = 5
    f, t, Zxx = signal.stft(sig, fs, window='hann', nperseg=fs//fr_dis)  # sig[0:10*fs]
    print(Zxx.shape)
    print(f.shape)
    print(t.shape)
    Zxx = np.abs(Zxx)
    #plt.pcolormesh(t, f, Zxx*10000, cmap='PRGn', vmax=300, alpha=0.6)
    #plt.colorbar()
    #plt.tight_layout()
    #plt.ylim(50, 800)

    #plt.show()
    return f, t, Zxx


def midi2note(file_path):
    mid = mido.MidiFile(file_path)
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)


def pitch2frequency(pitch):
    standard_frequency = 440.0
    fre = (standard_frequency / 32.0) * math.pow(2, (pitch - 9.0) / 12.0)
    return fre


def csv2note(file_path, label, frequency_stamp, time_stamp, max_map_value, min_map_value):
    # CSV score representation
    # 3 columns: note onset (in sec) + note offset (in sec) + MIDI pitch
    # number of rows == number of notes in the piece
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        #result = list(reader)
        for row in reader:
            print(row)

    # 音高取40-80
    pitch_map = np.ones((40, len(time_stamp))) * min_map_value
    #print(np.shape(result)[0])
    for i in range(np.shape(result)[0]):
        line = result[i]
        #print(line)
        if len(np.shape(line)) > 1:
            if np.shape(line)[1] < 2:
                continue
            num_parts = np.shape(line)[1] - 1
        else:
            continue
        #onset, offset, pitch = float(line[0]), float(line[1]), int(line[2])
        onset = line[0]
        frequencies = line[1:]
        onset = math.ceil(onset * 10) - 1
        # 只截取对应wav的一段
        if onset >= len(time_stamp):
            break
        #offset = math.ceil(offset * 10) - 1
        #fre = pitch2frequency(pitch)
        #fre = math.ceil(fre / 5) - 1
        for time in range(onset, onset+1):
            # 只截取对应wav的一段
            if time >= len(time_stamp):
                break
            for frequency in frequencies:
                pitch = int(round(math.log(frequency*32/440.0, 2)*12+9))
                print(pitch)
                pitch_map[pitch - 40][time] = max_map_value
    pitch_stamp = np.arange(40, 80)
    plt.pcolormesh(time_stamp, pitch_stamp, np.abs(pitch_map), cmap='inferno', vmax=300, alpha=0.6)
    plt.colorbar()
    plt.tight_layout()
    plt.ylim(40, 80)
    plt.title(label)
    plt.show()

    return pitch_map


if __name__ == '__main__':
    f1, t1, Zxx1 = read_wav_STFT(r'D:\CV\DagstuhlChoirSet_V1.2.3\audio_wav_22050_mono\DCS_LI_FullChoir_Take01_Stereo_STM.wav')
    csv2note(r'C:\Users\hcyuan\Desktop\DL project\multif0-estimation-polyvocals-master\experiments\DCS_LI_FullChoir_Take02_Stereo_STM.csv',
        'S', f1, t1, 300, -50)
    #for i in range(128):
    #    print(i, pitch2frequency(i))