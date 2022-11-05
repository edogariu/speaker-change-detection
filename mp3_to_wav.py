import pydub
import numpy as np
import pandas as pd

def read_mp3(path):
    """
    mp3 to numpy array
    """
    a = pydub.AudioSegment.from_mp3(f'data/clips/{path}')
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    return y, a.frame_rate
    
def write_wav(path, sr, x):
    """numpy array to wav"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    song = pydub.AudioSegment(x.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f'data/wavs/{path}', format="wav", bitrate="320k")
  
import multiprocessing; import tqdm; import time

def init_globals(counter, wav_counter):
    global count, wav_count
    count = counter
    wav_count = wav_counter

def thing(arr, files_to_ids):
    for d in arr:
        sid, p = d[1:3]
        y, sr = read_mp3(p)
        x = np.split(y, np.arange(0, len(y), sr))[1:-1]   # only full second intervals
        for a in x:
            with wav_count.get_lock():
                name = f'{wav_count.value}.wav'
                wav_count.value += 1
            write_wav(name, sr, a)
            files_to_ids[name] = sid
    with count.get_lock():
        count.value += 1

if __name__ == '__main__':
    df = pd.read_csv('data/dataset.csv')
    ncpu = multiprocessing.cpu_count()
        
    counter = multiprocessing.Value('i', 0)  # count of how many processes were finished
    wav_counter = multiprocessing.Value('i', 0)  # count of how many processes were finished
    files_to_ids = multiprocessing.Manager().dict()

    L = 200
    input_idxs = np.arange(0, len(df), L)
    inputs = np.split(df.values, input_idxs)[1:]
    n = len(inputs)
    
    print('prepping inputs')
    for i in range(n):
        inputs[i] = (inputs[i], files_to_ids)

    print('Spawning {} workers for {} jobs, each of size {}'.format(ncpu, n, L))
    with multiprocessing.Pool(ncpu, initializer=init_globals, initargs=(counter, wav_counter)) as p:
        p.starmap_async(thing, inputs[:n])
        prev_counter = counter.value
        with tqdm.tqdm(total=n) as pbar:
            while counter.value < n:
                if counter.value != prev_counter:
                    prev_counter = counter.value
                    pbar.update(1)
                else:
                    time.sleep(0.01)
        p.close()
        p.join()
        
    files_to_ids = dict(files_to_ids)
    import pickle
    pickle.dump(files_to_ids, open('dict.pkl', 'wb'))
