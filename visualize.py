import PySimpleGUI as sg
import pyaudio
import numpy as np
import time
from pydub import AudioSegment

from utils import QUERY_DURATION
from inferencer import Inferencer, CONTEXT_WINDOW_SIZE


USE_PODCAST = True   # if this is false use mic

# INIT vars:
CHUNK = 512  # Samples: 1024,  512, 256, 128
RATE = 44100  # Equivalent to Human Hearing at 40 kHz
INTERVAL = 1  # Sampling Interval in Seconds ie Interval to listen
TIMEOUT = 10  # In ms for the event loop
FORMAT = pyaudio.paInt16
CHANNELS = 1
COLOR = 'blue'

inf = Inferencer(RATE)

def detect_speaker_change():
    global COLOR
    ret = False
    if len(_VARS['context']) >= CONTEXT_WINDOW_SIZE * RATE // 2:
        if np.var(_VARS['curr']) < 0:
            ret = False
        else:
            ret = inf.infer(_VARS['context'], _VARS['curr'])
            # _VARS['responseWindow'].append(b)
            # _VARS['responseTimes'].append(time.perf_counter())
            # if _VARS['responseTimes'][-1] - _VARS['responseTimes'][0] > CONTEXT_WINDOW_SIZE / 1.5:   # ensure we are only looking at window of interest
            #     tf = _VARS['responseTimes'][-1] - CONTEXT_WINDOW_SIZE / 1.5
            #     for i, t in enumerate(_VARS['responseTimes']): 
            #         if t > tf: break
            #     _VARS['responseTimes'] = _VARS['responseTimes'][i:]
            #     _VARS['responseWindow'] = _VARS['responseWindow'][i:]
    COLOR = 'red' if ret else 'blue'
    return ret



""" RealTime Audio Waveform plot """

# VARS CONSTS:
_VARS = {'window': False,
         'stream': False,
         'audioData': np.array([]),
         'curr': np.array([]),
         'context': np.array([]),
         'responseWindow': [0],
         'responseTimes': [time.perf_counter()]}

# pysimpleGUI INIT:
AppFont = 'Any 16'
sg.theme('DarkBlue3')
layout = [[sg.Graph(canvas_size=(500, 500),
                    graph_bottom_left=(-2, -2),
                    graph_top_right=(102, 102),
                    background_color='#809AB6',
                    key='graph')],
          [sg.ProgressBar(4000, orientation='h',
                          size=(20, 20), key='-PROG-')],
          [sg.Button('Listen', font=AppFont),
           sg.Button('Stop', font=AppFont, disabled=True),
           sg.Button('Exit', font=AppFont)]]
_VARS['window'] = sg.Window('Mic to waveform plot + Max Level',
                            layout, finalize=True)

graph = _VARS['window']['graph']


pAud = pyaudio.PyAudio()

# FUNCTIONS:

# PYSIMPLEGUI PLOTS


def drawAxis(dataRangeMin=0, dataRangeMax=100):
    # Y Axis
    graph.DrawLine((0, 50), (100, 50))
    # X Axis
    graph.DrawLine((0, dataRangeMin), (0, dataRangeMax))

# PYAUDIO STREAM
def stop():
    if _VARS['stream']:
        _VARS['stream'].stop_stream()
        _VARS['stream'].close()
        _VARS['window']['-PROG-'].update(0)
        _VARS['window'].FindElement('Stop').Update(disabled=True)
        _VARS['window'].FindElement('Listen').Update(disabled=False)


def callback(in_data, frame_count, time_info, status):
    _VARS['audioData'] = np.frombuffer(in_data, dtype=np.int16)
    _VARS['curr'] = np.append(_VARS['curr'], (_VARS['audioData'] / 2 ** 15).astype(float))
    if len(_VARS['curr']) > QUERY_DURATION * RATE: 
        extra, _VARS['curr'] = _VARS['curr'][:-int(QUERY_DURATION * RATE)], _VARS['curr'][-int(QUERY_DURATION * RATE):]
        _VARS['context'] = np.append(_VARS['context'], extra)
    if len(_VARS['context']) > CONTEXT_WINDOW_SIZE * RATE:
        _VARS['context'] = _VARS['context'][-int(CONTEXT_WINDOW_SIZE * RATE):]
    changed = detect_speaker_change()
    print(changed)
    return (in_data, pyaudio.paContinue)


def listen():
    _VARS['window'].FindElement('Stop').Update(disabled=False)
    _VARS['window'].FindElement('Listen').Update(disabled=True)
    _VARS['stream'] = pAud.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=not USE_PODCAST,
                                output=USE_PODCAST,
                                output_device_index=1 if USE_PODCAST else None,  # 3 for speakers, 1 for airpods
                                frames_per_buffer=CHUNK,
                                stream_callback=callback if not USE_PODCAST else None)
    if not USE_PODCAST:
        _VARS['stream'].start_stream()


if USE_PODCAST:
    pd = AudioSegment.from_file('/Users/evandigiorno/Downloads/download.mp3')
    FORMAT = pyaudio.get_format_from_width(pd.sample_width)
    CHANNELS = pd.channels
    RATE = pd.frame_rate
    data = pd[:CHUNK]._data
    idx = 0
    
drawAxis()


# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=TIMEOUT)
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        stop()
        pAud.terminate()
        break
    if USE_PODCAST and type(_VARS['stream']) != bool:
        if data:
            _VARS['stream'].write(data)
            idx += CHUNK
            data = pd[idx:idx + CHUNK]._data
            callback(data, None, None, None)
        else:
            break
    
    if event == 'Listen':
        listen()
    if event == 'Stop':
        stop()


    # Along with the global audioData variable, this\
    # bit updates the waveform plot, left it here for
    # explanatory purposes, but could be a method.

    # elif _VARS['audioData'].size != 0:
    #     # Uodate volume meter
    #     _VARS['window']['-PROG-'].update(np.amax(_VARS['audioData']))
    #     # Redraw plot
    #     graph.erase()
    #     drawAxis()
        

    #     # Here we go through the points in the audioData object and draw them
    #     # Note that we are rescaling ( dividing by 100 ) and centering (+50 )
    #     # try different values to get a feel for what they do.          
    #     for x in range(CHUNK):
    #         graph.DrawCircle((x, (_VARS['audioData'][x]/100)+50), 0.4,
    #                          line_color=COLOR, fill_color=COLOR)



_VARS['window'].close()