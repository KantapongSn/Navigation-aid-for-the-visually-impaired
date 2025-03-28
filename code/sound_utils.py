
import numpy as np
import sounddevice as sd

def blink_sound(duration, frequency, fs=43200):
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    sound_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    sd.play(sound_wave, fs)
    sd.wait()
'''
import numpy as np
import sounddevice as sd

def blink_sound(duration, start_frequency, end_frequency, fs=44100):
    """
    สร้างเสียงที่เปลี่ยนความถี่จาก start_frequency ไปยัง end_frequency แบบ fade
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    # สร้างความถี่แบบ fade
    frequencies = np.linspace(start_frequency, end_frequency, t.shape[0])
    # สร้างคลื่นเสียง
    sound_wave = 0.5 * np.sin(2 * np.pi * frequencies * t)
    # เล่นเสียง
    sd.play(sound_wave, fs)
    sd.wait()

import threading
import numpy as np
import sounddevice as sd

def blink_sound(duration, frequency, fs):
    """
    Generate a sound for the specified duration and frequency using sounddevice in a separate thread.
    """
    def sound_thread():
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        sd.play(wave, samplerate=fs)
        sd.wait()

    # Run the sound generation in a separate thread
    thread = threading.Thread(target=sound_thread)
    thread.start()
'''
