import numpy as np
import pyaudio
import scipy.signal as signal
from dataclasses import dataclass
from typing import List, Optional
import random


@dataclass
class ResonancePattern:
    frequency: float
    amplitude: float
    phase: float
    duration: float


class ResonanceSpace:
    def __init__(
        self, sample_rate: int = 44100, buffer_size: int = 1024, memory_depth: int = 10
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.memory_depth = memory_depth
        self.memory: List[np.ndarray] = []
        self.resonance_patterns: List[ResonancePattern] = []

    def process_input(self, audio_buffer: np.ndarray) -> np.ndarray:
        # Zorg dat we werken met dubbelprecisie voor FFT stabiliteit
        audio_buffer = audio_buffer.astype(np.float64)

        # Update geheugen
        self.memory.append(audio_buffer)
        if len(self.memory) > self.memory_depth:
            self.memory.pop(0)

        # Windowing om spectral leakage te verminderen
        window = np.hanning(len(audio_buffer))
        windowed_signal = audio_buffer * window

        # Analyseer frequenties
        freqs = np.fft.fftfreq(len(windowed_signal), 1 / self.sample_rate)
        spectrum = np.fft.fft(windowed_signal)

        # Creëer nieuwe resonantie patronen
        dominant_freqs = self._find_dominant_frequencies(spectrum, freqs)
        self._generate_resonance_patterns(dominant_freqs)

        # Genereer response
        response = self._create_response()
        return response.astype(np.float32)

    def _find_dominant_frequencies(
        self, spectrum: np.ndarray, freqs: np.ndarray, num_peaks: int = 3
    ) -> List[float]:
        # Alleen positieve frequenties beschouwen
        magnitude = np.abs(spectrum)
        half = len(magnitude) // 2
        pos_magnitude = magnitude[:half]
        pos_freqs = freqs[:half]

        peaks = signal.find_peaks(pos_magnitude)[0]

        if len(peaks) == 0:
            return [random.uniform(200, 2000) for _ in range(num_peaks)]

        # Sorteer peaks op magnitude en neem de top
        sorted_peaks = sorted(peaks, key=lambda p: pos_magnitude[p], reverse=True)
        top_peaks = sorted_peaks[:num_peaks]

        # Voeg subtiele random variatie toe, niet alleen door random selectie
        result_freqs = []
        for peak in top_peaks:
            base_freq = pos_freqs[peak]
            result_freqs.append(base_freq * (1 + random.uniform(-0.05, 0.05)))

        return result_freqs

    def _generate_resonance_patterns(self, frequencies: List[float]):
        new_patterns = []
        for freq in frequencies:
            freq_variation = freq * (1 + random.uniform(-0.1, 0.1))
            amplitude = random.uniform(0.3, 0.8)
            phase = random.uniform(0, 2 * np.pi)
            duration = random.uniform(0.5, 2.0)

            new_patterns.append(
                ResonancePattern(
                    frequency=freq_variation,
                    amplitude=amplitude,
                    phase=phase,
                    duration=duration,
                )
            )

        self.resonance_patterns.extend(new_patterns)

        # Beperk aantal patronen
        if len(self.resonance_patterns) > 5:
            self.resonance_patterns = self.resonance_patterns[-5:]

    def _create_response(self) -> np.ndarray:
        t = np.linspace(
            0, self.buffer_size / self.sample_rate, self.buffer_size, endpoint=False
        )
        response = np.zeros_like(t)

        for pattern in self.resonance_patterns:
            wave = pattern.amplitude * np.sin(
                2 * np.pi * pattern.frequency * t + pattern.phase
            )
            # Envelope
            wave *= np.exp(-t / pattern.duration)
            # Harmonischen toevoegen
            wave += 0.1 * np.sin(3 * 2 * np.pi * pattern.frequency * t)

            response += wave

        max_val = np.max(np.abs(response))
        if max_val > 1e-9:
            response /= max_val

        return response


class AudioInterface:
    def __init__(self, resonance_space: ResonanceSpace):
        self.resonance_space = resonance_space
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_stream(self):
        def callback(in_data, frame_count, time_info, status):
            audio_buffer = np.frombuffer(in_data, dtype=np.float32)
            response = self.resonance_space.process_input(audio_buffer)
            out_data = response.tobytes()
            return (out_data, pyaudio.paContinue)

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.resonance_space.sample_rate,
            input=True,
            output=True,
            frames_per_buffer=self.resonance_space.buffer_size,
            stream_callback=callback,
        )

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


if __name__ == "__main__":
    space = ResonanceSpace()
    interface = AudioInterface(space)
    print("Starting resonance space...")
    interface.start_stream()

    try:
        input("Press Enter to stop...")
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing resonance space...")
        interface.stop_stream()
