"""
Cortex notifications — morse code beeps + macOS speech.

Audio feedback for system events using only macOS built-ins:
  - say(text) — TTS via macOS `say` command
  - morse_beep(text) — morse code via synthesized audio tones
  - notify(title, body) — macOS notification center
  - boot_announce() — full boot sequence announcement

No external dependencies. Uses `say`, `osascript`, and `afplay` with
dynamically generated WAV files for morse tones.
"""

import os
import struct
import subprocess
import math
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Morse code table
# ---------------------------------------------------------------------------

MORSE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',
    'E': '.',     'F': '..-.',  'G': '--.',   'H': '....',
    'I': '..',    'J': '.---',  'K': '-.-',   'L': '.-..',
    'M': '--',    'N': '-.',    'O': '---',   'P': '.--.',
    'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',
    'Y': '-.--',  'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', ' ': ' ',
}

# Timing (seconds)
DIT_DURATION = 0.08       # dot length
DAH_DURATION = DIT_DURATION * 3  # dash = 3x dot
INTRA_GAP = DIT_DURATION  # gap between dits/dahs within a character
CHAR_GAP = DIT_DURATION * 3     # gap between characters
WORD_GAP = DIT_DURATION * 7     # gap between words

# Tone
FREQUENCY = 700  # Hz — classic CW frequency
SAMPLE_RATE = 44100
AMPLITUDE = 0.5


# ---------------------------------------------------------------------------
# WAV tone generation (pure Python, no deps)
# ---------------------------------------------------------------------------

def _generate_tone(duration_s: float, freq: float = FREQUENCY) -> bytes:
    """Generate raw PCM samples for a sine wave tone."""
    n_samples = int(SAMPLE_RATE * duration_s)
    samples = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        value = AMPLITUDE * math.sin(2 * math.pi * freq * t)
        # Apply short fade in/out to avoid clicks (5ms)
        fade_samples = int(0.005 * SAMPLE_RATE)
        if i < fade_samples:
            value *= i / fade_samples
        elif i > n_samples - fade_samples:
            value *= (n_samples - i) / fade_samples
        samples.append(int(value * 32767))
    return struct.pack(f'<{len(samples)}h', *samples)


def _generate_silence(duration_s: float) -> bytes:
    """Generate raw PCM silence."""
    n_samples = int(SAMPLE_RATE * duration_s)
    return b'\x00\x00' * n_samples


def _pcm_to_wav(pcm_data: bytes) -> bytes:
    """Wrap raw 16-bit PCM in a WAV header."""
    n_channels = 1
    sample_width = 2  # 16-bit
    data_size = len(pcm_data)
    
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    fmt = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, n_channels,
                      SAMPLE_RATE, SAMPLE_RATE * n_channels * sample_width,
                      n_channels * sample_width, sample_width * 8)
    data_header = struct.pack('<4sI', b'data', data_size)
    
    return header + fmt + data_header + pcm_data


def text_to_morse_wav(text: str) -> bytes:
    """Convert text to a WAV file of morse code beeps."""
    pcm = b''
    
    for char in text.upper():
        if char == ' ':
            pcm += _generate_silence(WORD_GAP)
            continue
            
        code = MORSE.get(char)
        if not code:
            continue
            
        for i, symbol in enumerate(code):
            if symbol == '.':
                pcm += _generate_tone(DIT_DURATION)
            elif symbol == '-':
                pcm += _generate_tone(DAH_DURATION)
            
            # Gap between symbols (not after last)
            if i < len(code) - 1:
                pcm += _generate_silence(INTRA_GAP)
        
        # Gap between characters
        pcm += _generate_silence(CHAR_GAP)
    
    return _pcm_to_wav(pcm)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def morse_beep(text: str, blocking: bool = True) -> None:
    """
    Play morse code for the given text via macOS audio.
    
    Uses synthesized WAV tones played through afplay.
    No external dependencies.
    """
    wav_data = text_to_morse_wav(text)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(wav_data)
        wav_path = f.name
    
    try:
        proc = subprocess.Popen(
            ['afplay', wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if blocking:
            proc.wait()
    finally:
        if blocking:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def speak(text: str, blocking: bool = False) -> None:
    """Speak text using macOS built-in TTS."""
    proc = subprocess.Popen(
        ['say', text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if blocking:
        proc.wait()


def notify(title: str, body: str) -> None:
    """Send a macOS notification center alert."""
    script = f'display notification "{body}" with title "{title}"'
    subprocess.Popen(
        ['osascript', '-e', script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def boot_announce(models_loaded: int = 0, max_tier: str = "?") -> None:
    """
    Full boot announcement sequence:
    1. macOS notification
    2. Speech: "Cortex online"  
    3. Morse: "CX" (short identifier)
    """
    notify("Cortex", f"Daemon online — {models_loaded} models, max tier {max_tier}")
    speak(f"Cortex online. {models_loaded} models loaded. Max tier {max_tier}.", blocking=True)
    morse_beep("CX")


def event_announce(event: str) -> None:
    """
    Announce a system event with appropriate audio.
    
    Events:
      boot     — full boot sequence (speech + morse CX)
      request  — short blip (single dit)
      escalate — warning tone (morse E = single dit, higher pitch)
      error    — SOS in morse
      shutdown — morse SK (end of transmission)
    """
    if event == "boot":
        speak("Cortex online", blocking=True)
        morse_beep("CX")
    elif event == "request":
        # Single short blip — quiet feedback
        wav = _pcm_to_wav(_generate_tone(DIT_DURATION, FREQUENCY))
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav)
            subprocess.Popen(['afplay', f.name],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    elif event == "escalate":
        morse_beep("E", blocking=False)  # single dit = escalation
    elif event == "error":
        morse_beep("SOS")
    elif event == "shutdown":
        speak("Cortex shutting down")
        morse_beep("SK")  # ham radio "end of contact"
    elif event == "swarm":
        morse_beep("SW", blocking=False)
    elif event == "challenge":
        morse_beep("CH", blocking=False)


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.notify morse 'HELLO WORLD'")
        print("  python -m src.notify speak 'Cortex is alive'")
        print("  python -m src.notify boot")
        print("  python -m src.notify event boot|request|escalate|error|shutdown")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "morse":
        text = sys.argv[2] if len(sys.argv) > 2 else "CX"
        print(f"Playing morse: {text}")
        for c in text.upper():
            code = MORSE.get(c, '?')
            print(f"  {c} = {code}")
        morse_beep(text)
    elif cmd == "speak":
        text = sys.argv[2] if len(sys.argv) > 2 else "Cortex online"
        speak(text, blocking=True)
    elif cmd == "boot":
        boot_announce(models_loaded=3, max_tier="L4")
    elif cmd == "event":
        event = sys.argv[2] if len(sys.argv) > 2 else "boot"
        event_announce(event)
    elif cmd == "notify":
        notify("Cortex", sys.argv[2] if len(sys.argv) > 2 else "Test notification")
    else:
        print(f"Unknown command: {cmd}")
