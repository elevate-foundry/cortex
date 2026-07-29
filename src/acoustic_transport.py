"""
Acoustic Gossip Transport — air-gapped state sync via sound.

Encodes SCL deltas as audio tones and decodes them back. Two Cortex
nodes within earshot can gossip without any network connection.

Encoding: Binary FSK (Frequency-Shift Keying)
  - Bit 0 → 1200 Hz tone (mark)
  - Bit 1 → 2400 Hz tone (space)
  - Baud rate: 300 baud (300 bits/sec)
  - Preamble: 800 Hz sync tone (0.3s) + 600 Hz start tone (0.1s)
  - Framing: [preamble][length:2 bytes][payload][crc16:2 bytes]

A 4-byte Braille fingerprint transmits in ~0.2s of data + 0.4s preamble.
A typical delta (50–200 bytes) transmits in 2–6 seconds.

This is intentionally simple. No error correction beyond CRC-16.
No compression. No handshake. One-way broadcast, like a radio beacon.

Protocol:
  1. Sender encodes SCL delta → bytes → FSK audio → play via speaker
  2. Receiver records via microphone → detect preamble → demodulate FSK
     → bytes → verify CRC → parse SCL delta → apply to local state

Dependencies: None beyond stdlib (uses pyaudio only for microphone input,
falls back to macOS `rec` command if unavailable).

Zero-dependency encoding/playback via afplay (macOS) or aplay (Linux).
"""

import math
import struct
import subprocess
import tempfile
import os
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# FSK parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
BAUD_RATE = 300           # bits per second
FREQ_MARK = 1200          # bit 0
FREQ_SPACE = 2400         # bit 1
FREQ_PREAMBLE = 800       # sync tone
FREQ_START = 600          # start-of-frame marker
AMPLITUDE = 0.8

PREAMBLE_DURATION = 0.3   # seconds of sync tone
START_DURATION = 0.1      # seconds of start marker
BIT_DURATION = 1.0 / BAUD_RATE  # seconds per bit

# Derived
SAMPLES_PER_BIT = int(SAMPLE_RATE * BIT_DURATION)


# ---------------------------------------------------------------------------
# CRC-16 (CCITT) — simple integrity check
# ---------------------------------------------------------------------------

def crc16(data: bytes) -> int:
    """CRC-16-CCITT with polynomial 0x1021."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# WAV generation (pure Python)
# ---------------------------------------------------------------------------

def _tone_samples(freq: float, duration: float, amplitude: float = AMPLITUDE, fade_ms: float = 0.0) -> list[int]:
    """Generate 16-bit PCM samples for a sine wave."""
    n = int(SAMPLE_RATE * duration)
    samples = []
    fade = int(fade_ms * 0.001 * SAMPLE_RATE) if fade_ms > 0 else 0
    for i in range(n):
        t = i / SAMPLE_RATE
        value = amplitude * math.sin(2 * math.pi * freq * t)
        if fade > 0:
            if i < fade:
                value *= i / fade
            elif i > n - fade:
                value *= (n - i) / fade
        samples.append(int(value * 32767))
    return samples


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV header."""
    n_channels = 1
    sample_width = 2
    data_size = len(pcm)
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    fmt = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, n_channels,
                      SAMPLE_RATE, SAMPLE_RATE * n_channels * sample_width,
                      n_channels * sample_width, sample_width * 8)
    data_hdr = struct.pack('<4sI', b'data', data_size)
    return header + fmt + data_hdr + pcm


# ---------------------------------------------------------------------------
# Encoder: bytes → FSK audio
# ---------------------------------------------------------------------------

def encode_frame(payload: bytes) -> bytes:
    """
    Encode a byte payload as an FSK audio frame.

    Frame structure:
      [preamble: 800Hz 0.3s][start: 600Hz 0.1s]
      [length: 2 bytes big-endian][payload: N bytes][crc16: 2 bytes]

    Returns WAV file bytes.
    """
    # Build the frame data
    length = len(payload)
    frame_data = struct.pack('>H', length) + payload
    checksum = crc16(frame_data)
    frame_data += struct.pack('>H', checksum)

    # Convert to bit stream
    bits = []
    for byte in frame_data:
        for bit_pos in range(7, -1, -1):  # MSB first
            bits.append((byte >> bit_pos) & 1)

    # Generate audio
    pcm_samples = []

    # Preamble: sync tone (with fade-in)
    pcm_samples.extend(_tone_samples(FREQ_PREAMBLE, PREAMBLE_DURATION, fade_ms=5))

    # Start marker
    pcm_samples.extend(_tone_samples(FREQ_START, START_DURATION, fade_ms=2))

    # Data bits
    for bit in bits:
        freq = FREQ_SPACE if bit else FREQ_MARK
        pcm_samples.extend(_tone_samples(freq, BIT_DURATION))

    # Trailing silence (50ms)
    pcm_samples.extend([0] * int(SAMPLE_RATE * 0.05))

    pcm = struct.pack(f'<{len(pcm_samples)}h', *pcm_samples)
    return _pcm_to_wav(pcm)


def frame_duration(payload_size: int) -> float:
    """Calculate total audio duration for a payload of given size."""
    frame_bytes = 2 + payload_size + 2  # length + payload + crc
    data_bits = frame_bytes * 8
    return PREAMBLE_DURATION + START_DURATION + (data_bits * BIT_DURATION) + 0.05


# ---------------------------------------------------------------------------
# Decoder: FSK audio → bytes
# ---------------------------------------------------------------------------

def _goertzel_mag(samples: list[int], target_freq: float, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Goertzel algorithm — compute magnitude of a single DFT bin.
    Much more accurate than zero-crossing for short windows.
    """
    n = len(samples)
    if n == 0:
        return 0.0
    k = round(n * target_freq / sample_rate)
    w = 2.0 * math.pi * k / n
    coeff = 2.0 * math.cos(w)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for sample in samples:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return math.sqrt(abs(power)) / n


def _detect_fsk_bit(samples: list[int], sample_rate: int = SAMPLE_RATE) -> int:
    """
    Detect whether a sample window contains FREQ_MARK (bit 0) or
    FREQ_SPACE (bit 1) using Goertzel magnitude comparison.
    """
    mag_mark = _goertzel_mag(samples, FREQ_MARK, sample_rate)
    mag_space = _goertzel_mag(samples, FREQ_SPACE, sample_rate)
    return 0 if mag_mark > mag_space else 1


def _detect_frequency(samples: list[int], sample_rate: int = SAMPLE_RATE) -> float:
    """
    Detect dominant frequency among our known FSK frequencies.
    Used for preamble/start detection.
    """
    if len(samples) < 4:
        return 0.0
    candidates = [FREQ_PREAMBLE, FREQ_START, FREQ_MARK, FREQ_SPACE]
    best_freq = 0.0
    best_mag = 0.0
    for freq in candidates:
        mag = _goertzel_mag(samples, freq, sample_rate)
        if mag > best_mag:
            best_mag = mag
            best_freq = freq
    return best_freq


def _find_preamble(samples: list[int], sample_rate: int = SAMPLE_RATE) -> int:
    """
    Find data start by detecting preamble onset and computing offset.

    Strategy: find the first sample where 800Hz preamble begins, then
    add the known preamble + start durations to get the exact data start.
    This avoids boundary-detection issues at tone transitions.
    """
    window = int(sample_rate * 0.02)  # 20ms analysis window
    step = window // 4  # fine stepping for accuracy

    preamble_start = -1
    consecutive = 0

    for i in range(0, len(samples) - window, step):
        chunk = samples[i:i + window]
        freq = _detect_frequency(chunk, sample_rate)

        if freq == FREQ_PREAMBLE:
            consecutive += 1
            if consecutive >= 3 and preamble_start < 0:
                # Back-calculate the true start of preamble
                preamble_start = i - (consecutive - 1) * step
                if preamble_start < 0:
                    preamble_start = 0
                # Data starts at: preamble_start + preamble_duration + start_duration
                data_start = preamble_start + int((PREAMBLE_DURATION + START_DURATION) * sample_rate)
                return data_start
        else:
            consecutive = 0

    return -1


def decode_frame(wav_data: bytes) -> Optional[bytes]:
    """
    Decode an FSK audio frame back to bytes.

    Reads a WAV file, finds the preamble, demodulates FSK bits,
    verifies CRC, and returns the payload.

    Returns None if no valid frame found or CRC mismatch.
    """
    # Parse WAV header (skip 44 bytes)
    if len(wav_data) < 44:
        return None

    pcm = wav_data[44:]
    n_samples = len(pcm) // 2
    samples = list(struct.unpack(f'<{n_samples}h', pcm[:n_samples * 2]))

    # Find preamble
    data_start = _find_preamble(samples)
    if data_start < 0:
        return None

    # Demodulate bits
    bits = []
    pos = data_start
    while pos + SAMPLES_PER_BIT <= len(samples):
        chunk = samples[pos:pos + SAMPLES_PER_BIT]
        freq = _detect_frequency(chunk)

        # Check if there's meaningful signal energy
        max_amp = max(abs(s) for s in chunk) if chunk else 0
        if max_amp < 500:  # silence / end of data
            break
        bits.append(_detect_fsk_bit(chunk))

        pos += SAMPLES_PER_BIT

    # Convert bits to bytes
    if len(bits) < 32:  # Need at least 4 bytes (length + min CRC)
        return None

    frame_bytes = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        frame_bytes.append(byte)

    if len(frame_bytes) < 4:
        return None

    # Parse frame: [length:2][payload:N][crc:2]
    length = struct.unpack('>H', bytes(frame_bytes[:2]))[0]

    if length + 4 > len(frame_bytes):
        # Not enough data decoded
        return None

    payload = bytes(frame_bytes[2:2 + length])
    received_crc = struct.unpack('>H', bytes(frame_bytes[2 + length:4 + length]))[0]

    # Verify CRC
    expected_crc = crc16(bytes(frame_bytes[:2 + length]))
    if received_crc != expected_crc:
        return None

    return payload


# ---------------------------------------------------------------------------
# SCL delta encoding for acoustic transport
# ---------------------------------------------------------------------------

def encode_delta_acoustic(delta_dict: dict) -> bytes:
    """
    Encode an SCL delta as a compact binary payload for acoustic transmission.

    Format: JSON-encoded delta (simple, debuggable, not optimal).
    For production, this would use protobuf or CBOR.
    """
    import json
    return json.dumps(delta_dict, separators=(',', ':')).encode('utf-8')


def decode_delta_acoustic(payload: bytes) -> Optional[dict]:
    """Decode an acoustic payload back to a delta dict."""
    import json
    try:
        return json.loads(payload.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def encode_fingerprint_acoustic(fingerprint: str) -> bytes:
    """
    Encode a Braille fingerprint as a compact binary payload.
    Braille chars are U+2800–U+28FF, so each is 1 byte of offset.
    """
    return bytes(ord(c) - 0x2800 for c in fingerprint)


def decode_fingerprint_acoustic(payload: bytes) -> str:
    """Decode acoustic payload back to a Braille fingerprint string."""
    return ''.join(chr(0x2800 + b) for b in payload)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@dataclass
class AcousticFrame:
    """A decoded acoustic frame with metadata."""
    payload: bytes
    payload_type: str  # "fingerprint", "delta", "ping"
    duration_s: float
    bit_errors: int = 0


def transmit_fingerprint(fingerprint: str, play: bool = True) -> str:
    """
    Encode and optionally play a Braille fingerprint as FSK audio.

    Returns path to the WAV file.
    """
    # Frame type byte (0x01 = fingerprint) + payload
    payload = b'\x01' + encode_fingerprint_acoustic(fingerprint)
    wav = encode_frame(payload)

    path = tempfile.mktemp(suffix='.wav')
    with open(path, 'wb') as f:
        f.write(wav)

    if play:
        subprocess.Popen(
            ['afplay', path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).wait()

    return path


def transmit_delta(delta_dict: dict, play: bool = True) -> str:
    """
    Encode and optionally play an SCL delta as FSK audio.

    Returns path to the WAV file.
    """
    payload = b'\x02' + encode_delta_acoustic(delta_dict)
    wav = encode_frame(payload)

    dur = frame_duration(len(payload))
    print(f"  Acoustic TX: {len(payload)} bytes, {dur:.1f}s")

    path = tempfile.mktemp(suffix='.wav')
    with open(path, 'wb') as f:
        f.write(wav)

    if play:
        subprocess.Popen(
            ['afplay', path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).wait()

    return path


def receive_frame(wav_path: str) -> Optional[AcousticFrame]:
    """
    Decode an FSK audio frame from a WAV file.

    Returns AcousticFrame with parsed payload, or None if decoding fails.
    """
    with open(wav_path, 'rb') as f:
        wav_data = f.read()

    payload = decode_frame(wav_data)
    if payload is None or len(payload) < 1:
        return None

    frame_type = payload[0]
    data = payload[1:]

    if frame_type == 0x01:
        return AcousticFrame(
            payload=data,
            payload_type="fingerprint",
            duration_s=frame_duration(len(payload)),
        )
    elif frame_type == 0x02:
        return AcousticFrame(
            payload=data,
            payload_type="delta",
            duration_s=frame_duration(len(payload)),
        )
    else:
        return AcousticFrame(
            payload=data,
            payload_type="unknown",
            duration_s=frame_duration(len(payload)),
        )


def record_and_decode(duration_s: float = 5.0) -> Optional[AcousticFrame]:
    """
    Record audio from the microphone and attempt to decode an FSK frame.

    Uses macOS `rec` (from sox) or falls back to an AppleScript-based
    QuickTime recording.
    """
    wav_path = tempfile.mktemp(suffix='.wav')

    # Try sox's rec command first
    try:
        subprocess.run(
            ['rec', '-q', '-r', str(SAMPLE_RATE), '-c', '1', '-b', '16',
             wav_path, 'trim', '0', str(duration_s)],
            timeout=duration_s + 2,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return receive_frame(wav_path)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Fallback: use ffmpeg if available
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'avfoundation', '-i', ':0',
             '-t', str(duration_s), '-ar', str(SAMPLE_RATE),
             '-ac', '1', '-sample_fmt', 's16', wav_path],
            timeout=duration_s + 5,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return receive_frame(wav_path)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    print("  No recording tool available. Install sox (`brew install sox`)")
    print("  or ffmpeg (`brew install ffmpeg`) for microphone input.")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    def usage():
        print("Cortex Acoustic Gossip Transport")
        print()
        print("Usage:")
        print("  python -m src.acoustic_transport send-fingerprint ⣦⠷⡓⠠")
        print("  python -m src.acoustic_transport send-delta '{\"set\":{\"tier\":\"L4\"}}'")
        print("  python -m src.acoustic_transport decode file.wav")
        print("  python -m src.acoustic_transport listen [duration_s]")
        print("  python -m src.acoustic_transport roundtrip           # encode→decode self-test")
        print()
        print("Protocol: Binary FSK at 300 baud")
        print(f"  Bit 0 (mark):  {FREQ_MARK} Hz")
        print(f"  Bit 1 (space): {FREQ_SPACE} Hz")
        print(f"  Preamble:      {FREQ_PREAMBLE} Hz ({PREAMBLE_DURATION}s)")
        print(f"  Start marker:  {FREQ_START} Hz ({START_DURATION}s)")
        print(f"  Data rate:     {BAUD_RATE} baud (~{BAUD_RATE // 8} bytes/sec)")

    if len(sys.argv) < 2:
        usage()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "send-fingerprint":
        fp = sys.argv[2] if len(sys.argv) > 2 else "⣦⠷⡓⠠"
        print(f"TX fingerprint: {fp}")
        dur = frame_duration(5)  # 1 type byte + 4 fingerprint bytes
        print(f"  Duration: {dur:.1f}s")
        path = transmit_fingerprint(fp, play=True)
        print(f"  Saved: {path}")

    elif cmd == "send-delta":
        delta_json = sys.argv[2] if len(sys.argv) > 2 else '{"set":{"max_tier":"L4","accuracy":"0.91"}}'
        delta = json.loads(delta_json)
        print(f"TX delta: {delta}")
        path = transmit_delta(delta, play=True)
        print(f"  Saved: {path}")

    elif cmd == "decode":
        wav_path = sys.argv[2]
        print(f"Decoding: {wav_path}")
        frame = receive_frame(wav_path)
        if frame:
            print(f"  Type: {frame.payload_type}")
            print(f"  Duration: {frame.duration_s:.1f}s")
            if frame.payload_type == "fingerprint":
                fp = decode_fingerprint_acoustic(frame.payload)
                print(f"  Fingerprint: {fp}")
            elif frame.payload_type == "delta":
                delta = decode_delta_acoustic(frame.payload)
                print(f"  Delta: {json.dumps(delta, indent=2)}")
            else:
                print(f"  Raw: {frame.payload.hex()}")
        else:
            print("  Failed to decode frame")

    elif cmd == "listen":
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        print(f"Listening for {duration}s...")
        frame = record_and_decode(duration)
        if frame:
            print(f"  Received: {frame.payload_type}")
            if frame.payload_type == "fingerprint":
                print(f"  Fingerprint: {decode_fingerprint_acoustic(frame.payload)}")
            elif frame.payload_type == "delta":
                print(f"  Delta: {decode_delta_acoustic(frame.payload)}")
        else:
            print("  No frame detected")

    elif cmd == "roundtrip":
        print("=== Acoustic Roundtrip Self-Test ===\n")

        # Test 1: Fingerprint
        fp_in = "⣦⠷⡓⠠"
        print(f"1. Fingerprint: {fp_in}")
        payload = b'\x01' + encode_fingerprint_acoustic(fp_in)
        wav = encode_frame(payload)
        decoded = decode_frame(wav)
        if decoded and decoded[0] == 0x01:
            fp_out = decode_fingerprint_acoustic(decoded[1:])
            match = "✓" if fp_out == fp_in else "✗"
            print(f"   Decoded: {fp_out} {match}")
        else:
            print("   ✗ Decode failed")

        # Test 2: Delta
        delta_in = {"set": {"max_tier": "L4", "accuracy": "0.91"}, "agent": "cortex-11411"}
        print(f"\n2. Delta: {delta_in}")
        payload = b'\x02' + encode_delta_acoustic(delta_in)
        dur = frame_duration(len(payload))
        print(f"   Frame: {len(payload)} bytes, {dur:.1f}s audio")
        wav = encode_frame(payload)
        decoded = decode_frame(wav)
        if decoded and decoded[0] == 0x02:
            delta_out = decode_delta_acoustic(decoded[1:])
            match = "✓" if delta_out == delta_in else "✗"
            print(f"   Decoded: {delta_out} {match}")
        else:
            print("   ✗ Decode failed")

        # Test 3: Larger delta
        big_delta = {
            "agent": "cortex-11411",
            "set": {
                "max_tier": "L5",
                "accuracy_L2": "0.94",
                "accuracy_L3": "0.87",
                "policy_version": "7",
                "boot_count": "42",
                "thread_count": "8",
                "gpu_layers": "24",
            },
            "delete": ["old_config"]
        }
        payload = b'\x02' + encode_delta_acoustic(big_delta)
        dur = frame_duration(len(payload))
        print(f"\n3. Large delta: {len(payload)} bytes, {dur:.1f}s audio")
        wav = encode_frame(payload)
        print(f"   WAV size: {len(wav)} bytes")
        decoded = decode_frame(wav)
        if decoded and decoded[0] == 0x02:
            delta_out = decode_delta_acoustic(decoded[1:])
            match = "✓" if delta_out == big_delta else "✗"
            print(f"   Round-trip: {match}")
            if delta_out != big_delta:
                print(f"   Expected: {big_delta}")
                print(f"   Got:      {delta_out}")
        else:
            print("   ✗ Decode failed")

        print("\n=== Done ===")

    else:
        usage()
