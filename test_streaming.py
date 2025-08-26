#!/usr/bin/env python3
"""
Streaming SEANet Test Script
Tests streaming audio encoding/decoding using EnCodec with HuggingFace weights
"""

import os
import wave
import math
import numpy as np
import torch
import torch.nn.functional as F
from model.seanet import StreamingEncodecEncoder


def read_wav(path: str):
    """Read WAV file and return mono audio and sample rate."""
    with wave.open(path, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
    
    # Convert to float32
    if sample_width == 2:  # 16-bit PCM
        audio = np.frombuffer(raw_data, dtype='<i2').astype(np.float32) / 32768.0
    elif sample_width == 4:  # 32-bit float
        audio = np.frombuffer(raw_data, dtype='<f4').astype(np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    # Convert to mono if needed
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    
    return audio, sample_rate


def write_wav(path: str, audio: np.ndarray, sample_rate: int):
    """Write audio to WAV file."""
    # Clip and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)
    
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def calculate_snr(original: np.ndarray, reconstructed: np.ndarray):
    """Calculate Signal-to-Noise Ratio in dB."""
    # Align lengths
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Calculate SNR
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    
    if noise_power < 1e-10:
        return float('inf')
    
    snr = 10 * math.log10(signal_power / noise_power)
    return snr


def main():
    """Main test function."""
    print("="*60)
    print("Streaming Audio Encoding/Decoding Test")
    print("Using EnCodec with HuggingFace Weights")
    print("="*60)
    
    # Load input audio
    input_file = "original_audio.wav"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    audio, sample_rate = read_wav(input_file)
    print(f"\nInput: {input_file}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio)/sample_rate:.2f} seconds")
    print(f"Samples: {len(audio)}")
    
    # Resample to 24kHz if needed (EnCodec requirement)
    if sample_rate != 24000:
        print(f"Resampling from {sample_rate} Hz to 24000 Hz...")
        ratio = 24000 / sample_rate
        new_length = int(len(audio) * ratio)
        x = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=new_length, mode='linear', align_corners=False)
        audio = x.squeeze().numpy()
        sample_rate = 24000
    
    # Convert to tensor
    x = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    
    # Initialize streaming encoder
    print("\nInitializing StreamingEncodecEncoder...")
    encoder = StreamingEncodecEncoder(model_name="facebook/encodec_24khz")
    
    # Test 1: Offline encoding (baseline)
    print("\n" + "-"*60)
    print("Test 1: Offline Encoding (Baseline)")
    print("-"*60)
    
    z_offline = encoder.encode_offline(x)
    y_offline = encoder.decode(z_offline)
    y_offline_np = y_offline.squeeze().numpy()
    
    snr_offline = calculate_snr(audio, y_offline_np)
    print(f"Latent shape: {z_offline.shape}")
    print(f"SNR: {snr_offline:.2f} dB")
    
    write_wav("output_offline.wav", y_offline_np, sample_rate)
    print(f"Saved: output_offline.wav")
    
    # Test 2: Streaming encoding
    print("\n" + "-"*60)
    print("Test 2: Streaming Encoding")
    print("-"*60)
    
    encoder.reset_state()
    
    # Process in variable-sized chunks
    chunk_sizes = [1024,512,1024,1024,1024,1024,1024,1024,1024,1024]
    position = 0
    encoded_chunks = []
    
    print("Processing audio in chunks...")
    while position < x.size(-1):
        chunk_size = chunk_sizes[len(encoded_chunks) % len(chunk_sizes)]
        chunk = x[..., position:min(position + chunk_size, x.size(-1))]
        
        if chunk.size(-1) > 0:
            z_chunk = encoder.encode_streaming(chunk)
            if z_chunk.size(-1) > 0:
                encoded_chunks.append(z_chunk)
        
        position += chunk_size
    
    # Process remaining buffer
    z_final = encoder.flush_buffer()
    if z_final is not None and z_final.size(-1) > 0:
        encoded_chunks.append(z_final)
    
    # Concatenate all chunks
    z_streaming = torch.cat(encoded_chunks, dim=-1)
    print(f"Processed {len(encoded_chunks)} chunks")
    print(f"Latent shape: {z_streaming.shape}")
    
    # Decode
    y_streaming = encoder.decode(z_streaming)
    y_streaming_np = y_streaming.squeeze().numpy()
    
    snr_streaming = calculate_snr(audio, y_streaming_np)
    print(f"SNR: {snr_streaming:.2f} dB")
    
    write_wav("output_streaming.wav", y_streaming_np, sample_rate)
    print(f"Saved: output_streaming.wav")
    
    # Summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Offline SNR:   {snr_offline:.2f} dB")
    print(f"Streaming SNR: {snr_streaming:.2f} dB")
    print(f"Difference:    {abs(snr_offline - snr_streaming):.2f} dB")
    print("\nOutput files:")
    print("  - output_offline.wav   (offline encoding)")
    print("  - output_streaming.wav (streaming encoding)")
    print("="*60)


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # Disable gradients for inference
    main()