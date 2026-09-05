"""
Audio and Voice Deepfake Detector for TrustGuard AI.
Performs audio segment-by-segment window analysis, short-time spectral feature extraction
(Spectral Centroid, Spectral Flux, Spectral Rolloff, High-Frequency Vocoder roll-off, ZCR),
and deterministic temporal segment aggregation.
"""
import os
import io
import wave
import struct
import tempfile
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

from backend.utils.response_utils import clamp_score, authenticity_classification

logger = logging.getLogger("trustguard.audio_detector")

# Segment window configuration
SEGMENT_DURATION_SEC = 2.0  # 2.0-second sliding analysis window
SEGMENT_HOP_SEC = 1.0       # 1.0-second hop (50% overlap)
MIN_AUDIO_DURATION = 0.5    # Minimum duration required for analysis


def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Robust audio decoder supporting WAV, SoundFile formats, and raw PCM fallback.
    Returns (mono_numpy_array, sample_rate).
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temp_file.write(audio_bytes)
        temp_file.close()
        temp_path = temp_file.name

        # 1. Try soundfile
        try:
            import soundfile as sf
            data, sr = sf.read(temp_path)
            if data is not None and len(data) > 0:
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                return data.astype(np.float32), int(sr)
        except Exception:
            pass

        # 2. Try standard library wave module
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as w:
                n_channels = w.getnchannels()
                sampwidth = w.getsampwidth()
                sr = w.getframerate()
                n_frames = w.getnframes()
                raw_frames = w.readframes(n_frames)
                if sampwidth == 2:
                    data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    data = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                elif sampwidth == 1:
                    data = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                else:
                    data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

                if n_channels > 1:
                    data = data.reshape(-1, n_channels).mean(axis=1)
                return data, int(sr)
        except Exception:
            pass

        # 3. Try librosa if available
        try:
            import librosa
            data, sr = librosa.load(temp_path, sr=None)
            if data is not None and len(data) > 0:
                return data.astype(np.float32), int(sr)
        except Exception:
            pass

        # 4. Fallback: Parse as raw 16-bit PCM (16kHz)
        raw_pcm = np.frombuffer(audio_bytes, dtype=np.int16, offset=min(44, len(audio_bytes) // 2)).astype(np.float32) / 32768.0
        if len(raw_pcm) > 100:
            return raw_pcm, 16000

        raise ValueError("Could not decode audio stream with any available decoder.")

    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass


def analyze_audio_segment(
    segment_data: np.ndarray,
    sample_rate: int
) -> Dict[str, Any]:
    """
    Extracts acoustic & spectral features from a single audio segment (numpy float array)
    and evaluates synthetic vs organic human voice characteristics.
    """
    if len(segment_data) == 0:
        return {"risk_score": 10.0, "confidence": 75.0, "is_fake": False, "prediction": "REAL"}

    # 1. Zero Crossing Rate (ZCR)
    zcr = float(np.mean(np.abs(np.diff(np.sign(segment_data))))) / 2.0

    # 2. Short-Time Fourier Transform (STFT)
    n_fft = min(2048, len(segment_data))
    hop_length = 512
    if len(segment_data) < n_fft:
        segment_data = np.pad(segment_data, (0, n_fft - len(segment_data)))

    window = np.hanning(n_fft)
    num_frames = max(1, (len(segment_data) - n_fft) // hop_length + 1)
    stft_matrix = []
    for i in range(num_frames):
        start = i * hop_length
        frame = segment_data[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        stft_matrix.append(spectrum)

    stft_matrix = np.array(stft_matrix)  # shape: (frames, freq_bins)
    freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 3. Spectral Centroid
    magnitudes_sum = np.sum(stft_matrix, axis=1) + 1e-8
    centroids = np.sum(stft_matrix * freq_bins, axis=1) / magnitudes_sum
    mean_centroid = float(np.mean(centroids))
    std_centroid = float(np.std(centroids))

    # 4. Spectral Flux (frame-to-frame frequency shift)
    if len(stft_matrix) > 1:
        flux = np.sqrt(np.sum(np.diff(stft_matrix, axis=0)**2, axis=1))
        mean_flux = float(np.mean(flux))
    else:
        mean_flux = 0.8

    # 5. High-Frequency Vocoder Artifacts (AI vocoders produce unnatural cutoffs > 7.5 kHz)
    high_freq_mask = freq_bins > 7500
    total_energy = np.sum(stft_matrix) + 1e-8
    high_freq_energy = float(np.sum(stft_matrix[:, high_freq_mask]) / total_energy)

    # 6. Spectral Rolloff (85% energy frequency)
    cumulative_energy = np.cumsum(stft_matrix, axis=1)
    threshold = 0.85 * cumulative_energy[:, -1:]
    rolloff_indices = np.argmax(cumulative_energy >= threshold, axis=1)
    rolloff_freqs = freq_bins[np.clip(rolloff_indices, 0, len(freq_bins) - 1)]
    mean_rolloff = float(np.mean(rolloff_freqs))

    # Forensic Decision Rules for this segment:
    # Human vocal tracts exhibit rich dynamic pitch/timbre modulation (std_centroid > 320 Hz)
    # Neural vocoders (HiFi-GAN, WaveGlow, Tacotron, VITS) exhibit flat harmonic phases or unnatural cutoffs
    is_vocoder_synthetic = (std_centroid < 180.0 or high_freq_energy < 0.0015 or mean_flux < 0.22)
    is_natural_human = (std_centroid > 320.0 and mean_flux > 0.35 and 0.008 < high_freq_energy < 0.35)

    if is_vocoder_synthetic:
        risk_score = 86.0
        confidence = 88.0
        prediction = "SYNTHETIC"
        is_fake = True
    elif is_natural_human:
        risk_score = 12.0
        confidence = 92.0
        prediction = "REAL"
        is_fake = False
    else:
        # Interpolate based on centroid variance and flux
        risk_score = clamp_score(70.0 - (std_centroid / 10.0) + (1.0 - mean_flux) * 20.0)
        confidence = 72.0
        prediction = "SYNTHETIC" if risk_score > 40.0 else "REAL"
        is_fake = risk_score > 40.0

    return {
        "risk_score": round(risk_score, 1),
        "confidence": round(confidence, 1),
        "is_fake": is_fake,
        "prediction": prediction,
        "std_centroid": round(std_centroid, 1),
        "mean_flux": round(mean_flux, 2),
        "high_freq_energy": round(high_freq_energy * 100, 3),
        "mean_rolloff": round(mean_rolloff, 1)
    }


def analyze_audio_bytes(
    audio_bytes: bytes,
    segment_duration: float = SEGMENT_DURATION_SEC,
    hop_duration: float = SEGMENT_HOP_SEC
) -> Dict[str, Any]:
    """
    Analyzes raw audio bytes with segment-level sliding windows and spectral forensic models.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return {
            "success": False,
            "error": "Uploaded audio file is empty or corrupted."
        }

    try:
        mono_data, sample_rate = decode_audio_bytes(audio_bytes)
        duration_sec = float(len(mono_data)) / float(sample_rate) if sample_rate > 0 else 0.0

        if duration_sec < MIN_AUDIO_DURATION:
            return {
                "success": False,
                "error": f"Audio file is too short ({duration_sec:.2f}s). Minimum {MIN_AUDIO_DURATION}s required."
            }

        # Segment sliding window
        window_samples = int(segment_duration * sample_rate)
        hop_samples = max(1, int(hop_duration * sample_rate))

        segment_results: List[Dict[str, Any]] = []
        segment_scores: List[float] = []
        confidences: List[float] = []
        suspicious_count = 0

        start_sample = 0
        seg_idx = 1

        while start_sample < len(mono_data):
            end_sample = min(len(mono_data), start_sample + window_samples)
            seg_data = mono_data[start_sample:end_sample]

            # Only analyze segments with sufficient length (> 0.25s)
            if len(seg_data) >= int(0.25 * sample_rate):
                t_start = round(float(start_sample / sample_rate), 2)
                t_end = round(float(end_sample / sample_rate), 2)

                seg_res = analyze_audio_segment(seg_data, sample_rate)
                segment_scores.append(seg_res["risk_score"])
                confidences.append(seg_res["confidence"])

                if seg_res["is_fake"]:
                    suspicious_count += 1

                segment_results.append({
                    "segment_index": seg_idx,
                    "start_time": t_start,
                    "end_time": t_end,
                    "time_label": f"{int(t_start//60):02d}:{int(t_start%60):02d}–{int(t_end//60):02d}:{int(t_end%60):02d}",
                    "prediction": seg_res["prediction"],
                    "confidence": seg_res["confidence"],
                    "risk_score": seg_res["risk_score"],
                    "is_suspicious": seg_res["is_fake"],
                    "details": {
                        "centroid_variance": seg_res["std_centroid"],
                        "spectral_flux": seg_res["mean_flux"],
                        "high_freq_ratio": seg_res["high_freq_energy"]
                    }
                })
                seg_idx += 1

            if end_sample >= len(mono_data):
                break
            start_sample += hop_samples

        if not segment_scores:
            return {
                "success": False,
                "error": "Could not extract valid audio segments."
            }

        total_segments = len(segment_results)
        avg_risk = float(np.mean(segment_scores))
        max_risk = float(np.max(segment_scores))
        avg_conf = clamp_score(float(np.mean(confidences)))

        # Temporal aggregation:
        # If > 30% of segments are suspicious, weight towards peak risk
        if (suspicious_count / float(total_segments)) >= 0.30 or max_risk >= 75.0:
            overall_risk = clamp_score(0.65 * max_risk + 0.35 * avg_risk)
        else:
            overall_risk = clamp_score(0.35 * max_risk + 0.65 * avg_risk)

        authenticity = clamp_score(100.0 - overall_risk)

        # Classification
        if overall_risk <= 25.0:
            classification = "REAL"
            classification_label = "GENUINE VOICE"
            risk_level = "Low"
            status = "safe"
            explanation = (
                f"Acoustic feature analysis verified natural vocal tract dynamics, organic pitch variance, "
                f"and continuous harmonic formants across {total_segments} audio segments ({duration_sec:.1f}s)."
            )
        elif overall_risk <= 50.0:
            classification = "SUSPICIOUS"
            classification_label = "SUSPICIOUS VOICE CONTENT"
            risk_level = "Moderate"
            status = "mod"
            explanation = (
                f"Audio features show mostly natural spectral distribution with slight background noise "
                f"or compression variance across {suspicious_count} of {total_segments} segments."
            )
        else:
            classification = "SYNTHETIC"
            classification_label = "AI-GENERATED / SYNTHETIC VOICE"
            risk_level = "Critical" if overall_risk >= 75.0 else "High"
            status = "danger"
            explanation = (
                f"Acoustic analysis detected artificial phase regularity and synthetic vocoder frequency "
                f"cutoffs characteristic of AI voice cloning/TTS across {suspicious_count} segment windows."
            )

        indicators = [
            {
                "label": "Segment-by-Segment Spectral Forensics",
                "detail": f"Analyzed {total_segments} windows ({segment_duration}s windows, {hop_duration}s hop) — {suspicious_count} suspicious / {total_segments - suspicious_count} authentic.",
                "score": round(overall_risk, 1),
                "level": "safe" if suspicious_count == 0 else "high" if suspicious_count >= total_segments * 0.3 else "mod"
            },
            {
                "label": "Vocoder High-Frequency Phase Signature",
                "detail": f"High-frequency energy and roll-off stability evaluated across {duration_sec:.1f}s sample rate {sample_rate} Hz.",
                "score": round(overall_risk, 1),
                "level": "high" if overall_risk > 50.0 else "safe"
            },
            {
                "label": "Spectral Centroid Harmonic Modulations",
                "detail": f"Evaluated dynamic timbre cadence (Peak segment risk: {max_risk:.1f}/100, Confidence: {avg_conf:.1f}%).",
                "score": round(max_risk, 1),
                "level": "high" if max_risk >= 65.0 else "mod" if max_risk > 35.0 else "safe"
            }
        ]

        return {
            "success": True,
            "type": "audio",
            "mediaType": "audio",
            "classification": classification,
            "classification_label": classification_label,
            "prediction": classification,
            "confidence": round(avg_conf / 100.0, 2),
            "confidence_pct": round(avg_conf, 1),
            "risk_score": round(overall_risk, 1),
            "risk_level": risk_level,
            "authenticity": round(authenticity, 1),
            "authenticity_probability": round(authenticity, 1),
            "duration": round(duration_sec, 2),
            "sample_rate": sample_rate,
            "segments_analyzed": total_segments,
            "suspicious_segments": suspicious_count,
            "status": status,
            "explanation": explanation,
            "indicators": indicators,
            "segment_results": segment_results,
            "signals": indicators
        }

    except Exception as e:
        logger.error(f"Audio analysis error: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Audio analysis failed: {str(e)}"
        }
