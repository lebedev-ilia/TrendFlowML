"""
Renderer для asr_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np
from pathlib import Path
import html

logger = logging.getLogger(__name__)

def safe_log_warning(logger_instance, message, *args, **kwargs):
    """Safely log a warning message, catching I/O errors from closed handlers."""
    try:
        # Try to log directly - catch all exceptions to prevent crashes
        logger_instance.warning(message, *args, **kwargs)
    except Exception:
        # Catch ALL exceptions silently - handlers may be closed, streams may be closed,
        # or logging infrastructure may be in an invalid state during shutdown
        pass

from ...core.renderer import load_npz, extract_meta

def render_asr_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для asr_extractor (privacy-safe, без raw текста)."""
    render = {
        "component": "asr_extractor",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract scalar features
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    
    features = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            features[name] = feature_values[i]
    
    # Helper function to safely convert to int (handles NaN)
    def safe_int(value, default=0):
        if value is None:
            return default
        try:
            val = float(value)
            if np.isnan(val):
                return default
            return int(val)
        except (ValueError, TypeError):
            return default
    
    # Helper function to safely convert to float (handles NaN)
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            val = float(value)
            if np.isnan(val):
                return default
            return val
        except (ValueError, TypeError):
            return default

    # Timing / processing info (best-effort)
    stage_timings = meta.get("stage_timings_ms") if isinstance(meta, dict) else None
    if not isinstance(stage_timings, dict):
        stage_timings = {}
    extractor_wall_ms = stage_timings.get("extractor_wall_ms")
    extractor_reported_ms = stage_timings.get("extractor_reported_ms")
    render["summary"]["processing_time_ms"] = safe_float(extractor_wall_ms, None) if extractor_wall_ms is not None else None
    render["summary"]["processing_time_reported_ms"] = safe_float(extractor_reported_ms, None) if extractor_reported_ms is not None else None
    # Resource metrics (if present)
    res_metrics = meta.get("resource_metrics") if isinstance(meta, dict) else None
    if isinstance(res_metrics, dict):
        render["summary"]["cpu_rss_peak_mb"] = safe_float(res_metrics.get("cpu_rss_peak_mb"), None)
        render["summary"]["gpu_vram_peak_mb"] = safe_float(res_metrics.get("gpu_vram_peak_mb"), None)
    
    # Summary (privacy-safe: только статистики, без raw текста)
    render["summary"].update({
        "segments_count": safe_int(features.get("segments_count", 0)),
        "token_total": safe_int(features.get("token_total", 0)),
        "token_density_per_sec": safe_float(features.get("token_density_per_sec", 0.0)),
        "speech_rate_wpm": safe_float(features.get("speech_rate_wpm", 0.0)),
        "segments_with_speech": safe_int(features.get("segments_with_speech", 0)),
        "avg_segment_duration_sec": safe_float(features.get("avg_segment_duration_sec", 0.0)),
        "token_variance": safe_float(features.get("token_variance", 0.0)),
    })
    
    # Language distribution
    lang_dist = npz_data.get("lang_distribution")
    if lang_dist is not None:
        if isinstance(lang_dist, np.ndarray) and lang_dist.dtype == object:
            lang_dist = lang_dist.item() if lang_dist.size == 1 else {}
        if isinstance(lang_dist, dict):
            render["summary"]["lang_distribution"] = {str(k): safe_int(v, 0) for k, v in lang_dist.items()}
    
    # Timeline (без raw текста, только временные метки и token counts)
    segment_start_sec = npz_data.get("segment_start_sec")
    segment_end_sec = npz_data.get("segment_end_sec")
    segment_center_sec = npz_data.get("segment_center_sec")
    token_counts = npz_data.get("token_counts")
    lang_id_by_segment = npz_data.get("lang_id_by_segment")
    
    if segment_center_sec is not None:
        if isinstance(segment_center_sec, np.ndarray):
            segment_center_sec = segment_center_sec.tolist()
        if isinstance(segment_start_sec, np.ndarray):
            segment_start_sec = segment_start_sec.tolist()
        if isinstance(segment_end_sec, np.ndarray):
            segment_end_sec = segment_end_sec.tolist()
        if isinstance(token_counts, np.ndarray):
            token_counts = token_counts.tolist()
        if isinstance(lang_id_by_segment, np.ndarray):
            lang_id_by_segment = lang_id_by_segment.tolist()
        
        timeline = []
        for i, center_sec in enumerate(segment_center_sec):
            entry = {
                "center_sec": float(center_sec),
                "segment_index": i,
            }
            if segment_start_sec and i < len(segment_start_sec):
                entry["start_sec"] = float(segment_start_sec[i])
            if segment_end_sec and i < len(segment_end_sec):
                entry["end_sec"] = float(segment_end_sec[i])
            if token_counts and i < len(token_counts):
                entry["token_count"] = safe_int(token_counts[i], 0)
            if lang_id_by_segment and i < len(lang_id_by_segment):
                entry["lang_id"] = safe_int(lang_id_by_segment[i], -1)
            timeline.append(entry)
        render["timeline"] = timeline
        
        # Distribution of token counts
        if token_counts and len(token_counts) > 0:
            # Filter out NaN values for statistics
            valid_counts = [c for c in token_counts if not (isinstance(c, float) and np.isnan(c))]
            if valid_counts:
                render["distributions"]["token_counts"] = {
                    "min": safe_int(np.min(valid_counts), 0),
                    "max": safe_int(np.max(valid_counts), 0),
                    "mean": safe_float(np.mean(valid_counts), 0.0),
                    "std": safe_float(np.std(valid_counts), 0.0),
                    "median": safe_float(np.median(valid_counts), 0.0),
                }
    
    return render


def render_asr_extractor_html(npz_path: str, output_path: str, decode_tokenizer: bool = True) -> str:
    """
    Генерировать HTML render для asr_extractor (debug mode, может включать raw текст).
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
        decode_tokenizer: Если True, декодировать token IDs в текст (только для локального дебага)
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_asr_extractor(npz_data, meta)
    
    # Decode token IDs if requested (debug mode only)
    decoded_texts: list[str] = []
    decoded_full_text: str = ""
    if decode_tokenizer:
        try:
            from dp_models import get_global_model_manager  # type: ignore
            mm = get_global_model_manager()
            tok_spec = mm.get_spec(model_name="shared_tokenizer_v1")
            _d, _p, _rt, _eng, _wd, tok_artifacts = mm.resolve(tok_spec)
            tokenizer_path = list(tok_artifacts.values())[0] if tok_artifacts else None
            
            if tokenizer_path:
                # transformers expects a directory (or HF repo id), not a direct tokenizer.json file
                if os.path.isfile(tokenizer_path):
                    tokenizer_path = os.path.dirname(tokenizer_path)
                # Load tokenizer and decode
                try:
                    from transformers import AutoTokenizer  # type: ignore
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                    
                    token_ids_by_segment = npz_data.get("token_ids_by_segment")
                    if token_ids_by_segment is not None:
                        # load_npz() converts object arrays to list via .tolist(), so handle both ndarray and list
                        segments_tokens = None
                        if isinstance(token_ids_by_segment, np.ndarray) and token_ids_by_segment.dtype == object:
                            segments_tokens = list(token_ids_by_segment)
                        elif isinstance(token_ids_by_segment, list):
                            segments_tokens = token_ids_by_segment

                        if segments_tokens is not None:
                            full_parts: list[str] = []
                            for tok_arr in segments_tokens:
                                # tok_arr can be: np.ndarray, list[int], tuple[int], or nested
                                if isinstance(tok_arr, np.ndarray):
                                    tok_list = tok_arr.reshape(-1).tolist()
                                elif isinstance(tok_arr, (list, tuple)):
                                    tok_list = [int(t) for t in tok_arr]
                                else:
                                    tok_list = []
                                try:
                                    text = tokenizer.decode(tok_list, skip_special_tokens=True)
                                    decoded_texts.append(text)
                                    full_parts.append(text)
                                except Exception:
                                    decoded_texts.append("[decode_error]")
                                    full_parts.append("")
                            decoded_full_text = "\n".join([t for t in full_parts if t])
                        else:
                            decoded_texts = ["[no_token_ids]"]
                except Exception as e:
                    logger.warning(f"Failed to decode tokens: {e}")
                    decoded_texts = [f"[decode_failed: {e}]"]
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for decoding: {e}")
            decoded_texts = []

    # Fallback: if shared_tokenizer_v1 cannot represent Cyrillic (replacement chars),
    # decode text directly via Whisper on the original audio segments (debug-only).
    try:
        has_repl = ("\ufffd" in (decoded_full_text or "")) or any("\ufffd" in (t or "") for t in decoded_texts)
        if decode_tokenizer and has_repl:
            safe_log_warning(logger, "ASR HTML render: shared_tokenizer decode produced replacement chars; falling back to Whisper decode for debug text.")
            # Locate run root and frames_dir from manifest.json
            npz_p = Path(npz_path)
            run_root = npz_p.parent.parent  # .../<run_id>/
            manifest_path = run_root / "manifest.json"
            frames_dir = None
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    man = json.load(f) or {}
                frames_dir = ((man.get("run") or {}).get("frames_dir"))
            if frames_dir:
                audio_dir = Path(frames_dir).resolve().parent / "audio"
                audio_path = audio_dir / "audio.wav"
                segments_path = audio_dir / "segments.json"
                if audio_path.exists() and segments_path.exists():
                    with open(segments_path, "r", encoding="utf-8") as f:
                        seg_payload = json.load(f) or {}
                    asr_family = ((seg_payload.get("families") or {}).get("asr") or {})
                    segs = asr_family.get("segments") or []
                    # Load Whisper model via ModelManager
                    from dp_models import get_global_model_manager  # type: ignore
                    mm = get_global_model_manager()
                    model = mm.get(model_name="whisper_small_inprocess").handle
                    import torch
                    import whisper  # type: ignore
                    from src.core.audio_utils import AudioUtils  # type: ignore
                    au = AudioUtils(device="cpu", sample_rate=16000)
                    decoded_texts = []
                    parts: list[str] = []
                    # Keep count aligned with rendered timeline (if any)
                    timeline = render.get("timeline") or []
                    n_take = len(timeline) if isinstance(timeline, list) and timeline else len(segs)
                    segs = list(segs)[: int(n_take)]
                    for seg in segs:
                        ss = int(seg.get("start_sample", 0))
                        es = int(seg.get("end_sample", 0))
                        if es <= ss:
                            decoded_texts.append("")
                            continue
                        wav_t, sr = au.load_audio_segment(str(audio_path), start_sample=ss, end_sample=es, target_sr=16000)
                        wav = wav_t[0].detach().cpu()
                        mel = whisper.audio.log_mel_spectrogram(wav, n_mels=80)
                        # Pad/trim to Whisper expected frames
                        N_FRAMES = int(getattr(whisper.audio, "N_FRAMES", 3000))
                        if mel.shape[1] < N_FRAMES:
                            mel = torch.cat([mel, torch.zeros(80, N_FRAMES - mel.shape[1])], dim=1)
                        elif mel.shape[1] > N_FRAMES:
                            mel = mel[:, :N_FRAMES]
                        mel = mel.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                            opt = whisper.DecodingOptions(language=None, task="transcribe", fp16=torch.cuda.is_available(), without_timestamps=True)
                            res = model.decode(mel, opt)
                            res0 = res[0] if isinstance(res, list) and res else res
                            txt = str(getattr(res0, "text", "") or "")
                        decoded_texts.append(txt)
                        parts.append(txt)
                    decoded_full_text = "\n".join([t for t in parts if t])
    except Exception as e:
        logger.warning(f"ASR HTML render: whisper fallback decode failed: {e}")
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASR Extractor Debug View</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .stat-label {{ font-size: 0.9em; color: #666; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .timeline {{ margin: 20px 0; }}
        .segment {{ background: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 3px solid #4CAF50; border-radius: 3px; }}
        .segment-header {{ font-weight: bold; color: #333; }}
        .segment-text {{ margin-top: 5px; color: #555; }}
        .warning {{ background: #fff3cd; border-left-color: #ffc107; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ASR Extractor Debug View</h1>
        <p><strong>Status:</strong> {meta.get('status', 'unknown')}</p>
        <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
        <p><strong>Processing time (wall):</strong> {render.get('summary', {}).get('processing_time_ms', 'n/a')} ms</p>
        <p><strong>Processing time (reported):</strong> {render.get('summary', {}).get('processing_time_reported_ms', 'n/a')} ms</p>
        
        <h2>Summary Statistics</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Segments Count</div>
                <div class="stat-value">{render['summary'].get('segments_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Token Total</div>
                <div class="stat-value">{render['summary'].get('token_total', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Token Density (tokens/sec)</div>
                <div class="stat-value">{render['summary'].get('token_density_per_sec', 0.0):.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Speech Rate (WPM)</div>
                <div class="stat-value">{render['summary'].get('speech_rate_wpm', 0.0):.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Segments with Speech</div>
                <div class="stat-value">{render['summary'].get('segments_with_speech', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Segment Duration (sec)</div>
                <div class="stat-value">{render['summary'].get('avg_segment_duration_sec', 0.0):.2f}</div>
            </div>
        </div>
        
        <h2>Token Counts Distribution</h2>
        <div id="token-distribution-plot"></div>
        
        <h2>Decoded text (debug)</h2>
        <div class="segment">
            <div class="segment-header">Full decoded text</div>
            <div class="segment-text"><pre style="white-space: pre-wrap; margin: 0;">{html.escape(decoded_full_text or '[empty]')}</pre></div>
        </div>
        
        <h2>Timeline</h2>
        <div class="timeline">
"""
    
    # Add timeline entries
    for i, entry in enumerate(render.get("timeline", [])):
        start_sec = entry.get("start_sec", 0.0)
        end_sec = entry.get("end_sec", 0.0)
        token_count = entry.get("token_count", 0)
        lang_id = entry.get("lang_id", -1)
        duration = end_sec - start_sec if end_sec > start_sec else 0.0
        
        segment_class = "segment"
        if token_count == 0:
            segment_class += " warning"
        
        html_content += f"""
            <div class="{segment_class}">
                <div class="segment-header">
                    Segment {i}: {start_sec:.2f}s - {end_sec:.2f}s (duration: {duration:.2f}s)
                    | Tokens: {token_count} | Lang ID: {lang_id}
                </div>
"""
        if decode_tokenizer and i < len(decoded_texts):
            safe_text = (decoded_texts[i] or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_content += f'                <div class="segment-text">{safe_text}</div>\n'
        html_content += "            </div>\n"
    
    # Add Plotly chart for token distribution
    token_counts = [e.get("token_count", 0) for e in render.get("timeline", [])]
    if token_counts:
        html_content += f"""
        <script>
            var tokenCounts = {json.dumps(token_counts)};
            var trace = {{
                x: tokenCounts,
                type: 'histogram',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Token Counts Distribution',
                xaxis: {{ title: 'Token Count' }},
                yaxis: {{ title: 'Frequency' }}
            }};
            Plotly.newPlot('token-distribution-plot', [trace], layout);
        </script>
"""
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    # Atomic write
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"ASR HTML render saved to {output_path}")
    return output_path

__all__ = ["render_asr_extractor", "render_asr_extractor_html"]
