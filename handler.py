import os
import json
import base64
import time
import requests
import subprocess
import shutil
import websocket
import uuid
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import pyloudnorm as pyln
from huggingface_hub import hf_hub_download
import logging

import runpod
from runpod.serverless.utils import rp_download, rp_cleanup, rp_upload
from runpod.serverless.utils.rp_validator import Validator

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup for HF Transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Model Configuration ---
MODELS = [
    {"repo": "Kijai/WanVideo_comfy_fp8_scaled", "filename": "InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors", "subfolder": "", "target": "/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"},
    {"repo": "Kijai/WanVideo_comfy_fp8_scaled", "filename": "InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors", "subfolder": "", "target": "/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"},
    {"repo": "Kijai/WanVideo_comfy", "filename": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", "subfolder": "", "target": "/ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"},
    {"repo": "Kijai/WanVideo_comfy", "filename": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "subfolder": "", "target": "/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"},
    {"repo": "Kijai/WanVideo_comfy", "filename": "Wan2_1_VAE_bf16.safetensors", "subfolder": "", "target": "/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors"},
    {"repo": "Kijai/WanVideo_comfy", "filename": "umt5-xxl-enc-bf16.safetensors", "subfolder": "", "target": "/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors"},
    {"repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/clip_vision/clip_vision_h.safetensors", "subfolder": "", "target": "/ComfyUI/models/clip_vision/clip_vision_h.safetensors"},
    {"repo": "Kijai/MelBandRoformer_comfy", "filename": "MelBandRoformer_fp16.safetensors", "subfolder": "", "target": "/ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors"}
]

def download_models():
    """Download models if they don't exist, leveraging HF Transfer."""
    for model in MODELS:
        target_path = model["target"]
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
            logger.info(f"✅ Model exists: {target_path}")
            continue
        
        logger.info(f"⏳ Downloading {model['filename']} from {model['repo']}...")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        try:
            downloaded = hf_hub_download(repo_id=model["repo"], filename=model["filename"])
            shutil.copy(downloaded, target_path)
            logger.info(f"✅ Downloaded to: {target_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download {model['filename']}: {e}")

# Helpers from modal_infinitetalk_v2
class CustomWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, extract_features, attention_mask=None):
        res = self.encoder(extract_features, attention_mask=attention_mask)
        return res.last_hidden_state

def linear_interpolation(features, target_len):
    if features.shape[0] == target_len: return features
    features = features.transpose(0, 1).unsqueeze(0)
    features = F.interpolate(features, size=target_len, mode='linear', align_corners=False)
    return features.squeeze(0).transpose(0, 1)

def get_embedding(speech_array, wav2vec_fe, wav2vec_model, device="cuda"):
    inputs = wav2vec_fe(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        extract_features = wav2vec_model.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        res = wav2vec_model(extract_features)
    return linear_interpolation(res, target_len=int(len(speech_array)/16000 * 30))

def loudness_norm(audio_path, target_db=-23.0):
    data, rate = librosa.load(audio_path, sr=None)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    normed = pyln.normalize.loudness(data, loudness, target_db)
    librosa.output.write_wav(audio_path, normed, rate) if hasattr(librosa, 'output') else None # Older librosa
    import soundfile as sf
    sf.write(audio_path, normed, rate)

def get_workflow_path(input_type, person_count):
    if input_type == "image": return "I2V_single.json"
    return "V2V_single.json"

def calculate_max_frames_from_audio(wav_path, fps=30):
    duration = librosa.get_duration(filename=wav_path)
    return int(duration * fps)

# Initialization
download_models()
server_address = "127.0.0.1"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = requests.post(f"http://{server_address}:8188/prompt", data=data)
    return req.json()

def get_videos(ws, prompt, input_type, person_count):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else: continue
    
    history_req = requests.get(f"http://{server_address}:8188/history/{prompt_id}")
    history = history_req.json()[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'gifs' in node_output: output_videos[node_id] = [os.path.join("/ComfyUI/output", x['filename']) for x in node_output['gifs']]
    return output_videos

def handler(job):
    job_input = job["input"]
    temp_dir = f"/tmp/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Payload Mapping
    wav_url = job_input.get("cond_audio", {}).get("person1")
    image_url = job_input.get("cond_video")
    input_type = job_input.get("input_type", "image")
    person_count = job_input.get("person_count", 1)
    
    if not wav_url or not image_url:
        return {"error": "Missing cond_audio.person1 or cond_video"}

    # 1. Download Media
    wav_path = os.path.join(temp_dir, "input.wav")
    media_path = os.path.join(temp_dir, "input_media")
    subprocess.run(["wget", "-q", wav_url, "-O", wav_path])
    subprocess.run(["wget", "-q", image_url, "-O", media_path])
    
    # 2. Audio Processing
    try:
        loudness_norm(wav_path)
        speech_array, _ = librosa.load(wav_path, sr=16000)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        wav2vec_dir = "/ComfyUI/models/wav2vec"
        wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
        wav2vec_model = CustomWav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(device)
        audio_embedding = get_embedding(speech_array, wav2vec_fe, wav2vec_model, device=device)
        emb_path = os.path.join(temp_dir, "audio_embedding.pt")
        torch.save(audio_embedding, emb_path)
    except Exception as e:
        return {"error": f"Audio processing failed: {e}"}

    # 3. ComfyUI Interaction
    workflow = json.load(open(get_workflow_path(input_type, person_count), "r"))
    if input_type == "image": workflow["284"]["inputs"]["image"] = media_path
    else: workflow["228"]["inputs"]["video"] = media_path
    
    workflow["125"]["inputs"]["audio"] = wav_path
    workflow["194"]["inputs"]["audio_1"] = emb_path
    workflow["241"]["inputs"]["positive_prompt"] = job_input.get("prompt", "a person talking")
    workflow["245"]["inputs"]["value"] = job_input.get("width", 512)
    workflow["246"]["inputs"]["value"] = job_input.get("height", 512)
    workflow["270"]["inputs"]["value"] = job_input.get("max_frame") or calculate_max_frames_from_audio(wav_path)

    ws = websocket.WebSocket()
    for _ in range(30):
        try:
            ws.connect(f"ws://{server_address}:8188/ws?clientId={client_id}")
            break
        except Exception: time.sleep(2)
    
    try:
        videos = get_videos(ws, workflow, input_type, person_count)
        ws.close()
        for node_id in videos:
            if videos[node_id]:
                output_video_path = videos[node_id][0]
                with open(output_video_path, "rb") as f:
                    return {"video": base64.b64encode(f.read()).decode("utf-8")}
    except Exception as e:
        return {"error": f"ComfyUI processing failed: {e}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
