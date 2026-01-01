import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import librosa
import shutil
import time
import torch
import numpy as np
from einops import rearrange
import soundfile as sf
from transformers import Wav2Vec2Config, Wav2Vec2Model as HF_Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def loudness_norm(audio_array, sr=16000, lufs=-23):
    if pyln is None:
        return audio_array
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio
    except Exception as e:
        logger.warning(f"Loudness normalization failed: {e}")
        return audio_array

def truncate_base64_for_log(base64_str, max_length=50):
    if not base64_str:
        return "None"
    if len(base64_str) <= max_length:
        return base64_str
    return f"{base64_str[:max_length]}... (total {len(base64_str)} chars)"

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

# --- Embedding Model Logic (Ported from modal_infinitetalk_v2) ---

def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

class CustomWav2Vec2Model(HF_Wav2Vec2Model):
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 

    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if not embeddings.hidden_states:
        logger.error("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()

# --- ComfyUI Helper Functions ---

def download_file_from_url(url, output_path):
    try:
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=30", url],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info(f"✅ Downloaded: {url} -> {output_path}")
            return output_path
        else:
            raise Exception(f"wget failed: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        raise

def process_input(input_data, temp_dir, output_filename, input_type):
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
    if input_type == "path": return input_data
    elif input_type == "url": return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        decoded_data = base64.b64decode(input_data)
        with open(file_path, "wb") as f: f.write(decoded_data)
        return file_path
    else: raise Exception(f"Unsupported input type: {input_type}")

def queue_prompt(prompt, input_type="image", person_count="single"):
    url = f"http://{server_address}:8188/prompt"
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_videos(ws, prompt, input_type="image", person_count="single"):
    prompt_id = queue_prompt(prompt, input_type, person_count)["prompt_id"]
    logger.info(f"Executing workflow: prompt_id={prompt_id}")
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break
    history = get_history(prompt_id)[prompt_id]
    output_videos = {}
    for node_id, node_output in history["outputs"].items():
        if "gifs" in node_output:
            output_videos[node_id] = [v["fullpath"] for v in node_output["gifs"]]
    return output_videos

def get_workflow_path(input_type, person_count):
    if input_type == "image":
        return "/I2V_single.json" if person_count == "single" else "/I2V_multi.json"
    return "/V2V_single.json" if person_count == "single" else "/V2V_multi.json"

def calculate_max_frames_from_audio(wav_path, fps=25):
    try:
        duration = librosa.get_duration(path=wav_path)
        return int(duration * fps) + 81
    except Exception:
        return 81

def handler(job):
    job_input = job.get("input", {})
    
    # Payload Alignment
    if "cond_audio" in job_input and isinstance(job_input["cond_audio"], dict):
        if "person1" in job_input["cond_audio"]:
            job_input["wav_url"] = job_input["cond_audio"]["person1"]
    if "cond_video" in job_input:
        job_input["image_url"] = job_input["cond_video"]

    task_id = f"task_{uuid.uuid4()}"
    input_type = job_input.get("input_type", "image")
    person_count = job_input.get("person_count", "single")
    temp_dir = f"/tmp/{task_id}"
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Process Media
    media_url = job_input.get("image_url") or job_input.get("video_url")
    media_path = process_input(media_url, temp_dir, "input_media.jpg", "url") if media_url else "/examples/image.jpg"

    # 2. Process Audio & Generate Embeddings
    wav_url = job_input.get("wav_url")
    wav_path = process_input(wav_url, temp_dir, "input_audio.wav", "url") if wav_url else "/examples/audio.mp3"

    logger.info("Generating audio embeddings...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wav2vec_dir = "/ComfyUI/models/wav2vec" 
    
    try:
        speech_array, sr = librosa.load(wav_path, sr=16000)
        speech_array = loudness_norm(speech_array, sr=sr)
        wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
        wav2vec_model = CustomWav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(device)
        audio_embedding = get_embedding(speech_array, wav2vec_fe, wav2vec_model, device=device)
        emb_path = os.path.join(temp_dir, "audio_embedding.pt")
        torch.save(audio_embedding, emb_path)
        logger.info(f"✅ Embedding saved: {emb_path}")
    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        return {"error": f"Embedding failed: {e}"}

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

    # WebSocket setup
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
