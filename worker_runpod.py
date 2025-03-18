import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np

from nodes import NODE_CLASS_MAPPINGS, load_custom_node
from comfy_extras import  nodes_images, nodes_lt, nodes_custom_sampler
# Variables de connexion à Supabase 
SUPABASE_URL = "https://rvsykocedohfdfdvbrfe.supabase.co"
SUPABASE_API_KEY = "${SUPA_ROLE_TOKEN}"
SUPABASE_BUCKET = "video"

# Fonction d'envoi de la vidéo à Supabase
def upload_to_supabase(video_path, file_name):
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{file_name}"
    
    headers = {
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
    }

    with open(video_path, "rb") as file:
        files = {"file": file}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        print("Video uploaded successfully!")
        return response.json()
    else:
        print(f"Error uploading video: {response.status_code}, {response.text}")
        return {"error": response.text} 


load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Fluxpromptenhancer")

FluxPromptEnhance = NODE_CLASS_MAPPINGS["FluxPromptEnhance"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LTXVImgToVideo = nodes_lt.NODE_CLASS_MAPPINGS["LTXVImgToVideo"]()
LTXVConditioning = nodes_lt.NODE_CLASS_MAPPINGS["LTXVConditioning"]()
SamplerCustom = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustom"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
LTXVScheduler = nodes_lt.NODE_CLASS_MAPPINGS["LTXVScheduler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
SaveAnimatedWEBP = nodes_images.NODE_CLASS_MAPPINGS["SaveAnimatedWEBP"]()

with torch.inference_mode():
    clip = CLIPLoader.load_clip("t5xxl_fp16.safetensors", type="ltxv")[0]
    model, _, vae = CheckpointLoaderSimple.load_checkpoint("ltx-video-2b-v0.9.safetensors")

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def webp_to_mp4(input_webp, output_mp4, fps=10):
    with Image.open(input_webp) as img:
        frames = []
        try:
            while True:
                frame = img.copy()
                frames.append(frame)
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    temp_frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        frame.save(frame_path)
        temp_frame_paths.append(frame_path)
    clip = ImageSequenceClip(temp_frame_paths, fps=fps)
    clip.write_videofile(output_mp4, codec="libx264", fps=fps)
    for path in temp_frame_paths:
        os.remove(path)
    os.rmdir(temp_dir)

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    frame_rate = values['frame_rate']
    sampler_name = values['sampler_name']
    steps = values['steps']
    max_shift = values['max_shift']
    base_shift = values['base_shift']
    stretch = values['stretch']
    terminal = values['terminal']
    width = values['width']
    height = values['height']
    length = values['length']
    add_noise = values['add_noise']
    noise_seed = values['noise_seed']
    cfg = values['cfg']
    fps = values['fps']
    prompt_enhance = values['prompt_enhance']

    if noise_seed == 0:
        random.seed(int(time.time()))
        noise_seed = random.randint(0, 18446744073709551615)
    print(noise_seed)

    if prompt_enhance:
        positive_prompt = FluxPromptEnhance.enhance_prompt(positive_prompt, noise_seed)[0]
        print(positive_prompt)
    conditioning_positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    conditioning_negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    image = LoadImage.load_image(input_image)[0]
    positive, negative, latent_image = LTXVImgToVideo.generate(conditioning_positive, conditioning_negative, image, vae, width, height, length, batch_size=1)
    positive, negative = LTXVConditioning.append(positive, negative, frame_rate)
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = LTXVScheduler.get_sigmas(steps, max_shift, base_shift, stretch, terminal, latent=None)[0]
    samples = SamplerCustom.sample(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)[0]
    images = VAEDecode.decode(vae, samples)[0].detach()
    video = SaveAnimatedWEBP.save_images(images, fps, filename_prefix=f"ltx-video-{noise_seed}-tost", lossless=False, quality=90, method="default")
    source = video['ui']['images'][0]['filename']
    source = f"/content/ComfyUI/output/{source}"
    destination = f"/content/ltx-video-{noise_seed}-tost.webp"
    shutil.move(source, destination)
    webp_to_mp4(f"/content/ltx-video-{noise_seed}-tost.webp", f"/content/ltx-video-{noise_seed}-tost.mp4", fps=fps)
    
    result = f"/content/ltx-video-{noise_seed}-tost.mp4"
   
    # Vérification si le fichier existe
    if not os.path.exists(result):
        return {"status": "FAILED", "error": f"File {video_path} not found"}

    try:
        # Upload sur Supabase
        file_name = f"ltx-video-{seed}.mp4"
        upload_response = upload_to_supabase(result, file_name)

        # Vérification du retour de Supabase
        if "error" in upload_response:
            raise Exception(upload_response["error"])

        return {"status": "DONE", "supabase_response": upload_response}

    except Exception as e:
        return {"status": "FAILED", "error": str(e)}

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# Lancement du serveur Runpod
runpod.serverless.start({"handler": generate})
