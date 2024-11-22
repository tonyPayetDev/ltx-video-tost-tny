import os, shutil, json, requests, random, time, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
import comfy_extras
from comfy_extras import  nodes_images, nodes_lt, nodes_custom_sampler

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

    if noise_seed == 0:
        random.seed(int(time.time()))
        noise_seed = random.randint(0, 18446744073709551615)
    print(noise_seed)

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
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(source):
            os.remove(source)
        if os.path.exists(destination):
            os.remove(destination)

runpod.serverless.start({"handler": generate})