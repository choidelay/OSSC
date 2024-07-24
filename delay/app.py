# app.py
from flask import Flask, request, jsonify, render_template
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import os
from PIL import Image

app = Flask(__name__)

# Stable Diffusion 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)

img2img_model_id = "stabilityai/stable-diffusion-xl-img2img-1.0"
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(img2img_model_id)
img2img_pipe.to(device)

# Inpainting 모델 로드
inpainting_model_id = "stabilityai/stable-diffusion-xl-inpainting-1.0"
inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(inpainting_model_id)
inpainting_pipe.to(device)

# LoRA 설정 (LoRA 모델 ID를 실제 사용 중인 것으로 변경해야 할 수 있습니다)
lora_model_id = "your-lora-model-id"
lora_pipe = StableDiffusionPipeline.from_pretrained(lora_model_id)
lora_pipe.to(device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    guidance_scale = data.get("guidance_scale", 7.5)
    num_inference_steps = data.get("num_inference_steps", 50)
    use_lora = data.get("use_lora", False)

    pipe_to_use = lora_pipe if use_lora else pipe

    with autocast(device):
        image = pipe_to_use(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]

    image_path = os.path.join("static/generated_images", "generated_image.png")
    image.save(image_path)
    return jsonify({"message": "Image generated", "image_path": image_path})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400
    if file:
        file_path = os.path.join("static/uploads", "uploaded_image.png")
        file.save(file_path)
        
        prompt = request.form.get("prompt", "")
        with open(file_path, "rb") as f:
            uploaded_image = Image.open(f).convert("RGB")

        with autocast(device):
            generated_image = img2img_pipe(prompt, init_image=uploaded_image).images[0]

        generated_image_path = os.path.join("static/generated_images", "generated_image.png")
        generated_image.save(generated_image_path)
        return jsonify({"message": "Image generated", "image_path": generated_image_path})

@app.route("/inpaint", methods=["POST"])
def inpaint():
    if "file" not in request.files or "mask" not in request.files:
        return jsonify({"message": "No file or mask part"}), 400
    file = request.files["file"]
    mask = request.files["mask"]
    if file.filename == "" or mask.filename == "":
        return jsonify({"message": "No selected file or mask"}), 400
    
    file_path = os.path.join("static/uploads", "inpaint_image.png")
    mask_path = os.path.join("static/uploads", "mask_image.png")
    
    file.save(file_path)
    mask.save(mask_path)
    
    prompt = request.form.get("prompt", "")
    with open(file_path, "rb") as f:
        inpaint_image = Image.open(f).convert("RGB")
    
    with open(mask_path, "rb") as m:
        mask_image = Image.open(m).convert("L")
    
    with autocast(device):
        inpainted_image = inpainting_pipe(prompt, image=inpaint_image, mask_image=mask_image).images[0]
    
    inpainted_image_path = os.path.join("static/inpaint_images", "inpainted_image.png")
    inpainted_image.save(inpainted_image_path)
    return jsonify({"message": "Image inpainted", "image_path": inpainted_image_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

