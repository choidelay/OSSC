// static/script.js
async function generateImage() {
    const prompt = document.getElementById("prompt").value;
    const guidanceScale = parseFloat(document.getElementById("guidance_scale").value) || 7.5;
    const numInferenceSteps = parseInt(document.getElementById("num_inference_steps").value) || 50;
    const useLoRA = document.getElementById("use_lora").checked;

    const response = await fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            prompt: prompt,
            guidance_scale: guidanceScale,
            num_inference_steps: numInferenceSteps,
            use_lora: useLoRA
        })
    });

    const data = await response.json();
    document.getElementById("result").innerText = data.message;
    const img = document.getElementById("generated-image");
    img.src = data.image_path;
    img.style.display = "block";
}

async function uploadImage() {
    const fileInput = document.getElementById("file");
    const prompt = document.getElementById("upload_prompt").value;

    if (fileInput.files.length === 0) {
        alert("Please select an image file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("prompt", prompt);

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("result").innerText = data.message;
    const img = document.getElementById("generated-image");
    img.src = data.image_path;
    img.style.display = "block";
}

async function inpaintImage() {
    const fileInput = document.getElementById("inpaint_file");
    const maskInput = document.getElementById("inpaint_mask");
    const prompt = document.getElementById("inpaint_prompt").value;

    if (fileInput.files.length === 0 || maskInput.files.length === 0) {
        alert("Please select an image and a mask file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("mask", maskInput.files[0]);
    formData.append("prompt", prompt);

    const response = await fetch("/inpaint", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("result").innerText = data.message;
    const img = document.getElementById("generated-image");
    img.src = data.image_path;
    img.style.display = "block";
}