# --- Image Creation and PDF Conversion Integration (Instructions) ---
# To enable the model or pipeline to create images and convert to/from PDF, you can integrate:
# 1. Image generation APIs (e.g., Stable Diffusion, DALL-E, OpenAI, Replicate) or use PIL for basic image creation.
# 2. PDF generation/conversion libraries (e.g., reportlab, fpdf, PyPDF2, pdfplumber).
#
# Example: Create an image with PIL
# from PIL import Image, ImageDraw
# img = Image.new('RGB', (256, 256), color = 'white')
# d = ImageDraw.Draw(img)
# d.text((10,10), "Hello World", fill=(0,0,0))
# img.save('hello.png')
#
# Example: Generate a PDF with reportlab
# from reportlab.pdfgen import canvas
# c = canvas.Canvas("output.pdf")
# c.drawString(100, 750, "Hello PDF")
# c.save()
#
# Example: Convert PDF to text with PyPDF2
# import PyPDF2
# with open('file.pdf', 'rb') as f:
#     reader = PyPDF2.PdfReader(f)
#     text = " ".join(page.extract_text() for page in reader.pages)
#     print(text)
#
# Note: For training, you can generate synthetic image/PDF data or enable your inference pipeline to call these tools live.

import os
import subprocess

def download_math_dataset():
    if not os.path.exists("math"):
        print("Cloning Hendrycks MATH dataset...")
        subprocess.run(["git", "clone", "https://github.com/hendrycks/math.git"])
    else:
        print("MATH dataset already exists.")

def download_gsm8k():
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run(["pip", "install", "datasets"])
        from datasets import load_dataset
    print("Downloading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main")
    gsm8k['train'].to_json("gsm8k_train.jsonl")
    gsm8k['test'].to_json("gsm8k_test.jsonl")
    print("GSM8K dataset downloaded and saved as JSONL.")

def download_aqua_rat():
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run(["pip", "install", "datasets"])
        from datasets import load_dataset
    print("Downloading AQUA-RAT dataset...")
    aqua = load_dataset("aqua_rat")
    aqua['train'].to_json("aqua_train.jsonl")
    aqua['test'].to_json("aqua_test.jsonl")
    print("AQUA-RAT dataset downloaded and saved as JSONL.")

def download_svamp():
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run(["pip", "install", "datasets"])
        from datasets import load_dataset
    print("Downloading SVAMP dataset...")
    svamp = load_dataset("svamp")
    svamp['train'].to_json("svamp_train.jsonl")
    svamp['test'].to_json("svamp_test.jsonl")
    print("SVAMP dataset downloaded and saved as JSONL.")

if __name__ == "__main__":
    download_math_dataset()
    download_gsm8k()
    download_aqua_rat()
    download_svamp()
