import zipfile
import os
import json
import time
import re
import pytesseract
import cv2
import numpy as np
from ftfy import fix_text
from unidecode import unidecode
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
from multiprocessing import Pool
from functools import partial
from PIL import Image

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# === PATH SETUP ===
zip_path = "sample.zip"
extract_dir = "extracted_data"

# === UNZIP FILE ===
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("✅ ZIP berhasil diekstrak ke:", extract_dir)

# === FUNGSI OCR ===
def extract_text_from_pdf(path):
    try:
        text = extract_text(path)
        return text
    except Exception as e:
        print("Error:", e)
        return ""

# text dominant
def text_density(img):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    text_area = 0
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            w, h = data['width'][i], data['height'][i]
            text_area += w * h
    total_area = img.width * img.height
    return text_area / total_area

def fast_text_density(pil_img):
    img = np.array(pil_img.convert('L'))  # ubah ke grayscale
    
    # Threshold biar teks jadi hitam, background putih
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Hitung rasio area hitam terhadap keseluruhan area
    text_pixels = np.count_nonzero(binary)
    total_pixels = binary.size
    density = text_pixels / total_pixels
    
    return density
    

def cleaning_ocr_text(text: str) -> str:
    text = fix_text(text)
    text = unidecode(text)
    # text = text.lower()
    text = re.sub(r'[_\-–—=]+', ' ', text)
    text = re.sub(r'([a-z])\s*-\s*([a-z])', r'\1\2', text)
    text = re.sub(r'([a-z])\1{2,}', r'\1', text)
    text = re.sub(r'(ee|ae|oe|ie){2,}', ' ', text)
    text = re.sub(r'[^\w\s.,:()/\-]', ' ', text)
    text = re.sub(r'(\d)\s*[\.,]\s*(\d)', r'\1.\2', text)
    text = re.sub(r'(\d)\s*:\s*(\d)', r'\1:\2', text)
    text = re.sub(r'\s*r\s*p\s*', ' rp', text)
    text = re.sub(r'\s+[a-zA-Z0-9]\s+', ' ', text)
    text = re.sub(r'(\d):(\d)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b[a-zA-Z]{25,}\b', '', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'([a-z]{2,})([A-Z]{2,})', r'\1 \2', text)
    text = re.sub(r'\.{2,}', '. ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:()\-\n\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*\.\s*', '. ', text)
    return text.strip()

def normalize_ocr_text(text: str) -> str:
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def process_page(page, lang='ind+eng'):
    text  = pytesseract.image_to_string(page, lang=lang)
    text = re.sub(r'\s+', ' ', text).strip()
    # text = cleaning_ocr_text(text)
    # text = normalize_ocr_text(text)
    return text


def ocr_pdf_cached(path, dpi=300, cache_dir=None, lang='ind+eng', num_processes=2, char_limit=50000):
    if cache_dir is None:
        project_root = os.path.dirname(os.path.abspath(_file_))
        cache_dir = os.path.join(project_root, "ocr_cache")
        os.makedirs(cache_dir, exist_ok=True)

    pdf_name = os.path.splitext(os.path.basename(path))[0]
    cache_subdir = os.path.join(cache_dir, pdf_name)
    os.makedirs(cache_subdir, exist_ok=True)

    # Konversi ke gambar hanya jika belum ada cache
    existing_imgs = [f for f in os.listdir(cache_subdir) if f.endswith('.jpg')]
    if not existing_imgs:
        print(f"🔄 Konversi PDF ke gambar untuk {pdf_name} ...")
        start_time = time.time()

        pages = convert_from_path(path, dpi=dpi, fmt='jpeg', last_page=8, thread_count=4)
        for i, page in enumerate(pages):
            image_path = os.path.join(cache_subdir, f"page_{i+1:03d}.jpg")
            page.save(image_path, 'JPEG')
        print(f"convert_from_path: {time.time() - start_time:.2f} detik")
        start_time = time.time()

        # hapus pdf yang bukan dominan teks, seperti gambar (peta, tabel, foto)
        cached_images = [os.path.join(cache_subdir, f) for f in sorted(os.listdir(cache_subdir)) if f.endswith('.jpg')]
        kept_pages = [cached_images[0]] # anggep saja halaman pertama (cover / informasi penting)
        cached_images = cached_images[1:]
        
        for cache_image in cached_images:
            img = Image.open(cache_image)
            ratio = fast_text_density(img)
            filename = cache_image.split('/')[-1]
            # print(f"\n[INFO] {filename}: text_ratio={ratio:.4f}")
            if ratio > 0.09:  # threshold for text density
                os.remove(cache_image)
                # print(f"🗑️  Dihapus: {filename} (dominan gambar (peta, tabel, foto))")
                
        print(f"cek dominan gambar: {time.time() - start_time:.2f} detik")
    else:
        print(f"🗂 Gunakan cache gambar untuk {pdf_name} ...")

    cached_images = [os.path.join(cache_subdir, f) for f in sorted(os.listdir(cache_subdir)) if f.endswith('.jpg')]
    # cached_images = [cached_images[0]]
    cached_images = cached_images[:5]
    # return ''
    # print(json.dumps(cached_images, indent=2, ensure_ascii=False))

    texts = []
    total_chars = 0
    process_func = partial(process_page, lang=lang)

    with Pool(processes=num_processes) as pool:
        for text in pool.imap(process_func, cached_images):
            if total_chars >= char_limit:
                print("Limit 50.000 karakter tercapai, hentikan proses OCR lebih lanjut.")
                break
            total_chars += len(text)
            texts.append(text)

        return "\n".join(texts)


def extract_best_text(path):
    start_time = time.time()
    text = extract_text_from_pdf(path)
    if not text or len(text.strip()) < 50:
        start_time = time.time()
        text = ocr_pdf_cached(path)
        
    return {
        "text": text,
        "length": len(text),
        "time_taken": time.time() - start_time
    }


# === EKSEKUSI ===
pdf_folder = extract_dir
all_texts = []

for fname in os.listdir(pdf_folder):
    if fname.lower().endswith(".pdf"):
        path = os.path.join(pdf_folder, fname)
        response = extract_best_text(path)
        all_texts.append({
            "filename": fname,
            "time": f"{response['time_taken']:.2f} detik",
            "length": response['length'],
            "text": response['text'],  # Batasi output teks untuk tampilan
        })

print(json.dumps(all_texts, indent=2, ensure_ascii=False))