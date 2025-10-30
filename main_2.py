import os
import json
import time
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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


# === KONFIGURASI DASAR ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# === PATH SETUP ===
extract_dir = "extracted_data"
if not os.path.exists(extract_dir):
    raise FileNotFoundError(f"Folder '{extract_dir}' tidak ditemukan! Pastikan sudah ada folder dengan PDF di dalamnya.")
print("✅ Membaca PDF langsung dari folder:", extract_dir)


# === FUNGSI DASAR ===
def extract_text_from_pdf(path):
    """Ekstrak teks langsung dari PDF (tanpa OCR)."""
    try:
        return extract_text(path)
    except Exception as e:
        print("Error extract_text_from_pdf:", e)
        return ""


def fast_text_density(pil_img):
    """Hitung rasio area teks terhadap keseluruhan gambar (lebih cepat)."""
    img = np.array(pil_img.convert('L'))
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    return np.count_nonzero(binary) / binary.size


# === PEMBERSIHAN TEKS ===
def cleaning_ocr_text(text: str) -> str:
    """Membersihkan teks hasil OCR agar hanya menyisakan kata baku (hasil stemming)."""
    text = fix_text(text)
    text = unidecode(text)
    text = re.sub(r'[_\-–—=•·]+', ' ', text)
    text = re.sub(r'([a-z])\s*-\s*([a-z])', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:()\-\n\s]', ' ', text)

    # Pisahkan kata dan lakukan stemming
    words = text.lower().split()
    clean_words = []

    for w in words:
        w = w.strip(".,:;!?()[]")
        if not w:
            continue
        if len(w) <= 2 or w.isdigit():
            continue
        if w in stopwords:
            continue
        # Ubah ke kata dasar baku
        w_stem = stemmer.stem(w)
        clean_words.append(w_stem)

    text = ' '.join(clean_words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# === OCR DENGAN PREPROCESSING ===
def process_page(image_path, lang='ind+eng'):
    """Lakukan OCR pada satu halaman (gambar) dengan preprocessing & fallback otomatis."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Gagal membaca gambar")

        # --- PREPROCESSING ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 11
        )
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)

        # --- OCR UTAMA ---
        text = pytesseract.image_to_string(gray, lang=lang, config="--psm 6")
        text = re.sub(r'\s+', ' ', text).strip()

        # --- Fallback jika hasilnya kacau ---
        if len(text) < 10 or re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', text):
            alt = pytesseract.image_to_string(gray, lang=lang, config="--psm 7")
            alt = re.sub(r'\s+', ' ', alt).strip()
            if len(alt) > len(text):
                text = alt

        return text

    except Exception as e:
        print(f"❌ Gagal OCR {image_path}: {e}")
        return ""


# === OCR PDF DENGAN CACHE ===
def ocr_pdf_cached(path, dpi=300, cache_dir=None, lang='ind+eng', num_processes=2, word_limit=25000):
    """OCR PDF dengan cache gambar agar cepat saat dijalankan ulang."""
    if cache_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(project_root, "ocr_cache")
        os.makedirs(cache_dir, exist_ok=True)

    pdf_name = os.path.splitext(os.path.basename(path))[0]
    cache_subdir = os.path.join(cache_dir, pdf_name)
    os.makedirs(cache_subdir, exist_ok=True)

    existing_imgs = [f for f in os.listdir(cache_subdir) if f.endswith('.jpg')]
    if not existing_imgs:
        print(f"🔄 Konversi PDF ke gambar untuk {pdf_name} ...")
        start_time = time.time()

        # hanya ambil 5 halaman pertama
        pages = convert_from_path(path, dpi=dpi, fmt='jpeg', last_page=5, thread_count=4)
        for i, page in enumerate(pages):
            image_path = os.path.join(cache_subdir, f"page_{i+1:03d}.jpg")
            page.save(image_path, 'JPEG')

        print(f"convert_from_path: {time.time() - start_time:.2f} detik")

        # Filter halaman dominan gambar
        start_time = time.time()
        cached_images = [os.path.join(cache_subdir, f)
                         for f in sorted(os.listdir(cache_subdir)) if f.endswith('.jpg')]
        for cache_image in cached_images[1:]:
            img = Image.open(cache_image)
            ratio = fast_text_density(img)
            if ratio > 0.09:
                os.remove(cache_image)
        print(f"cek dominan gambar: {time.time() - start_time:.2f} detik")
    else:
        print(f"🗂 Gunakan cache gambar untuk {pdf_name} ...")

    # Ambil maksimal 5 halaman
    cached_images = [os.path.join(cache_subdir, f)
                     for f in sorted(os.listdir(cache_subdir)) if f.endswith('.jpg')][:5]

    texts = []
    total_words = 0
    process_func = partial(process_page, lang=lang)

    with Pool(processes=num_processes) as pool:
        for text in pool.imap(process_func, cached_images):
            cleaned = cleaning_ocr_text(text)
            word_count = len(cleaned.split())
            if total_words + word_count > word_limit:
                print("⚠️ Limit 50.000 kata tercapai, hentikan OCR.")
                break
            total_words += word_count
            texts.append(cleaned)

    return " ".join(texts)


def extract_best_text(path):
    """Pilih metode terbaik antara PDFMiner atau OCR + cleaning."""
    start_time = time.time()
    text = extract_text_from_pdf(path)
    if not text or len(text.strip()) < 50:
        text = ocr_pdf_cached(path)

    text = cleaning_ocr_text(text)
    return {"text": text, "length": len(text.split()), "time_taken": time.time() - start_time}


# === EKSEKUSI UTAMA ===
if __name__ == "__main__":
    pdf_folder = extract_dir
    all_texts = []

    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, fname)
            print(f"\n📄 Memproses: {fname}")
            response = extract_best_text(path)
            all_texts.append({
                "filename": fname,
                "time": f"{response['time_taken']:.2f} detik",
                "length": response['length'],
                "text": response['text'],
            })

    print("\n=== HASIL AKHIR ===")
    print(json.dumps(all_texts, indent=2, ensure_ascii=False))
