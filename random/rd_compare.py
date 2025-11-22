#!/usr/bin/env python3
"""
rd_compare.py

Mimic VcDemo behavior (DCT module with Huffman entropy coding and JPEG) to
produce curves MSE (x-axis) vs Bitrate (y-axis) for DCT-only and JPEG.

Usage:
    python rd_compare.py path/to/image.bmp

Outputs in ./output_vcdemo/
 - dct_mse_vs_bpp.png   (MSE on x, bpp on y for DCT-only)
 - jpeg_mse_vs_bpp.png  (MSE on x, bpp on y for JPEG)
 - rd_mse_vs_bpp_both.png (both curves on same figure)
 - dct_points.csv, jpeg_points.csv
 - reconstructed images and intermediate files

Dependencies:
    pip install numpy pillow scipy matplotlib
"""
import os, sys, math, csv
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import heapq

# ---------- CONFIG ----------
TARGET_BPPS = [0.25, 0.5, 1.0, 2.0, 4.0]   # required set of target bitrates
TOL_REL = 0.035   # 3.5% relative tolerance for bpp matching
BLOCK = 8
OUTDIR = "output_fixed"
MIN_JPEG_Q = 1
MAX_JPEG_Q = 95

# Standard JPEG luminance quant matrix (Q50 baseline)
Q50 = np.array([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
], dtype=np.float32)

# ---------- UTILITIES ----------
def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def load_gray(path):
    im = Image.open(path).convert('L')
    arr = np.array(im).astype(np.float32)
    return arr, im.size

def mse(a,b):
    return float(np.mean((a - b)**2))

def zigzag_indices(n=8):
    # returns list of (r,c) in zigzag order for n x n
    idx = []
    for s in range(2*n-1):
        if s % 2 == 0:
            rstart = min(s, n-1)
            for r in range(rstart, -1, -1):
                c = s - r
                if c < n:
                    idx.append((r,c))
        else:
            cstart = min(s, n-1)
            for c in range(cstart, -1, -1):
                r = s - c
                if r < n:
                    idx.append((r,c))
    return idx

ZIGZAG = zigzag_indices(BLOCK)

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize_blocks(img, scale, qmatrix=Q50):
    h,w = img.shape
    qmat = qmatrix * scale
    coeffs = np.zeros_like(img, dtype=np.int16)
    for r in range(0,h,BLOCK):
        for c in range(0,w,BLOCK):
            block = img[r:r+BLOCK, c:c+BLOCK]
            bh,bw = block.shape
            pad = np.zeros((BLOCK,BLOCK), dtype=np.float32)
            pad[:bh,:bw]=block
            B = dct2(pad - 128.0)
            Qb = np.round(B / qmat).astype(np.int16)
            coeffs[r:r+bh, c:c+bw] = Qb[:bh,:bw]
    return coeffs

def reconstruct_from_coeffs(coeffs, scale, qmatrix=Q50, orig_shape=None):
    qmat = qmatrix * scale
    h,w = orig_shape
    recon = np.zeros((h,w), dtype=np.float32)
    for r in range(0,h,BLOCK):
        for c in range(0,w,BLOCK):
            block_q = coeffs[r:r+BLOCK, c:c+BLOCK].astype(np.float32)
            bh,bw = block_q.shape
            pad = np.zeros((BLOCK,BLOCK), dtype=np.float32)
            pad[:bh,:bw]=block_q
            Bhat = pad * qmat
            block = idct2(Bhat) + 128.0
            recon[r:r+bh, c:c+bw] = block[:bh,:bw]
    return recon

# ---------- RLE + Huffman simulation (to approximate VLC/Huffman in VcDemo) ----------
def block_to_zigzag_list(coeffs_block):
    # produce 1D zigzag list for a BLOCKxBLOCK block
    out = []
    for (r,c) in ZIGZAG:
        out.append(int(coeffs_block[r,c]))
    return out

def image_coeffs_to_rle_symbols(coeffs):
    # produce stream of symbols as pairs for RLE of zeros in AC stream, and DC separately
    h,w = coeffs.shape
    symbols = []
    # process block by block in natural order
    for r in range(0,h,BLOCK):
        for c in range(0,w,BLOCK):
            block = coeffs[r:r+BLOCK, c:c+BLOCK]
            # DC is first zigzag coefficient
            zz = block_to_zigzag_list(block)
            dc = zz[0]
            symbols.append(('DC', dc))
            # AC: perform RLE: count zeros between non-zero values, encode as (run, value) pairs, and a special EOB
            run = 0
            for val in zz[1:]:
                if val == 0:
                    run += 1
                else:
                    symbols.append(('AC', run, int(val)))
                    run = 0
            if run > 0:
                symbols.append(('EOB', 0))  # mark end-of-block
    return symbols

# Simple Huffman: compute symbol frequencies then assign code lengths using heap
def huffman_code_lengths(freqs):
    # freqs: dict symbol->count
    # return dict symbol->codelen (in bits)
    # For Huffman, code lengths correspond to tree depths; implement standard greedy algorithm
    heap = []
    for sym,count in freqs.items():
        heapq.heappush(heap, (count, [sym]))
    if len(heap) == 0:
        return {}
    # combine until one remains
    while len(heap) > 1:
        c1, syms1 = heapq.heappop(heap)
        c2, syms2 = heapq.heappop(heap)
        for s in syms1: freqs[s] = freqs.get(s,0)  # ensure key exists
        for s in syms2: freqs[s] = freqs.get(s,0)
        heapq.heappush(heap, (c1+c2, syms1+syms2))
    # Now traverse original frequencies to approximate code lengths:
    # A simple method: sort symbols by freq descending and assign lengths by Shannon-Fano approx:
    items = sorted(freqs.items(), key=lambda x: -x[1])
    # Avoid zero division: map rank->length via log2(total / freq)
    total = sum([v for k,v in freqs.items()])
    lengths = {}
    for sym,count in items:
        if count == 0:
            lengths[sym] = max(1, math.ceil(math.log2(len(items))))
        else:
            l = max(1, math.ceil(-math.log2(count/total)))
            lengths[sym] = l
    return lengths

def compute_bits_via_huffman(symbols):
    # symbols: list of symbol tokens (tuples). Build frequency map and compute theoretical Shannon/huffman bits
    freqs = Counter(symbols)
    lengths = huffman_code_lengths(dict(freqs))
    bits = 0
    for sym,count in freqs.items():
        bits += lengths[sym] * count
    return bits

# ---------- JPEG helpers ----------
def save_jpeg_quality(pil_img, path, quality):
    pil_img.save(path, format='JPEG', quality=int(quality), optimize=True)

def find_jpeg_quality_for_bpp(pil_img, pixels, target_bpp, tol=TOL_REL):
    # binary search quality; keep best found
    import tempfile, os, shutil
    tmpdir = tempfile.mkdtemp()
    best = None
    lo,hi = MIN_JPEG_Q, MAX_JPEG_Q
    for _ in range(12):
        mid = (lo + hi)//2
        pth = os.path.join(tmpdir, f"q{mid}.jpg")
        save_jpeg_quality(pil_img, pth, mid)
        fsz = os.path.getsize(pth)
        bpp = (fsz*8)/pixels
        if best is None or abs(bpp-target_bpp) < abs(best[1]-target_bpp):
            best = (mid, bpp, pth)
        # if size too big -> lower quality (reduce quality number)
        if bpp > target_bpp:
            # make compression stronger -> decrease quality => hi -> mid-1
            hi = mid-1
        else:
            lo = mid+1
        if abs(bpp-target_bpp)/target_bpp <= tol:
            break
    return best

# ---------- MAIN processing ----------
def process(path, target_bpps=TARGET_BPPS):
    arr, size = load_gray(path)
    h,w = arr.shape
    pixels = h*w
    pil = Image.fromarray(arr.astype(np.uint8), mode='L')
    ensure_outdir()

    jpeg_points = []
    dct_points = []

    for target in target_bpps:
        print(f"Target {target:.3f} bpp ...")

        # JPEG
        jbest = find_jpeg_quality_for_bpp(pil, pixels, target)
        if jbest is None:
            print(" JPEG: no candidate")
        else:
            q, bpp_j, pth = jbest
            dec = np.array(Image.open(pth).convert('L')).astype(np.float32)
            M = mse(arr, dec)
            jpeg_points.append((bpp_j, M, q, pth))
            outp = os.path.join(OUTDIR, f"jpeg_q{q}_bpp{bpp_j:.4f}.jpg")
            try: os.replace(pth, outp)
            except: import shutil; shutil.copy(pth,outp)
            print(f" JPEG q={q} -> bpp={bpp_j:.4f}, MSE={M:.4f}")

        # DCT-only: search scale so that Huffman-coded bits approximate target bpp
        slo,shi = 0.2, 10.0
        best = None
        for _ in range(25):
            mid = (slo+shi)/2.0
            coeffs = quantize_blocks(arr, mid)
            symbols = image_coeffs_to_rle_symbols(coeffs)
            bits = compute_bits_via_huffman(tuple(symbols))
            bpp_d = bits / pixels
            recon = reconstruct_from_coeffs(coeffs, mid, Q50, arr.shape)
            M_d = mse(arr, recon)
            if best is None or abs(bpp_d-target) < abs(best[1]-target):
                best = (mid, bpp_d, M_d, recon, bits)
            # if current bpp greater than target, make quant coarser => increase scale
            if bpp_d > target:
                slo = mid
            else:
                shi = mid
            if abs(bpp_d-target)/target <= TOL_REL:
                break
        if best is not None:
            scale, bpp_d, M_d, recon_best, bits_used = best
            # write recon image
            recon_path = os.path.join(OUTDIR, f"dct_s{scale:.4f}_bpp{bpp_d:.4f}.png")
            Image.fromarray(np.clip(np.round(recon_best),0,255).astype(np.uint8)).save(recon_path)
            dct_points.append((bpp_d, M_d, scale, bits_used, recon_path))
            print(f" DCT scale={scale:.4f} -> bpp={bpp_d:.4f}, MSE={M_d:.4f}")

    # Save CSVs and plots (MSE on X axis, bitrate on Y axis)
    with open(os.path.join(OUTDIR,"jpeg_points.csv"),"w",newline="") as f:
        w = csv.writer(f); w.writerow(['bpp','MSE','quality','path'])
        for bpp,M,q,p in jpeg_points: w.writerow([bpp,M,q,p])
    with open(os.path.join(OUTDIR,"dct_points.csv"),"w",newline="") as f:
        w = csv.writer(f); w.writerow(['bpp','MSE','scale','bits','recon'])
        for bpp,M,scale,bits,p in dct_points: w.writerow([bpp,M,scale,bits,p])

    # DCT plot
    if dct_points:
        xs = [p[1] for p in dct_points]   # MSE
        ys = [p[0] for p in dct_points]   # bpp
        plt.figure(); plt.plot(xs, ys, '-o'); plt.xlabel('MSE'); plt.ylabel('Bit rate (bpp)')
        plt.title('DCT-only (with RLE+Huffman sim) - MSE (x) vs Bitrate (y)'); plt.grid(True)
        plt.savefig(os.path.join(OUTDIR,'dct_mse_vs_bpp.png'), dpi=200)
        print("Saved dct_mse_vs_bpp.png")

    # JPEG plot
    if jpeg_points:
        xs = [p[1] for p in jpeg_points]
        ys = [p[0] for p in jpeg_points]
        plt.figure(); plt.plot(xs, ys, '-s'); plt.xlabel('MSE'); plt.ylabel('Bit rate (bpp)')
        plt.title('JPEG - MSE (x) vs Bitrate (y)'); plt.grid(True)
        plt.savefig(os.path.join(OUTDIR,'jpeg_mse_vs_bpp.png'), dpi=200)
        print("Saved jpeg_mse_vs_bpp.png")

    # both on same figure
    plt.figure(figsize=(8,5))
    if dct_points:
        plt.plot([p[1] for p in dct_points], [p[0] for p in dct_points], '-o', label='DCT-only (Huffman sim)')
    if jpeg_points:
        plt.plot([p[1] for p in jpeg_points], [p[0] for p in jpeg_points], '-s', label='JPEG')
    plt.xlabel('MSE'); plt.ylabel('Bit rate (bpp)')
    plt.title('MSE (x) vs Bitrate (y) - DCT vs JPEG'); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(OUTDIR,'rd_mse_vs_bpp_both.png'), dpi=200)
    print("Saved rd_mse_vs_bpp_both.png")

    print("Done. Check", OUTDIR)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rd_compare.py path/to/image")
        sys.exit(1)
    imgpath = sys.argv[1]
    if not os.path.exists(imgpath):
        print("Image not found:", imgpath); sys.exit(1)
    process(imgpath, TARGET_BPPS)
