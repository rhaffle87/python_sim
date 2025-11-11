import numpy as np

# Try to get DCT/IDCT implementations
def _get_dct_funcs():
    try:
        from scipy.fftpack import dct, idct
        def dct2(a):
            return dct(dct(a.T, norm='ortho').T, norm='ortho')
        def idct2(a):
            return idct(idct(a.T, norm='ortho').T, norm='ortho')
        return dct2, idct2
    except Exception:
        try:
            import cv2
            def dct2(a):
                return cv2.dct(a.astype(np.float32)).astype(np.float64)
            def idct2(a):
                return cv2.idct(a.astype(np.float32)).astype(np.float64)
            return dct2, idct2
        except Exception:
            # build DCT-II matrix (NxN) for arbitrary sizes
            def build_dct_matrix(N):
                C = np.zeros((N, N), dtype=np.float64)
                factor = np.pi / (2.0 * N)
                for k in range(N):
                    for n in range(N):
                        alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
                        C[k, n] = alpha * np.cos((2*n + 1) * k * factor)
                return C
            def dct2(a):
                N, M = a.shape
                Cx = build_dct_matrix(N)
                Cy = build_dct_matrix(M)
                return Cx @ a @ Cy.T
            def idct2(A):
                N, M = A.shape
                Cx = build_dct_matrix(N)
                Cy = build_dct_matrix(M)
                return Cx.T @ A @ Cy
            return dct2, idct2

dct2, idct2 = _get_dct_funcs()

# Input 8x8 block (diberikan pada soal)
block = np.array([
    [0, 0, 0, 0, 16, 0, 0, 0],
    [4, 0, 8, 16, 32, 16, 8, 0],
    [4, 0, 16, 32, 64, 32, 16, 0],
    [0, 0, 32, 64, 128, 64, 32, 0],
    [4, 0, 0, 32, 64, 32, 0, 0],
    [0, 16, 0, 0, 32, 0, 0, 0],
    [0, 0, 0, 0, 16, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.float64)

def pretty_print(mat, fmt="{:8.3f}"):
    for row in mat:
        print(" ".join(fmt.format(x) for x in row))

# 1) Standard 2D DCT (8x8)
std_dct = dct2(block)

# 2) Simple SA-DCT:
#    - bounding box of non-zero entries
#    - DCT on crop
#    - pack crop-coefs back into an 8x8 zero matrix at same location
mask = (block != 0)
if not mask.any():
    raise ValueError("Block is all zeros")

ys, xs = np.where(mask)
y0, y1 = ys.min(), ys.max()
x0, x1 = xs.min(), xs.max()

crop = block[y0:y1+1, x0:x1+1]

# compute DCT on crop: if dct2 supports arbitrary size, use it; else fallback
try:
    sa_crop_dct = dct2(crop)
except Exception:
    # generic DCT for arbitrary size
    def dct2_any(a):
        N, M = a.shape
        def build_dct(N):
            C = np.zeros((N, N), dtype=np.float64)
            factor = np.pi / (2.0 * N)
            for k in range(N):
                for n in range(N):
                    alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
                    C[k, n] = alpha * np.cos((2*n + 1) * k * factor)
            return C
        Cx = build_dct(N)
        Cy = build_dct(M)
        return Cx @ a @ Cy.T
    sa_crop_dct = dct2_any(crop)

sa_dct_packed = np.zeros_like(block)
sa_dct_packed[y0:y1+1, x0:x1+1] = sa_crop_dct

# Print outputs
print("Input 8x8 block:")
pretty_print(block, "{:4.0f}")
print("\nStandard 2D DCT (8x8):")
pretty_print(std_dct)
print("\nSA-DCT (crop DCT packed back into 8x8 at bounding-box):")
pretty_print(sa_dct_packed)
print(f"\nBounding box for crop: y={y0}..{y1}, x={x0}..{x1} (size {crop.shape})")

# Optional: inverse transforms to show reconstruction behavior
recon_std = idct2(std_dct)
# inverse of crop DCT (use same inverse if available)
try:
    inv_crop = idct2(sa_crop_dct)
except Exception:
    # simple pseudo-inverse fallback (not ideal)
    inv_crop = np.real(np.linalg.pinv(np.eye(sa_crop_dct.shape[0])) @ sa_crop_dct @ np.linalg.pinv(np.eye(sa_crop_dct.shape[1])))

recon_sa = np.zeros_like(block)
recon_sa[y0:y1+1, x0:x1+1] = inv_crop

print("\nReconstructed (IDCT) from standard DCT (rounded):")
pretty_print(np.round(recon_std), "{:4.0f}")
print("\nReconstructed (IDCT) from SA-DCT crop (placed back, rounded):")
pretty_print(np.round(recon_sa), "{:4.0f}")
