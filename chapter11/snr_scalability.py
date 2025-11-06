from typing import Tuple, Optional
import numpy as np

# Try imports for DCT/IDCT
try:
    from scipy.fftpack import dct, idct

    def dct2(block: np.ndarray) -> np.ndarray:
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(block: np.ndarray) -> np.ndarray:
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

except Exception:
    # fallback to OpenCV if available
    try:
        import cv2

        def dct2(block: np.ndarray) -> np.ndarray:
            # OpenCV expects float32
            return cv2.dct(block.astype(np.float32)).astype(np.float64)

        def idct2(block: np.ndarray) -> np.ndarray:
            return cv2.idct(block.astype(np.float32)).astype(np.float64)

    except Exception:
        raise ImportError("scipy or opencv required for DCT/IDCT. Install scipy or opencv-python.")


# Frame types
IFRAME = "I"
PFRAME = "P"
BFRAME = "B"


class Macroblock:
    """Representasi macroblock yang menyimpan blok pixel dan metadata."""
    def __init__(self, data: np.ndarray, frame_type: str = IFRAME, mv: Tuple[int, int] = (0, 0), prediction_type: Optional[str] = None):
        """
        data: 2D numpy array (block), e.g. 16x16 atau 8x8
        frame_type: 'I', 'P', atau 'B'
        mv: motion vector (dx, dy) integer pixels (used if P/B)
        prediction_type: e.g. 'forward', 'backward', 'bi' (not used extensively here)
        """
        self.data = data.astype(np.float64)  # use float for DCT
        self.frame_type = frame_type
        self.mv = mv
        self.prediction_type = prediction_type


class ReferenceFrame:
    """Simple reference frame storing image pixels (grayscale)."""
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float64)

    def get_block(self, x: int, y: int, w: int, h: int, mv: Tuple[int, int]) -> np.ndarray:
        """Return predicted block at (x,y) with motion vector mv (dx,dy).
        Edges are handled by clamping."""
        dx, dy = mv
        sx = int(x + dx)
        sy = int(y + dy)
        # clamp
        sx = max(0, min(self.image.shape[1] - w, sx))
        sy = max(0, min(self.image.shape[0] - h, sy))
        return self.image[sy:sy + h, sx:sx + w].copy()


def motion_compensate(mb: Macroblock, x: int, y: int, ref1: Optional[ReferenceFrame], ref2: Optional[ReferenceFrame]) -> np.ndarray:
    """
    Simple motion compensation:
      - if forward ref exists, use ref1.get_block
      - otherwise fallback to zeros
    For bi-prediction you might average both refs if both given (not fully implemented).
    """
    h, w = mb.data.shape
    if mb.prediction_type == 'bi' and ref1 is not None and ref2 is not None:
        b1 = ref1.get_block(x, y, w, h, mb.mv)
        b2 = ref2.get_block(x, y, w, h, mb.mv)
        return 0.5 * (b1 + b2)
    elif ref1 is not None:
        return ref1.get_block(x, y, w, h, mb.mv)
    elif ref2 is not None:
        return ref2.get_block(x, y, w, h, mb.mv)
    else:
        return np.zeros_like(mb.data)


def quantize_block(coef: np.ndarray, qstep: float, scale: float = 8.0) -> np.ndarray:
    """Quantize using rounding similar to sample: Q = round(coef * scale / qstep)."""
    return np.round(coef * scale / qstep).astype(np.int32)


def dequantize_block(qcoef: np.ndarray, qstep: float, scale: float = 8.0) -> np.ndarray:
    """Dequantize inverse of the above: coef_approx = qcoef / scale * qstep."""
    return (qcoef.astype(np.float64) / scale) * qstep


def SNRScalability(mb: Macroblock, Q1step: float, Q2step: float,
                   x: int = 0, y: int = 0,
                   RefImg1: Optional[ReferenceFrame] = None,
                   RefImg2: Optional[ReferenceFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process one macroblock with SNR scalability.
    Returns:
       - bits_base: array of integers (quantized coefficients by Q1)
       - bits_enhance: array of integers (quantized residual coefficients by Q2)
       - reconstructed_block: spatial-domain block after IDCT (to update reference)
    Note: "bits" here are simply lists of quantized coefficient integers.
    """
    # 1) If frame is P or B, compute prediction and subtract
    work_block = mb.data.copy()
    if mb.frame_type in (PFRAME, BFRAME):
        pred = motion_compensate(mb, x, y, RefImg1, RefImg2)
        work_block = work_block - pred

    # 2) DCT
    DCTMB = dct2(work_block)

    # 3) Quantize base layer with Q1
    Q1DCT = quantize_block(DCTMB, Q1step)

    # 4) Dequantize base to produce base reconstruction
    recon_base = dequantize_block(Q1DCT, Q1step)

    # 5) Compute residual in DCT domain
    residual = DCTMB - recon_base

    # 6) Quantize residual with Q2 for enhancement stream
    Q2DCT = quantize_block(residual, Q2step)

    # Output "bitstreams": we just return the quantized arrays flattened (could be packed)
    bits_base = Q1DCT.flatten().astype(np.int32)
    bits_enhance = Q2DCT.flatten().astype(np.int32)

    # 7) Reconstruct final DCT coefficients = recon_base + dequantized enhancement
    recon_enh = dequantize_block(Q2DCT, Q2step)
    final_recon_DCT = recon_base + recon_enh

    # 8) IDCT to get spatial block
    IQDCTMB = idct2(final_recon_DCT)
    # If prediction was used, add prediction back to get full pixel values
    if mb.frame_type in (PFRAME, BFRAME):
        pred = motion_compensate(mb, x, y, RefImg1, RefImg2)
        recon_spatial = IQDCTMB + pred
    else:
        recon_spatial = IQDCTMB

    return bits_base, bits_enhance, recon_spatial


# ---- Example usage ----
if __name__ == "__main__":
    # Example: 8x8 macroblock
    block = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ], dtype=np.float64)

    # Create macroblock (I-frame example)
    mb = Macroblock(block, frame_type=IFRAME)

    # Quantization steps (example)
    Q1 = 10.0
    Q2 = 2.0

    bits_base, bits_enhance, recon = SNRScalability(mb, Q1, Q2)

    print("Bits base (quantized coefficients):", bits_base)
    print("Bits enhance (quantized residuals):", bits_enhance)
    print("Reconstructed block (rounded):\n", np.round(recon).astype(int))