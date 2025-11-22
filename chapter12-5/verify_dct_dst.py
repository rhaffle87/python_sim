import numpy as np
from scipy.fftpack import dct, dst
from collections import Counter

# Parameters
BLOCK = 4          # use 4x4 intra blocks
N_SAMPLES = 2000   # number of synthetic blocks
Q = 8              # quantization step (tunable)

def two_d_transform(block, transform1d):
    # apply 1D transform on rows then columns
    tmp = transform1d(block, axis=1)
    res = transform1d(tmp.T, axis=1).T
    return res

def dct1(x, axis=1):
    return dct(x, type=2, norm='ortho', axis=axis)

def dst1(x, axis=1):
    return dst(x, type=2, norm='ortho', axis=axis)

def quantize(arr, q):
    return np.round(arr / q).astype(np.int32)

def entropy_bits(symbols):
    # empirical entropy in bits * number of symbols
    if len(symbols) == 0:
        return 0.0
    cnt = Counter(symbols)
    total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()])
    h = -(probs * np.log2(probs)).sum()
    return h * total

def make_category2_block(top, left):
    """
    Construct a 4x4 block where prediction uses both top and left.
    We synthesize interior as simple bilinear blend + small noise to simulate
    directional structure where DST often concentrates energy.
    """
    # top: length BLOCK, left: length BLOCK
    b = np.zeros((BLOCK, BLOCK), dtype=np.float32)
    for i in range(BLOCK):
        for j in range(BLOCK):
            # bilinear-like interpolation biased toward top+left structure
            w_t = (BLOCK - i) / BLOCK
            w_l = (BLOCK - j) / BLOCK
            val = (w_t * top[j] + w_l * left[i]) / (w_t + w_l + 1e-9)
            b[i,j] = val
    # add small structured variation
    b += np.outer(np.linspace(0,1,BLOCK), np.linspace(0,1,BLOCK))*5.0
    # add tiny gaussian noise
    b += np.random.normal(0, 1.0, b.shape)
    return b

def experiment(n_samples=N_SAMPLES, q=Q):
    symbols_dct = []
    symbols_dst = []
    for _ in range(n_samples):
        # random neighbor boundaries typical image values (0-255)
        top = np.random.randint(10, 246, size=BLOCK).astype(np.float32)
        left = np.random.randint(10, 246, size=BLOCK).astype(np.float32)

        block = make_category2_block(top, left)

        # We consider residual = block - predictor (here predictor we set as DC of neighbors)
        # But the goal is to test transforms; use residual that keeps directional energy:
        predictor = (np.concatenate([top, left]).mean())  # simple DC predictor baseline
        residual = block - predictor

        # DCTxDCT
        coeffs_dct = two_d_transform(residual, dct1)
        qd_dct = quantize(coeffs_dct, q)
        symbols_dct.extend(qd_dct.flatten().tolist())

        # DSTxDST
        coeffs_dst = two_d_transform(residual, dst1)
        qd_dst = quantize(coeffs_dst, q)
        symbols_dst.extend(qd_dst.flatten().tolist())

    # compute estimated bits
    bits_dct = entropy_bits(symbols_dct)
    bits_dst = entropy_bits(symbols_dst)

    # also report nonzero counts
    nz_dct = sum(1 for s in symbols_dct if s != 0)
    nz_dst = sum(1 for s in symbols_dst if s != 0)
    total_coeffs = n_samples * BLOCK * BLOCK

    return {
        'bits_dct': bits_dct,
        'bits_dst': bits_dst,
        'nz_dct': nz_dct,
        'nz_dst': nz_dst,
        'total_coeffs': total_coeffs
    }

if __name__ == "__main__":
    np.random.seed(1)
    res = experiment(n_samples=2000, q=8)
    print("Results (estimate):")
    print(f"Total coeffs per method: {res['total_coeffs']}")
    print(f"DCTxDCT estimated bits: {res['bits_dct']:.1f}, nonzero coeffs: {res['nz_dct']}")
    print(f"DSTxDST estimated bits: {res['bits_dst']:.1f}, nonzero coeffs: {res['nz_dst']}")
    if res['bits_dst'] < res['bits_dct']:
        print("=> DSTxDST produces shorter estimated code (per entropy) for these Category-2 samples.")
    else:
        print("=> DCTxDCT equal/shorter for these samples.")
