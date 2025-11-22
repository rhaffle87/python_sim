import numpy as np

# Integer Transform Matrices (4x4)
H = np.array([
    [1,  1,  1,  1],
    [2,  1, -1, -2],
    [1, -1, -1,  1],
    [1, -2,  2, -1]
], dtype=np.int32)

H_inv = np.array([
    [1,  1,  1,  1],
    [1,  1, -1, -1],
    [1, -1, -1,  1],
    [2, -1,  1, -2]
], dtype=np.int32)

# Scaling and quantization matrices
Mf = np.ones((4,4), dtype=np.int32) * 16      # Eq (12.5)
Vi = np.ones((4,4), dtype=np.int32) * 16      # Eq (12.7)


def int_transform(block4):
    F = H @ block4 @ H.T
    F_hat = np.round((F * Mf) / (2**15)).astype(np.int32)
    return F_hat

def int_inverse(F_hat):
    temp = H_inv @ (F_hat * Vi) @ H_inv.T
    block_rec = np.round(temp / (2**6)).astype(np.int32)
    return block_rec

def sad(a, b):
    return np.sum(np.abs(a - b))

def log_search(ref, target, x, y, p=7):
    best_mv = (0,0)
    best_cost = 1e12
    step = p // 2

    while step >= 1:
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                xx = x + dx
                yy = y + dy
                if xx < 0 or yy < 0 or xx+8 > ref.shape[0] or yy+8 > ref.shape[1]:
                    continue
                cost = sad(target, ref[xx:xx+8, yy:yy+8])
                if cost < best_cost:
                    best_cost = cost
                    best_mv = (dx, dy)
        step //= 2

    return best_mv

def intra4_predict(block_top, block_left, mode):
    if mode == "vertical":
        return np.tile(block_top, (4,1))
    elif mode == "horizontal":
        return np.tile(block_left.reshape(4,1), (1,4))
    elif mode == "dc":
        dc = int((np.sum(block_top)+np.sum(block_left))/8)
        return np.ones((4,4), dtype=np.int32)*dc
    else:
        raise ValueError("Unknown Intra mode")

def encode_I_frame(frame):
    h, w = frame.shape
    coeffs = np.zeros_like(frame)

    for i in range(0, h, 4):
        for j in range(0, w, 4):

            block = frame[i:i+4, j:j+4]

            # choose best mode
            top = frame[i-1, j:j+4] if i > 0 else np.array([128]*4)
            left = frame[i:i+4, j-1] if j > 0 else np.array([128]*4)

            modes = ["vertical", "horizontal", "dc"]
            best_mode = None
            best_err = 1e12
            best_pred = None

            for m in modes:
                pred = intra4_predict(top, left, m)
                err = np.sum(np.abs(block - pred))
                if err < best_err:
                    best_err = err
                    best_mode = m
                    best_pred = pred

            residual = block - best_pred
            coeffs[i:i+4, j:j+4] = int_transform(residual)

    return coeffs

def encode_P_frame(ref, frame):
    h, w = frame.shape
    out = np.zeros_like(frame)

    for i in range(0, h, 8):
        for j in range(0, w, 8):

            cur_blk = frame[i:i+8, j:j+8]
            mv = log_search(ref, cur_blk, i, j, p=7)
            dx, dy = mv

            pred_blk = ref[i+dx:i+dx+8, j+dy:j+dy+8]

            residual = cur_blk - pred_blk

            # apply 4×4 transforms inside this 8×8
            for ii in range(0, 8, 4):
                for jj in range(0, 8, 4):
                    res4 = residual[ii:ii+4, jj:jj+4]
                    out[i+ii:i+ii+4, j+jj:j+jj+4] = int_transform(res4)

    return out

def decode_frame(ref, coeffs, is_I_frame):
    h, w = coeffs.shape
    rec = np.zeros_like(coeffs)

    for i in range(0, h, 4):
        for j in range(0, w, 4):
            rec[i:i+4, j:j+4] = int_inverse(coeffs[i:i+4, j:j+4])

    if not is_I_frame:
        # P-frame: add prediction
        rec += ref

    return rec

if __name__ == "__main__":
    frame1 = np.random.randint(0, 255, (16,16)).astype(np.int32)
    frame2 = frame1.copy()
    frame2[4:12,4:12] += 10   # simulate motion

    print("Encoding I-frame...")
    I_coeff = encode_I_frame(frame1)

    print("Encoding P-frame...")
    P_coeff = encode_P_frame(frame1, frame2)

    print("Decoding P-frame...")
    P_rec = decode_frame(frame1, P_coeff, is_I_frame=False)

    print("Done.")
