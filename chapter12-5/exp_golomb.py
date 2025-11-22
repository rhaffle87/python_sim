# Exp-Golomb Encoder & Decoder (Order-k)
# --------------------------------------

def eg0_encode(N: int) -> str:
    """Encode unsigned integer N to Exp-Golomb order-0 (EG0)."""
    if N < 0:
        raise ValueError("N must be non-negative")
    N1 = N + 1
    m = N1.bit_length() - 1
    prefix = '0' * m + '1'
    info_bits = format(N1, 'b')[1:]
    return prefix + info_bits

def eg0_decode(bitstr: str):
    """Decode EG0 from the beginning of bitstr. Returns (value, bits_used)."""
    m = 0
    i = 0
    n = len(bitstr)
    while i < n and bitstr[i] == '0':
        m += 1
        i += 1
    if i >= n or bitstr[i] != '1':
        raise ValueError("Malformed EG0")
    i += 1
    if i + m > n:
        raise ValueError("Malformed EG0")
    info = bitstr[i:i+m] if m > 0 else '0'
    info_val = int(info, 2)
    N1 = (1 << m) + info_val
    N = N1 - 1
    consumed = 2*m + 1
    return N, consumed

def egk_encode(N: int, k: int) -> str:
    """Encode unsigned integer N to Exp-Golomb order-k (EGk)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    q = N >> k
    r = N & ((1 << k) - 1)
    prefix = eg0_encode(q)
    r_bits = format(r, 'b').zfill(k) if k > 0 else ''
    return prefix + r_bits

def egk_decode(bitstr: str, k: int):
    """Decode EGk from the start of bitstr. Returns (value, bits_used)."""
    q, used = eg0_decode(bitstr)
    offset = used
    if k > 0:
        r_bits = bitstr[offset:offset+k]
        if len(r_bits) < k:
            raise ValueError("Malformed EGk")
        r = int(r_bits, 2)
    else:
        r = 0
    N = (q << k) | r
    return N, used + k


# ---------------------------------------------------
# Example tests (same as problem requirement)
# ---------------------------------------------------

if __name__ == "__main__":
    print("EG0 encode N=110 =", eg0_encode(110))
    decoded, used = eg0_decode("000000011010011")
    print("EG0 decode '000000011010011' =", decoded, "(bits used =", used, ")")
    print("EG3 encode N=110 =", egk_encode(110, 3))
