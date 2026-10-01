import numpy as np
import cv2


def build_intrinsics_matrix(fx, fy, cx, cy):
    """Return a 3x3 camera matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_remap(K_src, size_src, K_dst, size_dst):
    """
    Build remap grids (map_x, map_y) that transform an image from K_src, size_src
    to K_dst, size_dst assuming rectified pinhole model (no distortion/rotation).
    We map destination pixels -> source coordinates.
    """
    w_dst, h_dst = size_dst
    w_src, h_src = size_src

    fx_s, fy_s, cx_s, cy_s = K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2]
    fx_d, fy_d, cx_d, cy_d = K_dst[0, 0], K_dst[1, 1], K_dst[0, 2], K_dst[1, 2]

    # Create a meshgrid of destination pixel coordinates
    u_d = np.arange(w_dst, dtype=np.float32)
    v_d = np.arange(h_dst, dtype=np.float32)
    u_d, v_d = np.meshgrid(u_d, v_d)  # shape (h_dst, w_dst)

    # Back-project to normalized coords in destination camera
    x = (u_d - cx_d) / fx_d
    y = (v_d - cy_d) / fy_d

    # Reproject into source pixel coordinates (assumes identical orientation)
    u_s = x * fx_s + cx_s
    v_s = y * fy_s + cy_s

    # OpenCV remap expects float32 maps
    map_x = u_s.astype(np.float32)
    map_y = v_s.astype(np.float32)

    return map_x, map_y


def convert_to_intrinsics_bgr(src_bgr, map_x, map_y):
    """
    Apply a precomputed remap to convert src_bgr into the target intrinsics image.
    """
    dst = cv2.remap(
        src_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return dst
