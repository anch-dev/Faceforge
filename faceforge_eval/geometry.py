"""Face alignment: 5-point similarity transform onto the ArcFace template."""
from __future__ import annotations

import cv2
import numpy as np

# ArcFace 112x112 reference landmarks (InsightFace, MIT-licensed code constant).
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform src -> dst (rotation, uniform scale,
    translation), returned as a 3x3 matrix. Same estimator as
    skimage.transform.SimilarityTransform.estimate."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    cov = dst_d.T @ src_d / n
    d = np.ones(dim)
    if np.linalg.det(cov) < 0:
        d[-1] = -1
    U, S, Vt = np.linalg.svd(cov)
    T = np.eye(dim + 1)
    rank = np.linalg.matrix_rank(cov)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[-1]
            d[-1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[-1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt
    scale = (S @ d) / src_d.var(0).sum()
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def estimate_norm(kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """2x3 affine mapping frame landmarks onto the template at `image_size`.

    Sizes divisible by 112 scale the template; otherwise it is scaled for 128
    and shifted 8 px (per 128) in x — InsightFace's convention, which
    inswapper_128 was trained with."""
    if image_size % 112 == 0:
        ratio, diff_x = image_size / 112.0, 0.0
    elif image_size % 128 == 0:
        ratio = image_size / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError(f"unsupported crop size {image_size}")
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    return umeyama(np.asarray(kps, dtype=np.float32), dst)[:2, :]


def norm_crop(frame: np.ndarray, kps: np.ndarray, image_size: int = 112):
    M = estimate_norm(kps, image_size)
    crop = cv2.warpAffine(frame, M, (image_size, image_size), borderValue=0.0)
    return crop, M
