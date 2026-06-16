import cv2
import numpy as np
from typing import Tuple
from .masking import SkinMaskGenerator

def run_uv_spots_analysis(image: np.ndarray, landmarks: list) -> Tuple[float, np.ndarray]:
    orig = image.copy()
    mask_gen = SkinMaskGenerator()
    roi_mask = mask_gen.get_full_face_mask(orig.shape, landmarks)

    b_channel = orig[:, :, 0]
    clahe_base = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_blue = clahe_base.apply(b_channel)

    kernel_size = 51
    background = cv2.GaussianBlur(enhanced_blue, (kernel_size, kernel_size), 0)
    high_pass = np.clip(enhanced_blue.astype(np.int16) - background.astype(np.int16) + 127, 0, 255).astype(np.uint8)

    clahe_hp = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    bp_img = clahe_hp.apply(high_pass)

    _, thresh = cv2.threshold(bp_img, 95, 255, cv2.THRESH_BINARY_INV)
    masked_spots = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned_spots = cv2.morphologyEx(masked_spots, cv2.MORPH_OPEN, kernel, iterations=1)

    total_roi_area = cv2.countNonZero(roi_mask)
    damage_area = cv2.countNonZero(cleaned_spots)
    damage_percentage = (damage_area / total_roi_area) * 100 if total_roi_area > 0 else 0
    score = min(10.0, max(1.0, (damage_percentage / 35.0) * 10.0))

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    clahe_style = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    sepia_base = cv2.cvtColor(clahe_style.apply(gray), cv2.COLOR_GRAY2BGR)

    sepia_base[:, :, 0] = np.clip(sepia_base[:, :, 0] * 0.6, 0, 255)
    sepia_base[:, :, 1] = np.clip(sepia_base[:, :, 1] * 0.9, 0, 255)
    sepia_base[:, :, 2] = np.clip(sepia_base[:, :, 2] * 1.3, 0, 255)

    contours, _ = cv2.findContours(cleaned_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sepia_base, contours, -1, (255, 255, 0), 1)
    mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sepia_base, mask_contours, -1, (255, 255, 0), 2)

    return float(score), sepia_base


def run_redness_analysis(image: np.ndarray, landmarks: list) -> Tuple[float, np.ndarray]:
    orig = image.copy()
    lab_img = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)
    l, a_channel, b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_a = clahe.apply(a_channel)

    mask_gen = SkinMaskGenerator()
    roi_mask = mask_gen.get_full_face_mask(orig.shape, landmarks)

    _, thresh = cv2.threshold(enhanced_a, 155, 255, cv2.THRESH_BINARY)
    masked_redness = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_redness = cv2.morphologyEx(masked_redness, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_redness = cv2.morphologyEx(closed_redness, cv2.MORPH_OPEN, kernel, iterations=1)

    total_roi_area = cv2.countNonZero(roi_mask)
    inflamed_area = cv2.countNonZero(cleaned_redness)
    damage_percentage = (inflamed_area / total_roi_area) * 100 if total_roi_area > 0 else 0
    score = min(10.0, max(1.0, (damage_percentage / 20.0) * 10.0))

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    red_base = cv2.cvtColor(gray_clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    red_base[:, :, 0] = np.clip(red_base[:, :, 0] * 0.8, 0, 255)
    red_base[:, :, 1] = np.clip(red_base[:, :, 1] * 0.8, 0, 255)
    red_base[:, :, 2] = np.clip(red_base[:, :, 2] * 1.3, 0, 255)

    contours, _ = cv2.findContours(cleaned_redness, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(red_base, contours, -1, (255, 255, 0), thickness=cv2.FILLED)

    blended = cv2.addWeighted(red_base, 0.5, orig, 0.5, 0)
    mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, mask_contours, -1, (255, 255, 0), 2)

    return float(score), blended


def run_brown_spots_analysis(image: np.ndarray, landmarks: list) -> Tuple[float, np.ndarray]:
    orig = image.copy()
    mask_gen = SkinMaskGenerator()
    roi_mask = mask_gen.get_full_face_mask(orig.shape, landmarks)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_spots = clahe.apply(blackhat)

    _, thresh = cv2.threshold(enhanced_spots, 130, 255, cv2.THRESH_BINARY)
    masked_spots = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned_spots = cv2.morphologyEx(masked_spots, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    total_roi_area = cv2.countNonZero(roi_mask)
    spot_area = cv2.countNonZero(cleaned_spots)
    damage_percentage = (spot_area / total_roi_area) * 100 if total_roi_area > 0 else 0
    score = min(10.0, max(1.0, (damage_percentage / 15.0) * 10.0))

    contrasted_gray = clahe.apply(gray)
    orange_base = cv2.cvtColor(contrasted_gray, cv2.COLOR_GRAY2BGR)
    orange_base[:, :, 0] = np.clip(orange_base[:, :, 0] * 0.4, 0, 255)
    orange_base[:, :, 1] = np.clip(orange_base[:, :, 1] * 0.8, 0, 255)
    orange_base[:, :, 2] = np.clip(orange_base[:, :, 2] * 1.3, 0, 255)

    contours, _ = cv2.findContours(cleaned_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(orange_base, contours, -1, (0, 255, 255), 1)
    mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(orange_base, mask_contours, -1, (255, 255, 0), 2)

    return float(score), orange_base


def run_porphyrins_analysis(image: np.ndarray, landmarks: list) -> Tuple[float, np.ndarray]:
    orig = image.copy()
    mask_gen = SkinMaskGenerator()
    roi_mask = mask_gen.get_full_face_mask(orig.shape, landmarks)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_pores = clahe.apply(blackhat)

    _, thresh = cv2.threshold(enhanced_pores, 80, 255, cv2.THRESH_BINARY)
    masked_pores = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    total_roi_area = cv2.countNonZero(roi_mask)
    damage_area = cv2.countNonZero(masked_pores)
    damage_percentage = (damage_area / total_roi_area) * 100 if total_roi_area > 0 else 0
    score = min(10.0, max(1.0, (damage_percentage / 6.0) * 10.0))

    darkened_gray = cv2.convertScaleAbs(gray, alpha=0.4, beta=0)
    uv_base = cv2.cvtColor(darkened_gray, cv2.COLOR_GRAY2BGR)
    uv_base[:, :, 0] = np.clip(uv_base[:, :, 0] * 1.8, 0, 255)
    uv_base[:, :, 1] = np.clip(uv_base[:, :, 1] * 0.4, 0, 255)
    uv_base[:, :, 2] = np.clip(uv_base[:, :, 2] * 0.6, 0, 255)

    contours, _ = cv2.findContours(masked_pores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(uv_base, (cx, cy), 1, (0, 255, 255), -1)

    mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(uv_base, mask_contours, -1, (255, 255, 0), 2)

    return float(score), uv_base


def run_spots_blemishes_analysis(image: np.ndarray, landmarks: list) -> Tuple[float, np.ndarray]:
    orig = image.copy()
    mask_gen = SkinMaskGenerator()
    roi_mask = mask_gen.get_full_face_mask(orig.shape, landmarks)

    b, g, r = cv2.split(orig)
    blurred_g = cv2.GaussianBlur(g, (5, 5), 0)
    background = cv2.medianBlur(blurred_g, 35)
    high_pass = cv2.absdiff(blurred_g, background)

    _, thresh = cv2.threshold(high_pass, 15, 255, cv2.THRESH_BINARY)
    masked_spots = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_spots = cv2.morphologyEx(masked_spots, cv2.MORPH_OPEN, kernel, iterations=1)

    total_roi_area = cv2.countNonZero(roi_mask)
    damage_area = cv2.countNonZero(cleaned_spots)
    damage_percentage = (damage_area / total_roi_area) * 100 if total_roi_area > 0 else 0
    score = min(10.0, max(1.0, (damage_percentage / 10.0) * 10.0))

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    spot_base = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    spot_base[:, :, 0] = np.clip(spot_base[:, :, 0] * 1.2, 0, 255)
    spot_base[:, :, 1] = np.clip(spot_base[:, :, 1] * 1.2, 0, 255)
    spot_base[:, :, 2] = np.clip(spot_base[:, :, 2] * 0.8, 0, 255)

    contours, _ = cv2.findContours(cleaned_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(spot_base, contours, -1, (255, 0, 255), 2)
    mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(spot_base, mask_contours, -1, (200, 200, 200), 2)

    return float(score), spot_base


def run_texture_analysis(image: np.ndarray, landmarks: list) -> Tuple[float, np.ndarray]:
    orig = image.copy()
    mask_gen = SkinMaskGenerator()
    roi_mask = mask_gen.get_full_face_mask(orig.shape, landmarks)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    abs_laplacian = np.absolute(laplacian)
    texture_map = np.uint8(255 * (abs_laplacian / np.max(abs_laplacian)))

    _, rough_mask = cv2.threshold(texture_map, 20, 255, cv2.THRESH_BINARY)
    masked_texture = cv2.bitwise_and(rough_mask, rough_mask, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned_texture = cv2.morphologyEx(masked_texture, cv2.MORPH_OPEN, kernel, iterations=1)

    total_roi_area = cv2.countNonZero(roi_mask)
    damage_area = cv2.countNonZero(cleaned_texture)
    damage_percentage = (damage_area / total_roi_area) * 100 if total_roi_area > 0 else 0
    score = min(10.0, max(1.0, (damage_percentage / 25.0) * 10.0))

    output_viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(cleaned_texture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_viz, contours, -1, (0, 165, 255), 1)

    mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_viz, mask_contours, -1, (220, 220, 100), 2)

    return float(score), output_viz


def run_pore_size_analysis(image: np.ndarray, landmarks: list) -> Tuple[float, np.ndarray]:
    orig = image.copy()
    mask_gen = SkinMaskGenerator()
    roi_mask = mask_gen.get_full_face_mask(orig.shape, landmarks)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blackhat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)
    masked_pores = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    contours, _ = cv2.findContours(masked_pores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pore_contours = []
    total_pore_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15 or area > 100: continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0: continue
        aspect_ratio = float(bw) / bh

        if 0.5 < aspect_ratio < 2.0:
            pore_contours.append(cnt)
            total_pore_area += area

    total_roi_area = cv2.countNonZero(roi_mask)
    damage_percentage = (total_pore_area / total_roi_area) * 100 if total_roi_area > 0 else 0
    score = min(10.0, max(1.0, (damage_percentage / 5.0) * 10.0))

    output_viz = orig.copy()
    for cnt in pore_contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(output_viz, (int(x), int(y)), int(radius), (255, 255, 0), 1)

    mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_viz, mask_contours, -1, (220, 220, 100), 2)

    return float(score), output_viz


MODULE_ROUTING_MAP = {
    "uv_spots": run_uv_spots_analysis,
    "redness": run_redness_analysis,
    "brown_spots": run_brown_spots_analysis,
    "porphyrins": run_porphyrins_analysis,
    "spots_and_blemishes": run_spots_blemishes_analysis,
    "texture": run_texture_analysis,
    "pore_size": run_pore_size_analysis
}