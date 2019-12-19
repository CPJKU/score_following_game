import cv2

import matplotlib.pyplot as plt
import numpy as np


def prepare_spec_for_render(spec, resz_spec=4, transform_to_bgr=True):

    if transform_to_bgr:
        spec_prep = spec[0][::-1, :]
        spec_prep = cv2.resize(spec_prep, (spec[0].shape[1] * resz_spec, spec[0].shape[0] * resz_spec))
        spec_prep = plt.cm.viridis(spec_prep)[:, :, 0:3]
        spec_prep = (spec_prep * 255).astype(np.uint8)
        spec_transformed = cv2.cvtColor(spec_prep, cv2.COLOR_RGB2BGR)
    else:
        spec_prep = spec[::-1, :]
        spec_transformed = cv2.resize(spec_prep, (spec.shape[1] * resz_spec, spec.shape[0] * resz_spec))

    return spec_transformed


def prepare_sheet_for_render(sheet,  resz_x=8, resz_y=4, transform_to_bgr=True):

    if transform_to_bgr:
        w, h = int(sheet[0].shape[1] * resz_x), int(sheet[0].shape[0] * resz_y)
        sheet_transformed = cv2.resize(sheet[0], (w, h))
        sheet_transformed = cv2.cvtColor(sheet_transformed, cv2.COLOR_GRAY2BGR)
        sheet_transformed = sheet_transformed.astype(np.uint8)
    else:
        w, h = int(sheet.shape[1] * resz_x), int(sheet.shape[0] * resz_y)
        sheet_transformed = cv2.resize(sheet, (w, h))

    return sheet_transformed


def write_text(text, position, img, font_face=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0)):

    text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=0.6, thickness=1)[0]
    text_org = ((img.shape[1] - text_size[0] - 5), (img.shape[1]//2 + position * text_size[1] + 3))
    cv2.putText(img, text, text_org, fontFace=font_face, fontScale=0.6, color=color, thickness=1)

