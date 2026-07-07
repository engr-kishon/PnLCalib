import cv2
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict

torch.backends.cudnn.benchmark = True # Crucial for static batch sizes

lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [16.5, 54.16, 0.]], [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                [[88.5, 54.16, 0.], [105., 54.16, 0.]], [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]], [[0., 30.34, 0.], [0., 30.34, -2.44]],
                [[105., 37.66, -2.44], [105., 30.34, -2.44]], [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],
                [[52.5, 0., 0.], [52.5, 68, 0.]], [[0., 68., 0.], [105., 68., 0.]], [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]], [[0., 0., 0.], [105., 0., 0.]], [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]], [[5.5, 24.84, 0.], [0., 24.84, 0.]], [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                [[99.5, 43.16, 0.], [99.5, 24.84, 0.]], [[99.5, 24.84, 0.], [105., 24.84, 0.]]]


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])
    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]], [0, y_focal_length, principal_point[1]], [0, 0, 1]])
    P = Q @ (rotation @ It)
    return P


def project(frame, P):
    # Projects the 3D lines onto the broadcast camera view
    for line in lines_coords:
        w1, w2 = line[0], line[1]
        i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
        i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
        i1 /= i1[-1]
        i2 /= i2[-1]
        frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), (255, 0, 0), 3)

    r = 9.15
    pts1, pts2, pts3 = [], [], []
    base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        ipos = P @ (base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.]))
        pts1.append([ipos[0]/ipos[-1], ipos[1]/ipos[-1]])

    base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        ipos = P @ (base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.]))
        pts2.append([ipos[0]/ipos[-1], ipos[1]/ipos[-1]])

    base_pos = np.array([0, 0, 0., 0.])
    for ang in np.linspace(0, 360, 500):
        ang = np.deg2rad(ang)
        ipos = P @ (base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.]))
        pts3.append([ipos[0]/ipos[-1], ipos[1]/ipos[-1]])

    frame = cv2.polylines(frame, [np.array(pts1, np.int32)], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [np.array(pts2, np.int32)], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [np.array(pts3, np.int32)], False, (255, 0, 0), 3)
    return frame


def draw_pitch_template(width=1050, height=680):
    # Generates a scaled 2D football pitch background
    pitch = np.full((height, width, 3), (0, 100, 0), dtype=np.uint8)
    color = (255, 255, 255) # White lines
    thickness = 2
    
    cv2.rectangle(pitch, (0, 0), (width, height), color, thickness)
    cv2.line(pitch, (width//2, 0), (width//2, height), color, thickness)
    cv2.circle(pitch, (width//2, height//2), 91, color, thickness)
    
    pen_w, pen_h = 165, 403
    cv2.rectangle(pitch, (0, height//2 - pen_h//2), (pen_w, height//2 + pen_h//2), color, thickness)
    cv2.rectangle(pitch, (width - pen_w, height//2 - pen_h//2), (width, height//2 + pen_h//2), color, thickness)
    
    goal_w, goal_h = 55, 183
    cv2.rectangle(pitch, (0, height//2 - goal_h//2), (goal_w, height//2 + goal_h//2), color, thickness)
    cv2.rectangle(pitch, (width - goal_w, height//2 - goal_h//2), (width, height//2 + goal_h//2), color, thickness)
    
    return pitch


def generate_top_down_view(frame, P, pitch_width=1050, pitch_height=680):
    H = np.zeros((3, 3))
    H[:, 0] = P[:, 0]
    H[:, 1] = P[:, 1]
    H[:, 2] = P[:, 3]

    # The scale factor (10 pixels = 1 meter)
    scale_factor = 10
    scale_matrix = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]
    ])

    # Center the pitch origin (0,0) to the middle of the image
    shift_matrix = np.array([
        [1, 0, pitch_width / 2],
        [0, 1, pitch_height / 2],
        [0, 0, 1]
    ])
    
    # Final Transformation: Invert -> Scale up -> Shift to center
    H_inv = np.linalg.inv(H)
    H_final = shift_matrix @ scale_matrix @ H_inv

    # Generate the pristine 2D pitch template
    tactical_map = draw_pitch_template(pitch_width, pitch_height)

    # Warp the broadcast frame and lay it transparently over the pitch map
    top_down_frame = cv2.warpPerspective(
        frame, 
        H_final, 
        (pitch_width, pitch_height), 
        dst=tactical_map,                 # Draw ON TOP of our new pitch template
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT # Keeps the rest of the pitch template visible
    )
    
    return top_down_frame


def process_batch(cam, frames_buffer, raw_frames_buffer, model_kp, model_line, kp_threshold, line_threshold, pnl_refine, device, out):
    # Process the batch simultaneously
    batch_tensor = torch.stack(frames_buffer).to(device).half()
    
    with torch.inference_mode():
        heatmaps = model_kp(batch_
