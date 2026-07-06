import cv2
import argparse
import numpy as np
import torch
import onnxruntime as ort
from tqdm import tqdm

from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict

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


def process_batch(cam, frames_buffer, raw_frames_buffer, session_kp, session_line, kp_threshold, line_threshold, pnl_refine, device, out):
    # 1. Stack all arrays into a single batch: Shape (B, 3, 540, 960)
    batch_np = np.stack(frames_buffer)
    
    input_name_kp = session_kp.get_inputs()[0].name
    input_name_line = session_line.get_inputs()[0].name

    # 2. Run ONNX Session (Processes all frames instantly on CUDA)
    heatmaps_np = session_kp.run(None, {input_name_kp: batch_np})[0]
    heatmaps_l_np = session_line.run(None, {input_name_line: batch_np})[0]

    # 3. Fast tensor conversion for metric calculation
    heatmaps = torch.as_tensor(heatmaps_np, device=device)
    heatmaps_l = torch.as_tensor(heatmaps_l_np, device=device)

    # 4. Extract entire batch coordinates
    kp_coords_batch = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords_batch = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])

    # 5. Sequentially calibrate and map
    for i in range(len(raw_frames_buffer)):
        raw_frame = raw_frames_buffer[i]
        
        # Slicing [i:i+1] maintains the tensor format
        kp_dict = coords_to_dict(kp_coords_batch[i:i+1], threshold=kp_threshold)
        lines_dict = coords_to_dict(line_coords_batch[i:i+1], threshold=line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=960, h=540, normalize=True)

        cam.update(kp_dict, lines_dict)
        final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)

        if final_params_dict is not None:
            P = projection_from_cam_params(final_params_dict)
            projected_frame = project(raw_frame.copy(), P)
        else:
            projected_frame = raw_frame
            
        if out is not None:
            out.write(projected_frame)


def process_video(input_path, save_path, session_kp, session_line, kp_threshold, line_threshold, pnl_refine, device, batch_size):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)) if save_path else None

    frames_buffer = []
    raw_frames_buffer = []

    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        raw_frames_buffer.append(frame)

        # Pure OpenCV preprocessing
        if frame.shape[1] != 960 or frame.shape[0] != 540:
            frame_resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_np = frame_rgb.astype(np.float32) / 255.0
        input_np = np.transpose(input_np, (2, 0, 1))
        
        frames_buffer.append(input_np)

        # Process when batch is full
        if len(frames_buffer) == batch_size:
            process_batch(cam, frames_buffer, raw_frames_buffer, session_kp, session_line, kp_threshold, line_threshold, pnl_refine, device, out)
            pbar.update(batch_size)
            frames_buffer.clear()
            raw_frames_buffer.clear()

    # Flush remaining frames in the buffer
    if len(frames_buffer) > 0:
        process_batch(cam, frames_buffer, raw_frames_buffer, session_kp, session_line, kp_threshold, line_threshold, pnl_refine, device, out)
        pbar.update(len(frames_buffer))

    cap.release()
    if out:
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_kp", type=str, required=True, help="Path to dynamic batch Keypoint ONNX model")
    parser.add_argument("--onnx_line", type=str, required=True, help="Path to dynamic batch Line ONNX model")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of frames to process at once")
    parser.add_argument("--kp_threshold", type=float, default=0.3434)
    parser.add_argument("--line_threshold", type=float, default=0.7867)
    parser.add_argument("--pnl_refine", action="store_true")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()

    # Use standard CUDA provider. ONNX Runtime will handle the GPU offloading.
    providers = ['CUDAExecutionProvider']

    print("Loading Dynamic Batch ONNX Models to GPU...")
    session_kp = ort.InferenceSession(args.onnx_kp, providers=providers)
    session_line = ort.InferenceSession(args.onnx_line, providers=providers)
    print("Models Loaded. Starting Batch Processing!")

    # Device for the intermediate downstream tensor conversions
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    process_video(
        args.input_path, 
        args.save_path, 
        session_kp, 
        session_line, 
        args.kp_threshold, 
        args.line_threshold, 
        args.pnl_refine, 
        device, 
        args.batch_size
    )
