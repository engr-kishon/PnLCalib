import torch
import yaml
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

def export_to_onnx(model_path, config_path, is_line_model, output_name):
    print(f"Loading config from {config_path}...")
    cfg = yaml.safe_load(open(config_path, 'r'))
    
    # Initialize the correct model architecture
    if is_line_model:
        model = get_cls_net_l(cfg)
    else:
        model = get_cls_net(cfg)
        
    print(f"Loading weights from '{model_path}'...")
    # Load the extension-less weights
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # The inference script resizes images to 540x960, so we use that for the dummy input
    dummy_input = torch.randn(1, 3, 540, 960)

    print(f"Exporting to {output_name}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_name, 
        export_params=True,
        opset_version=12,          # Opset 12 is highly stable for TensorRT conversion
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output']
    )
    print("Done!\n")

if __name__ == "__main__":
    # Ensure you change "SV_kp" and "SV_lines" to the exact paths of your downloaded files
    
    # 1. Export the Keypoints Model
    export_to_onnx(
        model_path="/content/SV_kp", 
        config_path="/content/PnLCalib/config/hrnetv2_w48.yaml", 
        is_line_model=False, 
        output_name="keypoint_model.onnx"
    )

    # 2. Export the Lines Model
    export_to_onnx(
        model_path="/content/SV_lines", 
        config_path="/content/PnLCalib/config/hrnetv2_w48_l.yaml", 
        is_line_model=True, 
        output_name="line_model.onnx"
    )
