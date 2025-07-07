"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import torch
import torch.onnx
import onnxruntime as ort

from helpers import get_segmentation_model, get_transform
from torchvision.transforms import v2 as T


import cv2

def export_to_onnx(model, device, input_shape=(3, 400, 640), onnx_path="checkpoints/mask_rcnn_model.onnx"):
    """Export PyTorch model to ONNX format"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores', 'masks'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'batch_size'},
            'labels': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'masks': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_path}")

def inference_real_time_test_onnx(onnx_path, device, imgs):
    """Inference using ONNX model"""
    print("ONNX Inference")
    transform = get_transform(train=False)
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    confidence_threshold = 0.65
    
    for img_path in imgs:
        image_raw = cv2.imread(img_path)
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = T.ToTensor()(image)
        image = transform(image)  # Apply the same transformations as during training
        x = image.unsqueeze(0).to(torch.float).numpy()  # ONNX Runtime requires CPU numpy array
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outputs = ort_session.run(None, ort_inputs)

        # Process outputs (format depends on your model's output structure)
        # You may need to adjust this based on your model's specific outputs
        masks = torch.tensor(ort_outputs[3]) > confidence_threshold  # Assuming masks are the 4th output
        
        # Rest of visualization code remains the same
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = image[:, :, ::-1]
        overlay = image.copy()
        for pred_mask in masks.squeeze(1):
            pred_mask = pred_mask.numpy()
            overlay[pred_mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imshow("Overlay", blended)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    root = 'data/dataset'
    imgs = [ os.path.join(root, "images", img) for img in list(sorted(os.listdir(os.path.join(root, "images")))) ]
    indices = list(range(len(imgs)))
    test_imgs = [imgs[i] for i in indices[int(len(indices) * 0.9):]]
        
    # Export to ONNX (run once)
    onnx_path = "checkpoints/mask_rcnn_model.onnx"
    if not os.path.exists(onnx_path):
        model = get_segmentation_model(num_classes)
        model.load_state_dict(torch.load("checkpoints/mask_rcnn_model.pth"))
        model.to(device)
        export_to_onnx(model, device, onnx_path=onnx_path)
    
    # Use ONNX inference
    inference_real_time_test_onnx(onnx_path, device, test_imgs)
