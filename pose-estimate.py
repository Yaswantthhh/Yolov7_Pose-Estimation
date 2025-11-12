import cv2
import time
import torch
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer

@torch.no_grad()
def run_image(
        poseweights='yolov7-w6-pose.pt',
        source='image.jpeg',
        device='cpu'):

    # Select device
    device = select_device(device)

    # Load model
    model = attempt_load(poseweights, map_location=device)
    _ = model.eval()

    # Load image
    orig_image = cv2.imread(source)
    assert orig_image is not None, f"Image not found: {source}"

    # Convert to RGB and resize with stride 64
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, new_shape=(640, 640), stride=64, auto=True)[0]

    # To tensor
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device).float()

    start_time = time.time()

    # Predictions
    output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65,
                                     nc=model.yaml['nc'],
                                     nkpt=model.yaml['nkpt'],
                                     kpt_label=True)
    output = output_to_keypoint(output)

    # Prepare output image
    im0 = image[0].permute(1, 2, 0) * 255
    im0 = im0.cpu().numpy().astype(np.uint8)
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

    # Draw skeletons
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
        xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
        xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
        cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                      color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(im0, f'FPS: {int(fps)}', (11, 100), 0, 1, [255, 0, 0],
                thickness=2, lineType=cv2.LINE_AA)

    # Save result
    out_path = f"{source.split('.')[0]}_keypoints.jpg"
    cv2.imwrite(out_path, im0)
    print(f"Pose estimation complete. Saved to {out_path}")


if __name__ == "__main__":
    strip_optimizer('cpu', 'yolov7-w6-pose.pt')  # safe clean up
    run_image(poseweights='yolov7-w6-pose.pt', source='image.jpeg', device='cpu')
