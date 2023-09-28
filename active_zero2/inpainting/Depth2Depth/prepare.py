import numpy as np
import cv2
from path import Path
import h5py


def main():
    scene = "0-300103-2"
    scene = "0-300135-1"
    conf_threshold = 0.7

    output_dir = f"/media/DATA/LINUX_DATA/activezero2/outputs/psmnetedgeB_primv2rdpatrdlight_temporalirsimnoise/model_060000/{scene}_real"
    output_dir = Path(output_dir)

    gt_dir = f"/media/DATA/LINUX_DATA/ICCV2021_Diagnosis/real_data_v10/{scene}"
    gt_dir = Path(gt_dir)

    depth_pred = np.load(output_dir / "depth_pred.npy")
    print("depth pred min: ", depth_pred.min(), " max: ", depth_pred.max())
    depth_gt = cv2.imread(gt_dir / "depthL.png", cv2.IMREAD_UNCHANGED)
    depth_gt = (depth_gt.astype(float)) / 1000.0
    depth_gt = cv2.resize(depth_gt, (960, 540), interpolation=cv2.INTER_NEAREST)
    print("depth gt min: ", depth_gt.min(), " max: ", depth_gt.max())

    confidence_map = np.load(output_dir / "confidence.npy")
    confidence_mask = confidence_map[1] > conf_threshold
    depth_pred = depth_pred * confidence_mask
    depth_pred = cv2.resize(depth_pred, (480, 270), interpolation=cv2.INTER_NEAREST)
    depth_mask = (depth_pred > 0.2) * (depth_pred < 1.25)
    depth_pred = depth_pred * depth_mask

    gt_normal = cv2.imread(gt_dir / "normalL.png", cv2.IMREAD_UNCHANGED)
    gt_normal = cv2.resize(gt_normal, (480, 270), cv2.INTER_NEAREST)
    gt_normal = (gt_normal.astype(float)) / 1000.0 - 1

    normal_pred = cv2.imread(output_dir / "normal_pred_u16.png", cv2.IMREAD_UNCHANGED)
    normal_pred = cv2.resize(normal_pred, (480, 270), cv2.INTER_NEAREST)
    normal_pred = (normal_pred.astype(float)) / 1000.0 - 1

    normal_diff = np.arccos(np.clip(np.sum(gt_normal * normal_pred, axis=-1), -1, 1))
    normal_mask = normal_diff < 15 / 180 * np.pi
    cv2.imwrite(output_dir / "normal_mask.png", (normal_mask.astype(np.uint8)) * 255)

    gt_normal = np.transpose(gt_normal, (2, 0, 1))
    gt_normal2 = np.stack([gt_normal[0], gt_normal[2], gt_normal[1]])
    with h5py.File(gt_dir / "gt_normal.h5", "w") as f:
        f.create_dataset("result", data=gt_normal2)

    depth_pred = depth_pred * normal_mask
    cv2.imwrite(output_dir / "depth_pred_u16.png", (depth_pred * 4000).astype(np.uint16))
    edge = cv2.imread(output_dir / "edge_gt.png", 0)
    kernel = np.ones((3, 3), dtype=np.uint8) * 255
    edge = cv2.dilate(edge, kernel, iterations=3)
    edge = edge.astype(np.uint16)
    edge = cv2.resize(edge, (480, 270), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_dir / "edge_gt_u16.png", (1021 - edge * 4))


if __name__ == "__main__":
    main()
