import numpy as np
import open3d as o3d
import cv2
import pickle

from active_zero2.utils.geometry import depth2pts_np


def main():
    scene = "0-300103-2"
    scene = "0-300135-1"
    depth_in = cv2.imread(
        f"/media/DATA/LINUX_DATA/activezero2/outputs/psmnetedgeB_primv2rdpatrdlight_temporalirsimnoise/model_060000/{scene}_real/depth_pred_u16.png",
        cv2.IMREAD_UNCHANGED,
    )
    depth_in = (depth_in.astype(float)) / 4000.0
    depth_out = cv2.imread(
        f"/media/DATA/LINUX_DATA/activezero2/outputs/psmnetedgeB_primv2rdpatrdlight_temporalirsimnoise/model_060000/{scene}_real/depth_pred_u16_out.png",
        cv2.IMREAD_UNCHANGED,
    )
    depth_out = (depth_out.astype(float)) / 4000.0

    meta = pickle.load(open(f"/media/DATA/LINUX_DATA/ICCV2021_Diagnosis/real_data_v10/{scene}/meta.pkl", "rb"))
    intrinsic = meta["intrinsic_l"]
    intrinsic[:2] /= 4

    pts_in = depth2pts_np(depth_in, intrinsic)
    pts_out = depth2pts_np(depth_out, intrinsic)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_in)
    pcd = pcd.crop(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8]))
    )
    o3d.io.write_point_cloud("in.pcd", pcd)
    pcd.points = o3d.utility.Vector3dVector(pts_out)
    pcd = pcd.crop(
        o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([-10, -10, 0.1]), max_bound=np.array([10, 10, 1.8]))
    )
    o3d.io.write_point_cloud("out.pcd", pcd)


if __name__ == "__main__":
    main()
