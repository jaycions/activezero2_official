import cv2
import numpy as np
import open3d as o3d


def cal_normals(points: np.ndarray, camera_location=np.array([0.0, 0.0, 0.0]), radius=0.01, max_nn=30):
    cloud = o3d.geometry.PointCloud()
    cld = points.astype(np.float32)
    cloud.points = o3d.utility.Vector3dVector(cld)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    cloud.orient_normals_towards_camera_location(camera_location)
    n = np.array(cloud.normals).copy()
    return n


def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4)):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


def cal_normal_map(depth_map, cam_intrinsic, radius=0.01, max_nn=30):
    points = depth2pts_np(depth_map, cam_intrinsic)
    normals = cal_normals(points, radius=radius, max_nn=max_nn)
    normal_map = normals.reshape((depth_map.shape[0], depth_map.shape[1], 3))
    return normal_map
