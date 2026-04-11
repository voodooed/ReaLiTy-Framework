import numpy as np
import open3d as o3d


def calculate_incidence_angle(points_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
        fast_normal_computation=False
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.])
    )

    normals = np.asarray(pcd.normals)
    #distances = np.linalg.norm(points_xyz, axis=1)
    #directions = points_xyz / distances[:, np.newaxis]
    distances = np.linalg.norm(points_xyz, axis=1)
    safe_distances = np.where(distances == 0, 1e-6, distances)
    directions = points_xyz / safe_distances[:, np.newaxis]

    dot_product = np.sum(directions * normals, axis=1)
    normals[dot_product < 0] *= -1
    dot_product = np.sum(directions * normals, axis=1)

    incidence = np.arccos(np.clip(dot_product, -1.0, 1.0))
    incidence = np.nan_to_num(incidence, nan=0.0)

    return incidence


def calculate_material_reflectance(labels, reflectance_map):
    return np.array([reflectance_map.get(int(l), 0.5) for l in labels])


def normalize_channel(channel):
    if channel.size == 0 or np.all(channel == 0):
        return channel

    cmin, cmax = np.min(channel), np.max(channel)
    if cmax == cmin:
        return channel

    return (channel - cmin) / (cmax - cmin)


def project_pointcloud(pc, config):
    """
    pc: numpy array (N, 5) → x,y,z,intensity,label
    config: dict with sensor parameters
    """

    fov_up = np.deg2rad(config["fov_up"])
    fov_down = np.deg2rad(config["fov_down"])
    width = config["width"]
    height = config["height"]
    reflectance_map = config["reflectance_map"]

    pc = pc[~np.isnan(pc[:, :3]).any(axis=1)]

    if pc.shape[0] == 0:
        return None, None

    xyz = pc[:, :3]
    intensity = pc[:, 3]
    labels = pc[:, 4]

    r = np.linalg.norm(xyz, axis=1)

    # horizontal mapping
    u = 0.5 * (1 - np.arctan2(xyz[:, 1], xyz[:, 0]) / np.pi)

    # vertical mapping
    fov = fov_up - fov_down
    v = 1 - (np.arcsin(xyz[:, 2] / np.where(r == 0, 1e-6, r)) - fov_down) / fov

    if v.size == 0 or np.min(v) == np.max(v):
        return None, None

    v = (v - np.min(v)) / (np.max(v) - np.min(v))

    u = np.floor(u * 0.999 * width).astype(int)
    v = np.floor(v * 0.999 * height).astype(int)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    incidence = calculate_incidence_angle(xyz)
    reflectance = calculate_material_reflectance(labels, reflectance_map)

    range_image = np.zeros((4, height, width), dtype=np.float32)

    for i in range(pc.shape[0]):
        if range_image[0, v[i], u[i]] == 0 or r[i] < range_image[0, v[i], u[i]]:
            range_image[0, v[i], u[i]] = r[i]
            range_image[1, v[i], u[i]] = incidence[i]
            range_image[2, v[i], u[i]] = reflectance[i]
            range_image[3, v[i], u[i]] = intensity[i]

    range_image[0] = normalize_channel(range_image[0])

    mapping = {"u": u, "v": v}

    return range_image, mapping