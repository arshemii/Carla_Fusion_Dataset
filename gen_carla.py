import argparse
import json
import math
import os
import queue as pyqueue
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import carla

from sim_carla import (
    CFG,
    CameraSpec,
    LidarSpec,
    Pose6,
    derive_model_camera_geometry,
    config_to_dict,
)

"""
CARLA LiDAR local frame:       x forward, y right, z up
nuScenes frame: 	       x forward, y left,  z up
"""

@dataclass
class RuntimeCamera:
    spec: CameraSpec
    model_height: int
    model_width: int
    model_K: np.ndarray
    native_K: np.ndarray
    effective_model_fov_x_deg: float
    resize_meta: Dict
    actor: Optional[carla.Actor] = None
    queue: Optional[pyqueue.Queue] = None


class SensorHandle:
    def __init__(self, name: str, actor: carla.Actor, data_queue: pyqueue.Queue):
        self.name = name
        self.actor = actor
        self.queue = data_queue


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def expand_user_path(path_str: str) -> Path:
    return Path(os.path.expanduser(path_str)).resolve()


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def sparsefusion_from_carla_lidar_4x4() -> np.ndarray:

    F = np.eye(4, dtype=np.float64)
    F[1, 1] = -1.0
    return F


def map_layer_mask_from_names(layer_names: List[str]) -> carla.MapLayer:
    if not layer_names:
        return carla.MapLayer.NONE
    mask = carla.MapLayer.NONE
    for name in layer_names:
        if not hasattr(carla.MapLayer, name):
            raise ValueError(f"Unknown map layer name: {name}")
        mask |= getattr(carla.MapLayer, name)
    return mask


def choose_world(client: carla.Client) -> carla.World:
    if CFG.use_current_world:
        return client.get_world()
    if CFG.map_name.endswith("_Opt"):
        return client.load_world(CFG.map_name, map_layer_mask_from_names(CFG.map_layers))
    return client.load_world(CFG.map_name)


def pose_to_transform(pose: Pose6) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=pose.x, y=pose.y, z=pose.z),
        carla.Rotation(roll=pose.wx, pitch=pose.wy, yaw=pose.wz),
    )


def matrix_from_transform(transform: carla.Transform) -> np.ndarray:
    return np.array(transform.get_matrix(), dtype=np.float64)


def inverse_matrix_from_transform(transform: carla.Transform) -> np.ndarray:
    return np.array(transform.get_inverse_matrix(), dtype=np.float64)


def make_runtime_camera(spec: CameraSpec) -> RuntimeCamera:
    geom = derive_model_camera_geometry(spec, CFG)
    return RuntimeCamera(
        spec=spec,
        model_height=int(geom["model_size_hw"][0]),
        model_width=int(geom["model_size_hw"][1]),
        model_K=np.array(geom["model_intrinsics"]["K"], dtype=np.float64),
        native_K=np.array(geom["native_intrinsics"]["K"], dtype=np.float64),
        effective_model_fov_x_deg=float(geom["effective_model_horizontal_fov_deg"]),
        resize_meta=geom["resize_meta"],
    )

def create_camera_actor(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    runtime_cam: RuntimeCamera,
    attach_to: carla.Actor,
    default_sensor_tick: float,
) -> carla.Actor:
    spec = runtime_cam.spec
    bp = bp_lib.find(spec.blueprint)
    bp.set_attribute("image_size_x", str(runtime_cam.model_width))
    bp.set_attribute("image_size_y", str(runtime_cam.model_height))
    bp.set_attribute("fov", str(runtime_cam.effective_model_fov_x_deg))
    bp.set_attribute("sensor_tick", str(spec.sensor_tick if spec.sensor_tick is not None else default_sensor_tick))

    for key, value in spec.extra_attributes.items():
        bp.set_attribute(str(key), str(value))

    return world.spawn_actor(bp, pose_to_transform(spec.pose), attach_to=attach_to)


def create_lidar_actor(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    spec: LidarSpec,
    attach_to: carla.Actor,
    default_sensor_tick: float,
) -> carla.Actor:
    bp = bp_lib.find(spec.blueprint)
    bp.set_attribute("channels", str(spec.channels))
    bp.set_attribute("range", str(spec.range_m))
    bp.set_attribute("points_per_second", str(spec.points_per_second))
    bp.set_attribute("rotation_frequency", str(spec.rotation_frequency_hz))
    bp.set_attribute("upper_fov", str(spec.upper_fov_deg))
    bp.set_attribute("lower_fov", str(spec.lower_fov_deg))
    bp.set_attribute("horizontal_fov", str(spec.horizontal_fov_deg))
    bp.set_attribute("sensor_tick", str(spec.sensor_tick if spec.sensor_tick is not None else default_sensor_tick))

    for key, value in spec.extra_attributes.items():
        bp.set_attribute(str(key), str(value))

    return world.spawn_actor(bp, pose_to_transform(spec.pose), attach_to=attach_to)


def make_queue_for_sensor(sensor: carla.Actor) -> pyqueue.Queue:
    q = pyqueue.Queue()
    sensor.listen(q.put)
    return q


def spawn_ego_vehicle(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    traffic_manager: carla.TrafficManager,
) -> carla.Actor:
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("This map has no spawn points.")

    idx = max(0, min(CFG.spawn_point_index, len(spawn_points) - 1))
    spawn_tf = spawn_points[idx]

    bp_list = bp_lib.filter(CFG.ego_vehicle_filter)
    if not bp_list:
        raise RuntimeError(f"No blueprint matched ego_vehicle_filter={CFG.ego_vehicle_filter}")

    ego_bp = bp_list[0]
    if ego_bp.has_attribute("role_name"):
        ego_bp.set_attribute("role_name", "ego")

    ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle. Try another spawn_point_index.")

    if CFG.ego_autopilot:
        ego.set_autopilot(True, traffic_manager.get_port())

    return ego


def spawn_npc_vehicles(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    traffic_manager: carla.TrafficManager,
    ego_vehicle: carla.Actor,
) -> List[carla.Actor]:
    actors: List[carla.Actor] = []

    if CFG.npc_vehicle_count <= 0:
        print("Spawned 0 NPC vehicles.")
        return actors

    spawn_points = world.get_map().get_spawn_points()
    ego_loc = ego_vehicle.get_transform().location

    # Prefer spawn points near the ego route to avoid empty frames.
    spawn_points = sorted(spawn_points, key=lambda sp: sp.location.distance(ego_loc))
    spawn_points = spawn_points[: max(80, CFG.npc_vehicle_count * 3)]
    random.shuffle(spawn_points)

    vehicle_bps = list(bp_lib.filter("vehicle.*"))
    random.shuffle(vehicle_bps)

    count = 0
    for sp in spawn_points:
        if count >= CFG.npc_vehicle_count:
            break

        bp = random.choice(vehicle_bps)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")

        actor = world.try_spawn_actor(bp, sp)
        if actor is None:
            continue

        actor.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(-20.0, 20.0))

        actors.append(actor)
        count += 1

    print(f"Spawned {len(actors)} NPC vehicles.")
    return actors


def spawn_walkers(world: carla.World, bp_lib: carla.BlueprintLibrary) -> Tuple[List[carla.Actor], List[carla.Actor]]:
    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []

    if CFG.npc_walker_count <= 0:
        print("Spawned 0 walkers.")
        return walkers, controllers

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")

    spawn_points = []
    for _ in range(CFG.npc_walker_count * 5):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_points.append(carla.Transform(loc))
        if len(spawn_points) >= CFG.npc_walker_count:
            break

    for sp in spawn_points:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        walker = world.try_spawn_actor(bp, sp)
        if walker is not None:
            walkers.append(walker)

    if walkers:
        world.tick()

    for walker in walkers:
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            controllers.append(controller)

    if controllers:
        world.tick()
        for controller in controllers:
            controller.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(1.2 + random.random())

    print(f"Spawned {len(walkers)} walkers.")
    return walkers, controllers

def carla_actor_to_dataset_class(actor: carla.Actor) -> Optional[str]:
    tid = actor.type_id.lower()

    if tid.startswith("walker."):
        return "pedestrian"

    if not tid.startswith("vehicle."):
        return None

    if "bus" in tid:
        return "bus"

    if any(k in tid for k in ["firetruck", "ambulance", "carlacola", "cybertruck", "truck"]):
        return "truck"

    if any(k in tid for k in ["harley", "vespa", "yamaha", "kawasaki", "motorcycle"]):
        return "motorcycle"

    if any(k in tid for k in ["bh.crossbike", "gazelle", "diamondback", "century", "bicycle"]):
        return "bicycle"

    return "car"


def save_carla_rgb_image(image: carla.Image, filepath: Path) -> None:

    try:
        from PIL import Image

        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = bgra[:, :, :3][:, :, ::-1]
        Image.fromarray(rgb, mode="RGB").save(str(filepath))
    except Exception:
        image.save_to_disk(str(filepath))


def lidar_measurement_to_numpy(lidar_measurement: carla.LidarMeasurement) -> np.ndarray:
    points = np.frombuffer(bytes(lidar_measurement.raw_data), dtype=np.float32)
    return points.reshape(-1, 4).copy()


def carla_lidar_points_to_sparsefusion(points_carla: np.ndarray) -> np.ndarray:

    points = np.asarray(points_carla, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected CARLA LiDAR points with shape (N, >=4), got {points.shape}")

    out = points[:, :4].copy()
    out[:, 1] *= -1.0
    return out


def save_lidar_bin_5d(points_sparsefusion: np.ndarray, filepath: Path) -> None:

    points = np.asarray(points_sparsefusion, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (4, 5):
        raise ValueError(f"Expected LiDAR points with shape (N, 4) or (N, 5), got {points.shape}")

    if points.shape[1] == 4:
        time_lag = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.concatenate([points, time_lag], axis=1)

    points.astype(np.float32).tofile(str(filepath))


def carla_camera_to_standard_camera_axes_matrix() -> np.ndarray:

    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def lidar_carla_to_camera_standard_4x4(camera_actor: carla.Actor, lidar_actor: carla.Actor) -> np.ndarray:

    T_world_lidar_carla = matrix_from_transform(lidar_actor.get_transform())
    T_camUE_world = inverse_matrix_from_transform(camera_actor.get_transform())
    T_camUE_lidar_carla = T_camUE_world @ T_world_lidar_carla

    R_axes = np.eye(4, dtype=np.float64)
    R_axes[:3, :3] = carla_camera_to_standard_camera_axes_matrix()

    return R_axes @ T_camUE_lidar_carla


def lidar_sparsefusion_to_camera_standard_4x4(camera_actor: carla.Actor, lidar_actor: carla.Actor) -> np.ndarray:

    F = sparsefusion_from_carla_lidar_4x4()
    return lidar_carla_to_camera_standard_4x4(camera_actor, lidar_actor) @ F


def camera_standard_to_lidar_sparsefusion(
    camera_actor: carla.Actor,
    lidar_actor: carla.Actor,
) -> Tuple[np.ndarray, np.ndarray]:

    T_cam_lidar_sf = lidar_sparsefusion_to_camera_standard_4x4(camera_actor, lidar_actor)
    T_lidar_sf_cam = np.linalg.inv(T_cam_lidar_sf)

    R = T_lidar_sf_cam[:3, :3].astype(np.float32)
    t = T_lidar_sf_cam[:3, 3].astype(np.float32)

    return R, t


def lidar_to_image_4x4(K: np.ndarray, camera_actor: carla.Actor, lidar_actor: carla.Actor) -> np.ndarray:
    T_cam_lidar_sf = lidar_sparsefusion_to_camera_standard_4x4(camera_actor, lidar_actor)
    P = np.eye(4, dtype=np.float64)
    P[:3, :3] = K
    return P @ T_cam_lidar_sf


def lidar_sparsefusion_to_world_4x4(lidar_actor: carla.Actor) -> np.ndarray:
    F = sparsefusion_from_carla_lidar_4x4()
    return matrix_from_transform(lidar_actor.get_transform()) @ F


def world_to_lidar_sparsefusion_4x4(lidar_actor: carla.Actor) -> np.ndarray:
    F = sparsefusion_from_carla_lidar_4x4()
    return F @ inverse_matrix_from_transform(lidar_actor.get_transform())

def actor_dimensions_lwh(actor: carla.Actor) -> Tuple[float, float, float]:
    ext = actor.bounding_box.extent
    length = 2.0 * ext.x
    width = 2.0 * ext.y
    height = 2.0 * ext.z
    return float(length), float(width), float(height)


def actor_box_lidar_sparsefusion(actor: carla.Actor, lidar_actor: carla.Actor) -> List[float]:

    actor_tf = actor.get_transform()
    bbox_center_world = actor_tf.transform(actor.bounding_box.location)

    T_lidar_sf_world = world_to_lidar_sparsefusion_4x4(lidar_actor)
    p_world = np.array(
        [bbox_center_world.x, bbox_center_world.y, bbox_center_world.z, 1.0],
        dtype=np.float64,
    )
    p_lidar_sf = T_lidar_sf_world @ p_world

    forward = actor_tf.get_forward_vector()
    p0 = actor_tf.location
    p1 = carla.Location(x=p0.x + forward.x, y=p0.y + forward.y, z=p0.z + forward.z)

    v0 = T_lidar_sf_world @ np.array([p0.x, p0.y, p0.z, 1.0], dtype=np.float64)
    v1 = T_lidar_sf_world @ np.array([p1.x, p1.y, p1.z, 1.0], dtype=np.float64)
    v = v1[:3] - v0[:3]

    yaw = wrap_to_pi(math.atan2(v[1], v[0]))

    length, width, height = actor_dimensions_lwh(actor)

    return [
        float(p_lidar_sf[0]),
        float(p_lidar_sf[1]),
        float(p_lidar_sf[2]),
        length,
        width,
        height,
        yaw,
    ]


def actor_velocity_lidar_sparsefusion(actor: carla.Actor, lidar_actor: carla.Actor) -> List[float]:
    vel = actor.get_velocity()
    v_world = np.array([vel.x, vel.y, vel.z], dtype=np.float64)

    T_lidar_sf_world = world_to_lidar_sparsefusion_4x4(lidar_actor)
    R_lidar_sf_world = T_lidar_sf_world[:3, :3]
    v_lidar_sf = R_lidar_sf_world @ v_world

    return [float(v_lidar_sf[0]), float(v_lidar_sf[1])]


def is_box_center_in_point_cloud_range(box_lidar: List[float]) -> bool:
    x_min, y_min, z_min, x_max, y_max, z_max = CFG.point_cloud_range
    x, y, z = box_lidar[:3]
    return (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max)


def count_points_in_box_lidar(points: np.ndarray, box_lidar: List[float]) -> int:

    if points.size == 0:
        return 0

    cx, cy, cz, dx, dy, dz, yaw = box_lidar
    shifted = points[:, :3] - np.array([cx, cy, cz], dtype=np.float32)

    c = math.cos(-yaw)
    s = math.sin(-yaw)

    x_local = c * shifted[:, 0] - s * shifted[:, 1]
    y_local = s * shifted[:, 0] + c * shifted[:, 1]
    z_local = shifted[:, 2]

    mask = (
        (np.abs(x_local) <= dx / 2.0)
        & (np.abs(y_local) <= dy / 2.0)
        & (np.abs(z_local) <= dz / 2.0)
    )
    return int(mask.sum())


def build_objects_for_sample(
    actors_for_labels: List[carla.Actor],
    lidar_actor: carla.Actor,
    lidar_points_sparsefusion: np.ndarray,
) -> List[Dict]:
    objects: List[Dict] = []

    for actor in actors_for_labels:
        name = carla_actor_to_dataset_class(actor)
        if name is None or name not in CFG.class_names:
            continue

        box = actor_box_lidar_sparsefusion(actor, lidar_actor)
        if not is_box_center_in_point_cloud_range(box):
            continue

        npts = count_points_in_box_lidar(lidar_points_sparsefusion, box)
        if npts < CFG.min_lidar_points_per_object:
            continue

        actor_tf = actor.get_transform()

        objects.append(
            {
                "name": name,
                "box_lidar": box,
                "velocity": actor_velocity_lidar_sparsefusion(actor, lidar_actor),
                "num_lidar_pts": int(npts),
                "actor_id": int(actor.id),
                "actor_type_id": actor.type_id,
                "world_location_carla": [
                    float(actor_tf.location.x),
                    float(actor_tf.location.y),
                    float(actor_tf.location.z),
                ],
            }
        )

    return objects


def prepare_dataset_dirs(base_dir: Path) -> None:
    for cam_name in CFG.camera_order:
        ensure_dir(base_dir / "samples" / cam_name)

    ensure_dir(base_dir / "samples" / CFG.lidar.name)
    ensure_dir(base_dir / "annotations")
    ensure_dir(base_dir / "calib")
    ensure_dir(base_dir / "meta")


def save_sensor_manifest(base_dir: Path, runtime_cameras: Dict[str, RuntimeCamera]) -> None:
    payload = {
        "capture_config": config_to_dict(CFG),
        "runtime_cameras": {
            name: {
                "model_height": rc.model_height,
                "model_width": rc.model_width,
                "model_K": rc.model_K.tolist(),
                "native_K": rc.native_K.tolist(),
                "effective_model_fov_x_deg": rc.effective_model_fov_x_deg,
                "resize_meta": rc.resize_meta,
            }
            for name, rc in runtime_cameras.items()
        },
        "lidar_fields": ["x", "y", "z", "intensity", "time_lag"],
        "lidar_dim": 5,
        "coordinate_note": (
            "Generated LiDAR points, box_lidar, object velocity, sensor2lidar, and lidar2img use "
            "SparseFusion/nuScenes-style LiDAR coordinates: x forward, y left, z up. "
            "CARLA native LiDAR coordinates are x forward, y right, z up and are converted during export. "
            "Camera intrinsics use standard pinhole camera coordinates: x right, y down, z forward."
        ),
        "carla_lidar_to_sparsefusion_lidar_4x4": sparsefusion_from_carla_lidar_4x4().tolist(),
    }
    save_json(base_dir / "meta" / "sensor_setup.json", payload)


def build_sample_record(
    sample_id: str,
    timestamp_us: int,
    base_dir: Path,
    runtime_cameras: Dict[str, RuntimeCamera],
    lidar_actor: carla.Actor,
    lidar_rel_path: str,
    objects: List[Dict],
) -> Dict:
    cams: Dict[str, Dict] = {}

    for cam_name in CFG.camera_order:
        rc = runtime_cameras[cam_name]
        if rc.actor is None:
            raise RuntimeError(f"Camera actor for {cam_name} is None")

        img_rel_path = f"samples/{cam_name}/{sample_id}.png"
        R_s2l, t_s2l = camera_standard_to_lidar_sparsefusion(rc.actor, lidar_actor)

        cam_payload = {
            "data_path": img_rel_path,
            "sample_data_token": f"{sample_id}_{cam_name}",
            "sensor2lidar_rotation": R_s2l.tolist(),
            "sensor2lidar_translation": t_s2l.tolist(),
            "cam_intrinsic": rc.model_K.tolist(),
            "width": int(rc.model_width),
            "height": int(rc.model_height),
        }
        cams[cam_name] = cam_payload

        calib_payload = {
            "token": sample_id,
            "camera_name": cam_name,
            "cam_intrinsic": rc.model_K.tolist(),
            "sensor2lidar_rotation": R_s2l.tolist(),
            "sensor2lidar_translation": t_s2l.tolist(),
            "lidar2img": lidar_to_image_4x4(rc.model_K, rc.actor, lidar_actor).tolist(),
            "camera_to_world_4x4": matrix_from_transform(rc.actor.get_transform()).tolist(),
            "lidar_sparsefusion_to_world_4x4": lidar_sparsefusion_to_world_4x4(lidar_actor).tolist(),
            "world_to_lidar_sparsefusion_4x4": world_to_lidar_sparsefusion_4x4(lidar_actor).tolist(),
        }
        save_json(base_dir / "calib" / f"{sample_id}_{cam_name}.json", calib_payload)

    return {
        "token": sample_id,
        "timestamp": int(timestamp_us),
        "lidar_path": lidar_rel_path,
        "sweeps": [],
        "cams": cams,
        "objects": objects,
    }


def write_split_annotations(base_dir: Path, samples: List[Dict]) -> None:
    n = len(samples)
    n_train = int(round(n * CFG.train_ratio))
    n_train = max(1, min(n_train, n - 1)) if n > 1 else n

    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    common = {
        "camera_order": list(CFG.camera_order),
        "class_names": list(CFG.class_names),
        "point_cloud_range": list(CFG.point_cloud_range),
        "img_scale": [int(CFG.model_image_width), int(CFG.model_image_height)],
        "lidar_fields": ["x", "y", "z", "intensity", "time_lag"],
        "lidar_dim": 5,
        "coordinate_note": (
            "Generated LiDAR frame: SparseFusion/nuScenes-style x forward, y left, z up. "
            "CARLA native y-right LiDAR data has already been converted."
        ),
    }

    train_json = dict(common)
    train_json["samples"] = train_samples

    val_json = dict(common)
    val_json["samples"] = val_samples

    save_json(base_dir / "annotations" / "train.json", train_json)
    save_json(base_dir / "annotations" / "val.json", val_json)

    print(f"Wrote {len(train_samples)} train samples.")
    print(f"Wrote {len(val_samples)} val samples.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 4-camera + LiDAR SparseFusion-style dataset from CARLA."
    )
    parser.add_argument(
        "--map",
        "--map-name",
        dest="map_name",
        default=CFG.map_name,
        help="CARLA map name, for example Town03_Opt or Town01_Opt.",
    )
    parser.add_argument(
        "--name",
        "--data-name",
        dest="data_name",
        default=CFG.sequence_name,
        help="dataset name.",
    )
    parser.add_argument(
        "--frame",
        "--num-frame",
        dest="frame",
        type=int,
        default=CFG.num_frames,
        help="Number of frames.",
    )
    parser.add_argument(
        "--cars",
        "--num-cars",
        dest="cars",
        type=int,
        default=CFG.npc_vehicle_count,
        help="Number of NPC vehicles to spawn.",
    )
    parser.add_argument(
        "--humans",
        "--num-humans",
        dest="humans",
        type=int,
        default=CFG.npc_walker_count,
        help="Number of pedestrians/walkers to spawn.",
    )
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
    CFG.map_name = args.map_name
    CFG.npc_vehicle_count = max(0, int(args.cars))
    CFG.npc_walker_count = max(0, int(args.humans))
    CFG.num_frames = args.frame
    CFG.sequence_name = args.data_name

def main() -> None:
    args = parse_args()
    apply_cli_overrides(args)

    random.seed(CFG.seed)
    np.random.seed(CFG.seed)

    if CFG.lidar is None:
        raise RuntimeError("CFG.lidar is None. Define one LiDAR in sim_carla.py.")

    assert CFG.camera_order == ["CAM_FRONT", "CAM_LEFT", "CAM_BACK", "CAM_RIGHT"], (
        "Camera order must match the expected converter order."
    )

    base_dir = ensure_dir(expand_user_path(CFG.output_root) / CFG.sequence_name)
    prepare_dataset_dirs(base_dir)

    runtime_cameras: Dict[str, RuntimeCamera] = {
        spec.name: make_runtime_camera(spec) for spec in CFG.cameras
    }

    missing = set(CFG.camera_order) - set(runtime_cameras.keys())
    if missing:
        raise RuntimeError(f"Missing cameras from CFG.cameras: {sorted(missing)}")

    save_sensor_manifest(base_dir, runtime_cameras)
    
    llm = 2
    if llm == 2:
    	print(f"Saving dir is: {base_dir}")

    client = carla.Client(CFG.host, CFG.port)
    client.set_timeout(CFG.timeout_seconds)
    
    print(f"The selected map is {CFG.map_name}")	
    
    
    world = choose_world(client)
    bp_lib = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager()

    original_settings = world.get_settings()

    delta = 1.0 / CFG.simulation_fps

    all_spawned_actors: List[carla.Actor] = []
    walker_controllers: List[carla.Actor] = []
    sensor_handles: Dict[str, SensorHandle] = {}
    samples: List[Dict] = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = delta
        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(CFG.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
        try:
            traffic_manager.set_respawn_dormant_vehicles(True)
        except Exception:
            pass

        ego = spawn_ego_vehicle(world, bp_lib, traffic_manager)
        all_spawned_actors.append(ego)

        npc_vehicles = spawn_npc_vehicles(world, bp_lib, traffic_manager, ego)
        all_spawned_actors.extend(npc_vehicles)

        walkers, controllers = spawn_walkers(world, bp_lib)
        all_spawned_actors.extend(walkers)
        all_spawned_actors.extend(controllers)
        walker_controllers.extend(controllers)

        for cam_name in CFG.camera_order:
            rc = runtime_cameras[cam_name]
            actor = create_camera_actor(world, bp_lib, rc, ego, delta)
            q = make_queue_for_sensor(actor)
            rc.actor = actor
            rc.queue = q
            sensor_handles[cam_name] = SensorHandle(cam_name, actor, q)
            all_spawned_actors.append(actor)

        lidar_actor = create_lidar_actor(world, bp_lib, CFG.lidar, ego, delta)
        lidar_queue = make_queue_for_sensor(lidar_actor)
        sensor_handles[CFG.lidar.name] = SensorHandle(CFG.lidar.name, lidar_actor, lidar_queue)
        all_spawned_actors.append(lidar_actor)

        for _ in range(CFG.warmup_ticks):
            world.tick()
            for handle in sensor_handles.values():
                try:
                    handle.queue.get(timeout=5.0)
                except pyqueue.Empty:
                    pass

        print(f"Saving dataset to: {base_dir}")
        print(f"Map in use: {world.get_map().name}")
        print(f"NPC vehicles requested: {CFG.npc_vehicle_count}")
        print(f"NPC humans requested: {CFG.npc_walker_count}")
        print(f"Model image size: W={CFG.model_image_width}, H={CFG.model_image_height}")
        print(f"Camera order: {CFG.camera_order}")
        print(f"Class names: {CFG.class_names}")
        print(f"Point cloud range: {CFG.point_cloud_range}")
        print("Output LiDAR convention: x forward, y left, z up")
        print("Output LiDAR bin format: float32 [x, y, z, intensity, time_lag] = 5D")

        for cam_name in CFG.camera_order:
            rc = runtime_cameras[cam_name]
            print(
                f"{cam_name}: native={rc.spec.native_width}x{rc.spec.native_height} "
                f"-> model={rc.model_width}x{rc.model_height}, "
                f"effective_fov_x={rc.effective_model_fov_x_deg:.3f} deg"
            )

        saved_count = 0
        capture_tick = 0

        while saved_count < CFG.num_frames:
            frame = world.tick()
            capture_tick += 1

            synced = {}
            for name, handle in sensor_handles.items():
                data = handle.queue.get(timeout=10.0)
                while data.frame < frame:
                    data = handle.queue.get(timeout=10.0)
                if data.frame != frame:
                    raise RuntimeError(
                        f"Frame mismatch on sensor {name}: world={frame}, sensor={data.frame}"
                    )
                synced[name] = data

            if capture_tick % CFG.capture_every_n_ticks != 0:
                continue

            sample_id = f"{saved_count:06d}"
            timestamp_us = int(float(getattr(synced[CFG.lidar.name], "timestamp", frame * delta)) * 1_000_000)

            for cam_name in CFG.camera_order:
                img_path = base_dir / "samples" / cam_name / f"{sample_id}.png"
                save_carla_rgb_image(synced[cam_name], img_path)

            lidar_points_carla = lidar_measurement_to_numpy(synced[CFG.lidar.name])
            lidar_points_sparsefusion = carla_lidar_points_to_sparsefusion(lidar_points_carla)

            lidar_rel_path = f"samples/{CFG.lidar.name}/{sample_id}.bin"
            lidar_path = base_dir / lidar_rel_path
            save_lidar_bin_5d(lidar_points_sparsefusion, lidar_path)

            actors_for_labels = [a for a in world.get_actors() if a.id != ego.id]
            objects = build_objects_for_sample(
                actors_for_labels=actors_for_labels,
                lidar_actor=lidar_actor,
                lidar_points_sparsefusion=lidar_points_sparsefusion,
            )

            sample = build_sample_record(
                sample_id=sample_id,
                timestamp_us=timestamp_us,
                base_dir=base_dir,
                runtime_cameras=runtime_cameras,
                lidar_actor=lidar_actor,
                lidar_rel_path=lidar_rel_path,
                objects=objects,
            )
            samples.append(sample)

            saved_count += 1
            if saved_count % 10 == 0 or saved_count == CFG.num_frames:
                print(f"Captured {saved_count}/{CFG.num_frames} frames | objects in last frame: {len(objects)}")

        write_split_annotations(base_dir, samples)
        print("Capture finished successfully.")

    finally:
        for handle in sensor_handles.values():
            try:
                handle.actor.stop()
            except Exception:
                pass

        for controller in walker_controllers:
            try:
                controller.stop()
            except Exception:
                pass

        for actor in reversed(all_spawned_actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass

        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("Cleaned up actors and restored world settings.")


if __name__ == "__main__":
    main()
