from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import math

"""
nuScenes-style LiDAR frame used by generated files --> x forward, y left, z up

4-CAMERA + LIDAR SENSOR RIG
- CAM_FRONT: 85° FoV, Tier IV C1-085
- CAM_BACK:  85° FoV, Tier IV C1-085
- CAM_LEFT:  120° FoV, Tier IV C1-120
- CAM_RIGHT: 120° FoV, Tier IV C1-120

"""

MODEL_IMAGE_WIDTH = 1920
MODEL_IMAGE_HEIGHT = 1280

NATIVE_IMAGE_WIDTH = 1920
NATIVE_IMAGE_HEIGHT = 1280

CAMERA_ORDER = ["CAM_FRONT", "CAM_LEFT", "CAM_BACK", "CAM_RIGHT"]
CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]

TRAIN_RATIO = 0.8

POINT_CLOUD_RANGE = [-90.0, -90.0, -5.0, 90.0, 90.0, 3.0]


MAP_NAME = "Town03_Opt"
SPAWN_POINT_INDEX = 10
SIMULATION_FPS = 10.0
NUM_FRAMES = 100

NPC_VEHICLE_COUNT = 60
NPC_WALKER_COUNT = 25
CAPTURE_EVERY_N_TICKS = 10

OUTPUT_ROOT = "/data_ssd"  # it must be modified based on the mounted drive
SEQUENCE_NAME = "sparsefusion_no_sweep_v1"


@dataclass
class Pose6:
    x: float
    y: float
    z: float
    wx: float = 0.0  # roll around x, degrees
    wy: float = 0.0  # pitch around y, degrees
    wz: float = 0.0  # yaw around z, degrees


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> List[List[float]]:
        return [
            [float(self.fx), 0.0, float(self.cx)],
            [0.0, float(self.fy), float(self.cy)],
            [0.0, 0.0, 1.0],
        ]


@dataclass
class CameraSpec:
    name: str
    pose: Pose6

    native_height: int
    native_width: int
    native_intrinsics: Optional[Intrinsics] = None
    native_horizontal_fov_deg: Optional[float] = None

    model_height: Optional[int] = None
    model_width: Optional[int] = None

    resize_mode: str = "scale_to_fill_center_crop"

    sensor_tick: Optional[float] = None
    blueprint: str = "sensor.camera.rgb"
    extra_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LidarSpec:
    name: str
    pose: Pose6
    channels: int
    range_m: float
    points_per_second: int
    rotation_frequency_hz: float
    upper_fov_deg: float
    lower_fov_deg: float
    horizontal_fov_deg: float = 360.0
    sensor_tick: Optional[float] = None
    blueprint: str = "sensor.lidar.ray_cast"
    extra_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureConfig:
    host: str = "localhost"
    port: int = 2000
    timeout_seconds: float = 60.0

    use_current_world: bool = False
    map_name: str = MAP_NAME
    map_layers: List[str] = field(default_factory=list)

    simulation_fps: float = SIMULATION_FPS
    num_frames: int = NUM_FRAMES
    capture_every_n_ticks: int = CAPTURE_EVERY_N_TICKS
    warmup_ticks: int = 20
    seed: int = 0

    output_root: str = OUTPUT_ROOT
    sequence_name: str = SEQUENCE_NAME

    model_image_height: int = MODEL_IMAGE_HEIGHT
    model_image_width: int = MODEL_IMAGE_WIDTH

    camera_order: List[str] = field(default_factory=lambda: list(CAMERA_ORDER))
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    point_cloud_range: List[float] = field(default_factory=lambda: list(POINT_CLOUD_RANGE))
    train_ratio: float = TRAIN_RATIO

    ego_vehicle_filter: str = "vehicle.lincoln.mkz_2020"
    spawn_point_index: int = SPAWN_POINT_INDEX
    ego_autopilot: bool = True

    npc_vehicle_count: int = NPC_VEHICLE_COUNT
    npc_walker_count: int = NPC_WALKER_COUNT

    min_lidar_points_per_object: int = 4

    cameras: List[CameraSpec] = field(default_factory=list)
    lidar: Optional[LidarSpec] = None


def horizontal_fov_from_fx(width: int, fx: float) -> float:
    return math.degrees(2.0 * math.atan(width / (2.0 * fx)))


def infer_intrinsics_from_horizontal_fov(width: int, height: int, horizontal_fov_deg: float) -> Intrinsics:
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def resolve_native_intrinsics(camera: CameraSpec) -> Intrinsics:
    if camera.native_intrinsics is not None:
        return camera.native_intrinsics
    if camera.native_horizontal_fov_deg is None:
        raise ValueError(
            f"Camera {camera.name}: provide either native_intrinsics or native_horizontal_fov_deg"
        )
    return infer_intrinsics_from_horizontal_fov(
        width=camera.native_width,
        height=camera.native_height,
        horizontal_fov_deg=camera.native_horizontal_fov_deg,
    )


def derive_model_camera_geometry(camera: CameraSpec, cfg: CaptureConfig) -> Dict[str, Any]:
    native_K = resolve_native_intrinsics(camera)

    target_h = int(camera.model_height if camera.model_height is not None else cfg.model_image_height)
    target_w = int(camera.model_width if camera.model_width is not None else cfg.model_image_width)

    src_w = float(camera.native_width)
    src_h = float(camera.native_height)

    fx0 = float(native_K.fx)
    fy0 = float(native_K.fy)
    cx0 = float(native_K.cx)
    cy0 = float(native_K.cy)

    if camera.resize_mode == "scale_to_fill_center_crop":
        scale = max(target_w / src_w, target_h / src_h)
        scaled_w = src_w * scale
        scaled_h = src_h * scale
        crop_left = (scaled_w - target_w) / 2.0
        crop_top = (scaled_h - target_h) / 2.0

        fx = fx0 * scale
        fy = fy0 * scale
        cx = cx0 * scale - crop_left
        cy = cy0 * scale - crop_top

        resize_meta = {
            "mode": "scale_to_fill_center_crop",
            "scale_x": scale,
            "scale_y": scale,
            "scaled_width": scaled_w,
            "scaled_height": scaled_h,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_width": float(target_w),
            "crop_height": float(target_h),
        }

    elif camera.resize_mode == "scale_only":
        scale_x = target_w / src_w
        scale_y = target_h / src_h

        fx = fx0 * scale_x
        fy = fy0 * scale_y
        cx = cx0 * scale_x
        cy = cy0 * scale_y

        resize_meta = {
            "mode": "scale_only",
            "scale_x": scale_x,
            "scale_y": scale_y,
            "scaled_width": float(target_w),
            "scaled_height": float(target_h),
            "crop_left": 0.0,
            "crop_top": 0.0,
            "crop_width": float(target_w),
            "crop_height": float(target_h),
        }
    else:
        raise ValueError(f"Unsupported resize_mode={camera.resize_mode}")

    model_K = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    return {
        "camera_name": camera.name,
        "native_size_hw": [int(camera.native_height), int(camera.native_width)],
        "model_size_hw": [int(target_h), int(target_w)],
        "native_intrinsics": {
            "fx": fx0,
            "fy": fy0,
            "cx": cx0,
            "cy": cy0,
            "K": native_K.as_matrix(),
        },
        "model_intrinsics": {
            "fx": float(model_K.fx),
            "fy": float(model_K.fy),
            "cx": float(model_K.cx),
            "cy": float(model_K.cy),
            "K": model_K.as_matrix(),
        },
        "native_horizontal_fov_deg": float(horizontal_fov_from_fx(camera.native_width, fx0)),
        "effective_model_horizontal_fov_deg": float(horizontal_fov_from_fx(target_w, fx)),
        "resize_meta": resize_meta,
    }



FRONT_REAR_INTRINSICS_NATIVE = Intrinsics(fx=1047.8, fy=1047.8, cx=960.0, cy=640.0)
SIDE_INTRINSICS_NATIVE = Intrinsics(fx=554.26, fy=554.26, cx=960.0, cy=640.0)

CFG = CaptureConfig(
    cameras=[
        CameraSpec(
            name="CAM_FRONT",
            pose=Pose6(x=1.50, y=0.00, z=1.70, wx=0.0, wy=0.0, wz=0.0),
            native_height=NATIVE_IMAGE_HEIGHT,
            native_width=NATIVE_IMAGE_WIDTH,
            native_intrinsics=FRONT_REAR_INTRINSICS_NATIVE,
            native_horizontal_fov_deg=85.0,
            model_height=MODEL_IMAGE_HEIGHT,
            model_width=MODEL_IMAGE_WIDTH,
            resize_mode="scale_to_fill_center_crop",
        ),
        CameraSpec(
            name="CAM_LEFT",
            pose=Pose6(x=0.20, y=-0.35, z=1.70, wx=0.0, wy=0.0, wz=-90.0),
            native_height=NATIVE_IMAGE_HEIGHT,
            native_width=NATIVE_IMAGE_WIDTH,
            native_intrinsics=SIDE_INTRINSICS_NATIVE,
            native_horizontal_fov_deg=120.0,
            model_height=MODEL_IMAGE_HEIGHT,
            model_width=MODEL_IMAGE_WIDTH,
            resize_mode="scale_to_fill_center_crop",
        ),
        CameraSpec(
            name="CAM_BACK",
            pose=Pose6(x=-1.30, y=0.00, z=1.70, wx=0.0, wy=0.0, wz=180.0),
            native_height=NATIVE_IMAGE_HEIGHT,
            native_width=NATIVE_IMAGE_WIDTH,
            native_intrinsics=FRONT_REAR_INTRINSICS_NATIVE,
            native_horizontal_fov_deg=85.0,
            model_height=MODEL_IMAGE_HEIGHT,
            model_width=MODEL_IMAGE_WIDTH,
            resize_mode="scale_to_fill_center_crop",
        ),
        CameraSpec(
            name="CAM_RIGHT",
            pose=Pose6(x=0.20, y=0.35, z=1.70, wx=0.0, wy=0.0, wz=90.0),
            native_height=NATIVE_IMAGE_HEIGHT,
            native_width=NATIVE_IMAGE_WIDTH,
            native_intrinsics=SIDE_INTRINSICS_NATIVE,
            native_horizontal_fov_deg=120.0,
            model_height=MODEL_IMAGE_HEIGHT,
            model_width=MODEL_IMAGE_WIDTH,
            resize_mode="scale_to_fill_center_crop",
        ),
    ],
    lidar=LidarSpec(
        name="LIDAR_TOP",
        pose=Pose6(x=0.00, y=0.00, z=2.20, wx=0.0, wy=0.0, wz=0.0),
        channels=64,
        range_m=200.0,
        points_per_second=120000,
        rotation_frequency_hz=10.0,
        upper_fov_deg=15.0,
        lower_fov_deg=-25.0,
        horizontal_fov_deg=360.0,
        extra_attributes={
            "dropoff_general_rate": 0.0,
            "noise_stddev": 0.02,
        },
    ),
)


def config_to_dict(cfg: CaptureConfig) -> Dict[str, Any]:
    return asdict(cfg)
