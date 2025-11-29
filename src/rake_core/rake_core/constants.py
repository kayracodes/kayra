from enum import IntEnum


class Topics:
    DEVICE_STATE = "/rake/device_state"
    SYSTEM_STATE = "/rake/system_state"
    CONFIG_UPDATE = "/rake/config_updated"


class Services:
    DEVICE_STATE = "/rake/device_state_client"
    SYSTEM_STATE = "/rake/system_state_client"
    CONFIG_UPDATE = "/rake/update_config_client"
    RAMP_UNDETECTED = "/rake/ramp_undetected_client"
    RAMP_DETECTED = "/rake/ramp_detected_client"
    RAMP_CLOSE = "/rake/ramp_close_client"
    SIGN_UNDETECTED = "/rake/sign_undetected_client"
    IN_WATER = "/rake/in_water_client"
    OUT_WATER = "/rake/out_water_client"
    LOAD_CONFIG = "/rake/load_config_client"
    SAVE_CONFIG = "/rake/save_config_client"
    LOAD_MISSION = "/rake/load_mission_client"


class Actions:
    RAMP_ALIGNMENT = "/ika_actions/ramp_alignment"
    ALIGN_WITH_PATH = "/ika_actions/align_with_path"
    LOCK_TARGET = "/ika_actions/lock_target"
    LOCK_TARGET_REAL = "/ika_actions/lock_target_real"


SIGN_ID_MAP = {
    0: "1",
    1: "10",
    2: "11",
    3: "12",
    4: "2",
    5: "3",
    6: "4",
    7: "4-END",
    8: "5",
    9: "6",
    10: "7",
    11: "8",
    12: "9",
    13: "STOP",
}

ID_SIGN_MAP = {v: k for k, v in SIGN_ID_MAP.items()}
