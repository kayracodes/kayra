from enum import IntEnum


class SystemStateEnum(IntEnum):
    MANUAL = 0
    SEMI_AUTONOMOUS = 1
    IDLE_AUTONOMOUS = 2
    SLOW_AUTONOMOUS = 3
    FAST_AUTONOMOUS = 4
    CAUTIOUS_AUTONOMOUS = 5
    SHARP_TURN_AUTONOMOUS = 6
    TARGET_HITTING = 7
    ACTION_CONTROLLED = 8
    ACTION_CONTROL_TEST = 9
    DISABLED = 10
    SHUTDOWN = 11


class DeviceStateEnum(IntEnum):
    BOOTING = 0
    READY = 1
    WORKING = 2
    ERRORED = 3
    SHUTDOWN = 4


class SystemModeEnum(IntEnum):
    SIMULATION = 0
    PRACTICE = 1
    COMPETITION = 2
