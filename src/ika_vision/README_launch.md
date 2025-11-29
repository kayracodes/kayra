# Camera Shooter Launch File

This launch file orchestrates the camera shooter system with joystick control.

## Nodes Launched

1. **joy_controller** - Joystick controller node from ika_controller package
2. **shoot_camera_node** - Camera node from ika_controller package  
3. **camera_shooter** - Main shooter control node from ika_vision package

## Usage

### Basic Launch
```bash
ros2 launch ika_vision shoot_test.launch.py
```

### Launch with Custom Parameters
```bash
ros2 launch ika_vision shoot_test.launch.py \
    serial_port:=/dev/ttyUSB1 \
    laser_offset:=10.0 \
    tilt_sensitivity:=2.0 \
    pan_sensitivity:=2.0 \
    state:=AUTONOMOUS \
    camera_udev:=/dev/video0
```

## Parameters

- **serial_port**: Serial port for communication (default: `/dev/ttyUSB0`)
- **laser_offset**: Laser offset in cm (default: `5.0`)
- **tilt_sensitivity**: Tilt sensitivity in degrees (default: `1.0`)
- **pan_sensitivity**: Pan sensitivity in degrees (default: `1.0`)
- **state**: Initial state - `MANUAL` or `AUTONOMOUS` (default: `MANUAL`)
- **camera_udev**: Camera device path (default: USB webcam path)

## Topics

### Published Topics
- `/ika_controller/joy_cmd` - Joystick commands
- `/shoot_camera/image_raw` - Camera feed
- `/camera_shooter/debug_image` - Debug visualization
- `/camera_shooter/debug_info` - Debug information

### Joystick Controls
- **Button 0**: Switch to Autonomous mode
- **Button 1**: Toggle laser on/off
- **Button 2**: Switch to Manual mode
- **D-pad Up/Down**: Tilt control
- **D-pad Left/Right**: Pan control

## Prerequisites

1. Make sure your joystick is connected
2. Camera device is available at the specified path
3. Serial device is available (if using serial communication)

## Configuration

Edit `config/camera_shooter_params.yaml` to modify default parameters.
