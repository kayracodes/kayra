#!/bin/bash
# FastDDS Environment Setup Script

# Set the discovery server (replace with your robot's actual IP)
export ROS_DISCOVERY_SERVER=192.168.0.1:11811

# Set ROS domain ID (make sure both devices use the same)
export ROS_DOMAIN_ID=0

# Set FastDDS XML configuration file path
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/rosgeek/Documents/ika_ws/setup/super_client_configuration.xml

# Disable multicast discovery (since we're using discovery server)
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# Optional: Set log level for debugging
export FASTDDS_VERBOSITY=info

echo "FastDDS environment variables set:"
echo "ROS_DISCOVERY_SERVER: $ROS_DISCOVERY_SERVER"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "FASTRTPS_DEFAULT_PROFILES_FILE: $FASTRTPS_DEFAULT_PROFILES_FILE"
echo "ROS_AUTOMATIC_DISCOVERY_RANGE: $ROS_AUTOMATIC_DISCOVERY_RANGE"
