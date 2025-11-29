#!/bin/bash
# FastDDS Environment Setup Script - ROBOT (Server)

# Set ROS domain ID (same on both devices)
export ROS_DOMAIN_ID=0

# Robot acts as DISCOVERY SERVER - no ROS_DISCOVERY_SERVER variable needed
unset ROS_DISCOVERY_SERVER

# Robot uses SERVER configuration - no XML file needed
unset FASTRTPS_DEFAULT_PROFILES_FILE

# Enable multicast for server
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

# Optional: Set log level for debugging
export FASTDDS_VERBOSITY=info

echo "Robot (Discovery Server) environment variables set:"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "ROS_DISCOVERY_SERVER: (unset - robot is the server)"
echo "FASTRTPS_DEFAULT_PROFILES_FILE: (unset - server doesn't need XML)"
echo "ROS_AUTOMATIC_DISCOVERY_RANGE: $ROS_AUTOMATIC_DISCOVERY_RANGE"

# Start the discovery server
echo "Starting FastDDS Discovery Server on port 11811..."
fastdds discovery --server-id 0 --port 11811
