#!/usr/bin/env python3
import time
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped

def main():
    rclpy.init()
    nav = BasicNavigator()

    # --- CONFIGURATION ---
    # Set your 4 corner coordinates here (x, y)
    # These should be safe points AROUND the pond
    waypoints = [
        [5.0, 6.5],   # Corner 1
        [5.0, -6.5],  # Corner 2
        [-5.0, -6.5], # Corner 3
        [-5.0, 6.5]   # Corner 4
    ]
    # ---------------------

    # Wait for Nav2 to fully launch
    print("System is active (SLAM mode). Waiting 5 seconds to ensure connections...")
    time.sleep(5)

    # Create the list of PoseStamped messages
    goal_poses = []
    for point in waypoints:
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = nav.get_clock().now().to_msg()
        goal_pose.pose.position.x = point[0]
        goal_pose.pose.position.y = point[1]
        
        # Orientation (quaternion for "facing forward" approx)
        # For now, just keeping it 0.0 (no rotation logic yet)
        goal_pose.pose.orientation.w = 1.0 
        goal_poses.append(goal_pose)

    # --- THE PATROL LOOP ---
    # This matches your task: "Patrol the area"
    nav.followWaypoints(goal_poses)

    # Wait until the task is complete
    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        # You can print feedback here if you want
        
    result = nav.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Patrol Complete!')
    else:
        print('Patrol Failed!')

    rclpy.shutdown()

if __name__ == '__main__':
    main()