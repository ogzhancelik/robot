#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
from tf_transformations import quaternion_from_euler

# --- GLOBAL VARIABLES ---
trash_pose = None
trash_detected = False

def trash_callback(msg):
    global trash_pose, trash_detected
    if not trash_detected:
        print("\n!!! TRASH RECEIVED !!!")
        trash_pose = msg
        trash_detected = True

def get_approach_pose(robot_pose, target_pose, distance_offset=0.5):
    """
    Calculates a pose 'distance_offset' meters away from the target, 
    facing the target.
    """
    rx, ry = robot_pose.pose.position.x, robot_pose.pose.position.y
    tx, ty = target_pose.pose.position.x, target_pose.pose.position.y

    # Vector from Robot to Trash
    dx = tx - rx
    dy = ty - ry
    dist = math.sqrt(dx**2 + dy**2)
    yaw = math.atan2(dy, dx)

    # Calculate stopping point (short of the target)
    new_dist = dist - distance_offset
    new_x = rx + (new_dist * math.cos(yaw))
    new_y = ry + (new_dist * math.sin(yaw))

    approach_pose = PoseStamped()
    approach_pose.header.frame_id = 'map'
    approach_pose.pose.position.x = new_x
    approach_pose.pose.position.y = new_y
    
    # Orientation: Face the trash
    q = quaternion_from_euler(0, 0, yaw)
    approach_pose.pose.orientation.x = q[0]
    approach_pose.pose.orientation.y = q[1]
    approach_pose.pose.orientation.z = q[2]
    approach_pose.pose.orientation.w = q[3]

    return approach_pose

def main():
    rclpy.init()
    
    # 1. Setup Node and Clients
    node = rclpy.create_node('patrol_controller')
    global trash_detected, trash_pose  # <--- FIX: Ensure we use global vars
    
    node.create_subscription(PoseStamped, '/trash_pose', trash_callback, 10)
    grasp_client = node.create_client(Trigger, '/grasp_trash')
    marker_pub = node.create_publisher(Marker, 'approach_marker', 10)

    # 2. Start Navigator
    nav = BasicNavigator()

    # --- CONFIGURATION ---
    # Patrol corners
    waypoints_list = [
        [10.0, 0.0], 
        [10.0, -12.0], 
        [0.0, -12.0],
        [0.0, 0.0]
    ]
    # ---------------------

    # 3. Set Initial Pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.pose.position.x = 0.0
    initial_pose.pose.position.y = 0.0
    initial_pose.pose.orientation.z = 0.0
    initial_pose.pose.orientation.w = 1.0
    nav.setInitialPose(initial_pose)

    print("Waiting for Nav2...")
    nav.waitUntilNav2Active()
    print("Nav2 Active! Starting patrol...")

    # Prepare Goals
    goal_poses = []
    for (x, y) in waypoints_list:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        goal_poses.append(pose)

    # 4. Main Loop
    while True:
        for i, target_pose in enumerate(goal_poses):
            print(f"--> Headed to Waypoint {i+1}")
            nav.goToPose(target_pose)

            # WHILE MOVING
            while not nav.isTaskComplete():
                rclpy.spin_once(node, timeout_sec=0.1)
                
                # --- TRASH LOGIC ---
                if trash_detected and trash_pose is not None:
                    print(">>> TRASH DETECTED! Canceling patrol...")
                    nav.cancelTask()

                    # A. Get Current Robot Position (for math)
                    if nav.getFeedback():
                        current_robot_pose = nav.getFeedback().current_pose
                    else:
                        current_robot_pose = initial_pose 

                    # B. Calculate Safe Pose (0.5m away)
                    safe_pose = get_approach_pose(current_robot_pose, trash_pose, distance_offset=0.65)

                    # VISUALIZE: Show the exact target in RViz (Red Sphere)
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose = safe_pose.pose # <--- DIRECTLY USE TRASH POSE
                    marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
                    marker.color.a = 1.0; marker.color.r = 1.0 # Red
                    marker_pub.publish(marker)

                    print(f"Approaching EXACT trash coords: ({safe_pose.pose.position.x:.2f}, {safe_pose.pose.position.y:.2f})")
                    
                    # 1. GO DIRECTLY TO TRASH
                    nav.goToPose(safe_pose)
                    
                    # Wait for arrival
                    while not nav.isTaskComplete():
                        pass

                    # 2. Check Result
                    result = nav.getResult()
                    if result == TaskResult.SUCCEEDED:
                        print(">>> Arrived at trash. Telling Grasp Code to start...")
                        
                        if not grasp_client.wait_for_service(timeout_sec=2.0):
                            print("Warning: Grasp Service not available!")
                        else:
                            req = Trigger.Request()
                            future = grasp_client.call_async(req)
                            rclpy.spin_until_future_complete(node, future)
                            res = future.result()
                            print(f"Grasp Result: {res.message}")
                    else:
                        print(f"!!! FAILED to reach trash! Result: {result}")
                        print("Check RViz: Is the Red Sphere inside a wall or obstacle?")

                    # 3. Resume Patrol
                    print(">>> Resume Patrol...")
                    trash_detected = False
                    trash_pose = None
                    nav.goToPose(target_pose)

            # Check waypoint result
            if nav.getResult() == TaskResult.SUCCEEDED:
                print(f"Waypoint {i+1} reached.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()