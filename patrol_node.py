#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_srvs.srv import Trigger
from gazebo_msgs.srv import DeleteEntity
from visualization_msgs.msg import Marker
from tf_transformations import quaternion_from_euler
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String
import time

# --- GLOBAL VARIABLES ---
trash_pose = None
trash_detected = False
tf_buffer = None
tf_listener = None
latest_model_states = None

def model_states_callback(msg):
    global latest_model_states
    latest_model_states = msg

def get_specific_model_name(target_x, target_y):
    global latest_model_states
    if latest_model_states is None:
        return None

    closest_name = None
    min_dist = 9999.0

    # Loop through every object in the simulation
    for i, name in enumerate(latest_model_states.name):            
        # 2. Calculate distance to our target point
        obj_x = latest_model_states.pose[i].position.x + 5
        obj_y = latest_model_states.pose[i].position.y - 6
        
        dist = math.sqrt((obj_x - target_x)**2 + (obj_y - target_y)**2)
        
        # 3. Find the minimum
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name

def trash_callback(msg):
    global trash_pose, trash_detected, tf_buffer
    
    if not trash_detected:
        # 1. Strip leading slashes to handle '/map' and 'map' the same way
        incoming_frame = msg.header.frame_id.strip('/')
        
        print(f"\n!!! TRASH RECEIVED !!! Frame ID: '{incoming_frame}'")
        
        # 2. CHECK: If it is ALREADY 'map', accept it immediately.
        if incoming_frame == 'map':
            trash_pose = msg
            trash_detected = True
            print(f"   [Direct Map] Accepted: x={trash_pose.pose.position.x:.2f}, y={trash_pose.pose.position.y:.2f}")
            return

        # 3. OTHERWISE: Try to convert (e.g., from Camera)
        try:
            # Use Time() with seconds=0 to get the "Latest Available" transform
            if tf_buffer.can_transform('map', msg.header.frame_id, rclpy.time.Time()):
                trash_pose = tf_buffer.transform(msg, 'map')
                trash_detected = True
                print(f"   [Transformed] {incoming_frame} -> Map: x={trash_pose.pose.position.x:.2f}, y={trash_pose.pose.position.y:.2f}")
            else:
                print(f"   WARNING: TF Buffer cannot transform '{incoming_frame}' to 'map' yet.")
                print("            (Wait 5 seconds for the robot to wake up and try again)")
        except Exception as e:
            print(f"   Transformation Error: {e}")

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
    global tf_buffer, tf_listener, trash_detected, trash_pose  # <--- FIX: Ensure we use global vars

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    node.create_subscription(ModelStates, '/gazebo/model_states', model_states_callback, 10)
    node.create_subscription(PoseStamped, '/trash_pose', trash_callback, 10)
    grasp_client = node.create_client(Trigger, '/grasp_trash')
    delete_client = node.create_client(DeleteEntity, '/delete_entity')
    marker_pub = node.create_publisher(Marker, 'approach_marker', 10)
    grasp_name_pub = node.create_publisher(String, '/grasp_target', 10)

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

    last_trash_can_pose = initial_pose

    # 4. Main Loop
    while True:
        for i, target_pose in enumerate(goal_poses):
            print(f"--> Headed to Waypoint {i+1}")
            nav.goToPose(target_pose)

            # WHILE MOVING
            while not nav.isTaskComplete():
                for _ in range(10):
                    rclpy.spin_once(node, timeout_sec=0.01)
                
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
                    
                    nav_success = False
                    
                    for attempt in range(1, 4): # Try 1, 2, 3
                        print(f"   [Attempt {attempt}/3] Navigating to trash...")
                        
                        nav.goToPose(safe_pose)
                        
                        # Wait for arrival
                        while not nav.isTaskComplete():
                            pass

                        # Check Result
                        result = nav.getResult()
                        
                        if result == TaskResult.SUCCEEDED:
                            print("   >>> NAVIGATION SUCCESS!")
                            nav_success = True
                            break # Exit the retry loop
                        else:
                            print(f"   !!! Navigation Failed (Result: {result}). Retrying in 1s...")
                            time.sleep(0.5) # Give the costmap time to clear/update

                    # 2. PROCEED ONLY IF SUCCESSFUL
                    if nav_success:
                        detected_trash_name = get_specific_model_name(trash_pose.pose.position.x, trash_pose.pose.position.y)
                        print(f">>> Arrived at {detected_trash_name} trash. Telling Grasp Code to start...")
                        
                        name_msg = String()
                        name_msg.data = detected_trash_name
                        grasp_name_pub.publish(name_msg)

                        if grasp_client.wait_for_service(timeout_sec=2.0):
                            req = Trigger.Request()
                            future = grasp_client.call_async(req)
                            rclpy.spin_until_future_complete(node, future)
                            
                            # 3. ALWAYS RETURN TO BIN (Regardless of result)
                            print(f"   Navigating back to Bin at ({last_trash_can_pose.pose.position.x:.1f}, {last_trash_can_pose.pose.position.y:.1f})...")
                            nav.goToPose(last_trash_can_pose)
                            while not nav.isTaskComplete(): pass

                            if nav.getResult() == TaskResult.SUCCEEDED:
                                print("   At Bin. Throwing away trash (Deleting)...")
                                
                                # 4. DELETE MODEL
                                del_req = DeleteEntity.Request()
                                del_req.name = detected_trash_name
                                
                                # Increase wait time to 2.0 seconds
                                if delete_client.wait_for_service(timeout_sec=2.0):
                                    del_future = delete_client.call_async(del_req)
                                    rclpy.spin_until_future_complete(node, del_future)
                                    
                                    del_resp = del_future.result()
                                    if del_resp.success:
                                        print("   >>> SUCCESS: Trash Disposed (Deleted).")
                                    else:
                                        # If it fails, Gazebo will tell us why (e.g., "Model not found")
                                        print(f"   !!! DELETE FAILED. Gazebo said: '{del_resp.status_message}'")
                            else:
                                print("   Failed to return to bin.")
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
                last_trash_can_pose = target_pose

    rclpy.shutdown()

if __name__ == '__main__':
    main()