#!/usr/bin/env python3
"""
Grasp Trash Service Server
Implements /grasp_trash service to grasp trash detected by vision system.
Uses MoveIt for arm control and publishes gripper commands.

Includes joint utilities and move_to_joints functionality.
"""

import rclpy
import time
import math
import threading
import subprocess
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
from gazebo_msgs.srv import SetEntityState, DeleteEntity


# ============================================================================
# JOINT UTILITIES (from joint_utils.py)
# ============================================================================

# TIAGo arm joint names (arm_torso group)
ARM_JOINT_NAMES = [
    "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
    "arm_5_joint", "arm_6_joint", "arm_7_joint", "torso_lift_joint"
]

# FIXED ARM JOINT VALUES (in radians)
# These values are fixed - only torso_lift_joint changes
ARM_1_JOINT = math.radians(90)    
ARM_2_JOINT = math.radians(-38)   
ARM_3_JOINT = math.radians(0)   
ARM_4_JOINT = math.radians(0)     
ARM_5_JOINT = math.radians(-90)    
ARM_6_JOINT = math.radians(45)   
ARM_7_JOINT = math.radians(-0)    

# Initial torso_lift_joint position (in meters)
INITIAL_TORSO_LIFT = 0.7

# Base configuration with fixed arm joints and initial torso
BASE_JOINT_CONFIG = [
    ARM_1_JOINT,
    ARM_2_JOINT,
    ARM_3_JOINT,
    ARM_4_JOINT,
    ARM_5_JOINT,
    ARM_6_JOINT,
    ARM_7_JOINT,
    INITIAL_TORSO_LIFT
]

def get_joint_config(torso_lift_value):
    """
    Get joint configuration with fixed arm joints and specified torso_lift_joint
    
    Args:
        torso_lift_value: Torso lift joint value in meters
    
    Returns:
        List of joint positions [arm_1, arm_2, ..., arm_7, torso_lift]
    """
    return [
        ARM_1_JOINT,
        ARM_2_JOINT,
        ARM_3_JOINT,
        ARM_4_JOINT,
        ARM_5_JOINT,
        ARM_6_JOINT,
        ARM_7_JOINT,
        torso_lift_value
    ]

# Joint configurations for different positions
# Only torso_lift_joint changes - arm joints stay fixed
PRE_GRASP_JOINTS = get_joint_config(0.4)   # Higher torso for pre-grasp
GRASP_JOINTS = get_joint_config(0.3)      # Lower torso to reach floor



# ============================================================================
# MOVE TO JOINTS (from move_to_joints.py)
# ============================================================================

def move_arm_to_joints(node, move_group_client, joint_positions, joint_names, tolerance=0.05):
    """
    Move arm to target joint positions using MoveGroup action
    
    Args:
        node: ROS 2 node
        move_group_client: ActionClient for MoveGroup
        joint_positions: List of joint positions (values)
        joint_names: List of joint names
        tolerance: Joint position tolerance (increased default for better success)
    
    Returns:
        bool: True if successful
    """
    # Log target positions
    node.get_logger().info(f'Target joint positions:')
    for i, (name, pos) in enumerate(zip(joint_names, joint_positions)):
        if 'torso' in name:
            node.get_logger().info(f'  {name}: {pos:.3f}m')
        else:
            node.get_logger().info(f'  {name}: {math.degrees(pos):.1f}° ({pos:.3f}rad)')
    
    goal_msg = MoveGroup.Goal()
    goal_msg.request.workspace_parameters.header.frame_id = "base_footprint"
    goal_msg.request.workspace_parameters.header.stamp = node.get_clock().now().to_msg()
    
    # Create joint constraints
    constraints = Constraints()
    for i, joint_name in enumerate(joint_names):
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = joint_name
        joint_constraint.position = joint_positions[i]
        joint_constraint.tolerance_above = tolerance
        joint_constraint.tolerance_below = tolerance
        joint_constraint.weight = 1.0
        constraints.joint_constraints.append(joint_constraint)
    
    goal_msg.request.goal_constraints.append(constraints)
    goal_msg.request.group_name = "arm_torso"
    goal_msg.request.num_planning_attempts = 5
    goal_msg.request.allowed_planning_time = 5.0
    
    goal_msg.request.path_constraints = Constraints()
    
    node.get_logger().info('Sending goal to MoveGroup...')
    send_goal_future = move_group_client.send_goal_async(goal_msg)
    rclpy.spin_until_future_complete(node, send_goal_future, timeout_sec=5.0)
    goal_handle = send_goal_future.result()
    
    if goal_handle is None:
        node.get_logger().error('Goal handle is None - action server may not be responding')
        return False
    
    if goal_handle.accepted:
        node.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=30.0)
        result = result_future.result()
        
        if result is None:
            node.get_logger().error('Result is None - planning may have timed out')
            return False
        
        result_msg = result.result
        error_code = result_msg.error_code.val
        
        # Error code meanings: 1=SUCCESS, others are failures
        if error_code == 1:
            node.get_logger().info('Movement successful!')
            return True
        else:
            node.get_logger().warn(f'Movement failed with error code: {error_code}')
            return False
    else:
        node.get_logger().error('Goal was rejected by MoveGroup')
        return False


# ============================================================================
# GRASP TRASH SERVER
# ============================================================================

class GraspTrashServer(Node):
    def __init__(self):
        super().__init__('grasp_trash_server')
        
        # Create the services
        self.srv = self.create_service(Trigger, '/grasp_trash', self.grasp_callback)
        self.release_srv = self.create_service(Trigger, '/release_trash', self.release_callback)
        
        # Subscribe to get object name
        self.create_subscription(String, '/grasp_target', self.name_callback, 10)
        
        # MoveIt action client
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        
        self.target_object_name = "pringles"  # Default, can be overridden by /grasp_target topic
        self.held_object_name = None
        self.holding_thread = None
        self.stop_holding = threading.Event()
        
        self.get_logger().info('Grasp Trash Service Ready! Waiting for requests...')
        
        # Wait for MoveGroup server
        if self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.moveit_available = True
            self.get_logger().info('MoveGroup server ready!')
            
            # Wait for system to fully initialize before moving arm
            self.get_logger().info('Waiting for system to fully initialize...')
            time.sleep(2.0)
            
            # Initialize arm to base position at startup (with retry)
            self.get_logger().info('Initializing arm to base position...')
            success = False
            for attempt in range(4):
                if move_arm_to_joints(self, self.move_group_client, BASE_JOINT_CONFIG, ARM_JOINT_NAMES):
                    self.get_logger().info('Arm initialized to base position!')
                    success = True
                    break
                else:
                    self.get_logger().warn(f'Attempt {attempt + 1}/2 failed - retrying...')
                    time.sleep(1.0)
            
            if not success:
                self.get_logger().warn('Failed to initialize arm after 3 attempts - will try again on first grasp request')
        else:
            self.moveit_available = False
            self.get_logger().warn('MoveGroup server not available - grasp will fail')
    
    def name_callback(self, msg):
        """Callback to receive object name from /grasp_target topic"""
        self.target_object_name = msg.data
        self.get_logger().info(f'Received object name from /grasp_target: {self.target_object_name}')
    
    def open_gripper(self):
        """Open the gripper using bash command"""
        self.get_logger().info('Opening gripper...')
        # Stop holding object when opening gripper
        self.stop_holding_object()
        
        yaml_content = """header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names:
  - 'gripper_left_finger_joint'
  - 'gripper_right_finger_joint'
points:
  - positions: [0.05, 0.05]
    velocities: []
    accelerations: []
    effort: []
    time_from_start:
      sec: 2
      nanosec: 0"""
        
        cmd = [
            'ros2', 'topic', 'pub', '--once',
            '/gripper_controller/joint_trajectory',
            'trajectory_msgs/msg/JointTrajectory',
            yaml_content
        ]
        subprocess.run(cmd, check=False)
        time.sleep(2.5)  # Wait for gripper to open
    
    def close_gripper(self):
        """Close the gripper to grasp object using bash command"""
        self.get_logger().info('Closing gripper...')
        
        yaml_content = """header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names:
  - 'gripper_left_finger_joint'
  - 'gripper_right_finger_joint'
points:
  - positions: [0.02, 0.02]
    velocities: []
    accelerations: []
    effort: []
    time_from_start:
      sec: 2
      nanosec: 0"""
        
        cmd = [
            'ros2', 'topic', 'pub', '--once',
            '/gripper_controller/joint_trajectory',
            'trajectory_msgs/msg/JointTrajectory',
            yaml_content
        ]
        subprocess.run(cmd, check=False)
        time.sleep(2.5)  # Wait for gripper to close
        # Start holding object when gripper is closed
        if self.target_object_name:
            self.get_logger().info(f'Starting to hold object: {self.target_object_name}')
            self.start_holding_object(self.target_object_name)
        else:
            self.get_logger().warn('No target object name set - object will not be held!')

    
    def holding_loop(self):
        """Continuously teleport the object to follow the robot's gripper"""
        # Create client inside the loop thread
        client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # Wait for service to be available
        self.get_logger().info('Waiting for /gazebo/set_entity_state service...')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('SetEntityState service not available!')
            return
        self.get_logger().info('SetEntityState service available. Starting teleport loop...')
        
        teleport_count = 0
        while rclpy.ok() and not self.stop_holding.is_set():
            if self.held_object_name:
                # 1. Create a request to teleport the object
                req = SetEntityState.Request()
                req.state.name = self.held_object_name
                
                # 2. Set the destination to match the gripper EXACTLY
                # We use 'reference_frame' so Gazebo handles the math for us!
                req.state.reference_frame = "wrist_ft_link"
                
                # 3. Position relative to wrist (in the palm)
                req.state.pose.position.x = 0.15  # Adjust forward to be in fingers
                req.state.pose.position.y = 0.0
                req.state.pose.position.z = 0.0
                req.state.pose.orientation.w = 1.0
                
                # 4. Set velocity to zero (prevent falling due to gravity)
                req.state.twist.linear.x = 0.0
                req.state.twist.linear.y = 0.0
                req.state.twist.linear.z = 0.0
                req.state.twist.angular.x = 0.0
                req.state.twist.angular.y = 0.0
                req.state.twist.angular.z = 0.0
                
                # 5. Teleport it!
                future = client.call_async(req)
                
                teleport_count += 1
                if teleport_count == 1:
                    self.get_logger().info(f'First teleport of "{self.held_object_name}" - object picked up from ground!')
                elif teleport_count % 100 == 0:  # Log every 5 seconds (100 * 0.05s)
                    self.get_logger().info(f'Teleporting "{self.held_object_name}" - {teleport_count} updates')
                
            time.sleep(0.05)  # Update 20 times a second
        
        self.get_logger().info(f'Teleport loop ended after {teleport_count} updates')
    
    def start_holding_object(self, object_name):
        """Start teleporting the object to follow the gripper"""
        self.held_object_name = object_name
        self.stop_holding.clear()
        self.holding_thread = threading.Thread(target=self.holding_loop, daemon=True)
        self.holding_thread.start()
        self.get_logger().info(f'Started holding object "{object_name}" - teleporting to follow gripper')
    
    def delete_model_from_gazebo(self, model_name):
        """Delete a model from Gazebo simulation"""
        self.get_logger().info(f'Deleting model "{model_name}" from Gazebo...')
        
        delete_client = self.create_client(DeleteEntity, '/delete_entity')
        
        if delete_client.wait_for_service(timeout_sec=2.0):
            del_req = DeleteEntity.Request()
            del_req.name = model_name
            
            future = delete_client.call_async(del_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.result() is not None:
                del_resp = future.result()
                if del_resp.success:
                    self.get_logger().info(f'>>> SUCCESS: Model "{model_name}" deleted from Gazebo.')
                else:
                    self.get_logger().warn(f'!!! DELETE FAILED: {del_resp.status_message}')
            else:
                self.get_logger().warn('Delete request timed out')
        else:
            self.get_logger().warn('Delete service not available')
    
    def stop_holding_object(self):
        """Stop teleporting the object and delete it from Gazebo"""
        if self.held_object_name:
            object_to_delete = self.held_object_name
            self.get_logger().info(f'Stopped holding object "{object_to_delete}"')
            self.stop_holding.set()
            self.held_object_name = None
            if self.holding_thread:
                self.holding_thread.join(timeout=1.0)
            
            # Delete the model from Gazebo (trash disposed)
            self.delete_model_from_gazebo(object_to_delete)
    
    def grasp_callback(self, request, response):
        """Service callback to perform grasping sequence"""
        self.get_logger().info('=' * 50)
        self.get_logger().info('[Grasp Service] Request received! Starting grasp sequence...')

        # Step 1: Move to pre-grasp position 
        self.get_logger().info('Step 1: Moving to pre-grasp position...')
        move_arm_to_joints(self, self.move_group_client, PRE_GRASP_JOINTS, ARM_JOINT_NAMES)
        time.sleep(1.0)
        
        # Step 2: Open gripper
        self.get_logger().info('Step 2: Opening gripper...')
        self.open_gripper()
        
        # Step 3: Move to grasp position (torso lower to reach floor)
        self.get_logger().info('Step 3: Moving to grasp position...')
        grasp_config = GRASP_JOINTS.copy()
        grasp_config[7] = 0.05  # Set torso to reach object
        move_arm_to_joints(self, self.move_group_client, grasp_config, ARM_JOINT_NAMES)
        time.sleep(1.0)
        
        # Step 4: Close gripper to grasp object
        self.get_logger().info('Step 4: Closing gripper to grasp object...')
        self.close_gripper()
        time.sleep(1.0)
        
        # Step 5: Return arm to initial position (object is already being held via teleportation)
        self.get_logger().info('Step 5: Returning arm to initial position...')
        move_arm_to_joints(self, self.move_group_client, BASE_JOINT_CONFIG, ARM_JOINT_NAMES)
        self.get_logger().info('Arm returned to initial position successfully!')
        time.sleep(1.0)
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('[Grasp Service] Grasp sequence completed!')
        self.get_logger().info('=' * 50)
        
        response.success = True
        response.message = "Trash collected successfully!"
        return response
    
    def release_callback(self, request, response):
        """Service callback to release the held object (drop trash at bin)"""
        self.get_logger().info('=' * 50)
        self.get_logger().info('[Release Service] Request received! Releasing object...')
        
        # Open gripper to release object (this also stops holding/teleporting)
        self.open_gripper()
        
        self.get_logger().info('[Release Service] Object released!')
        self.get_logger().info('=' * 50)
        
        response.success = True
        response.message = "Trash released successfully!"
        return response


def main():
    rclpy.init()
    node = GraspTrashServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
