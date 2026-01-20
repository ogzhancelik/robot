import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.parameter import Parameter

class EcoBotVision(Node):
    def __init__(self):
        super().__init__('yolo_trash_detector')
        
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        # 1. Setup Publisher (Using PoseStamped)
        self.trash_pub = self.create_publisher(PoseStamped, '/trash_pose', 10)
        self.head_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)
        
        # 2. Setup Subscribers
        # IMPORTANT: 'self.rgb_callback' is defined below. 
        # If indentation is wrong, this line crashes.
        self.subscription = self.create_subscription(
            Image, '/head_front_camera/rgb/image_raw', self.rgb_callback, 10)
            
        self.depth_sub = self.create_subscription(
            Image, '/head_front_camera/depth/image_raw', self.depth_callback, 10)
            
        self.bridge = CvBridge()
        # Initialize YOLO (Using CPU to avoid memory crash)
        self.model = YOLO('yolo26m.pt') 
        self.latest_depth_frame = None
        
        # Camera Intrinsics (TIAGo Default)
        self.fx, self.fy = 522.19, 522.19
        self.cx, self.cy = 320.5, 240.5

        # Timer to tilt head down
        self.create_timer(2.0, self.tilt_head_down)
        self.get_logger().info('EcoBot Vision Started (PoseStamped Mode)...')

    def depth_callback(self, data):
        """ Stores the latest depth image for distance calculation. """
        try:
            self.latest_depth_frame = self.bridge.imgmsg_to_cv2(data, '32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')

    def tilt_head_down(self):
        """ Moves the robot head to look at the floor. """
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, -0.5] # Look down
        point.time_from_start = Duration(sec=1, nanosec=0)
        msg.points.append(point)
        self.head_pub.publish(msg)

    def rgb_callback(self, data):
        """ Main Logic: Finds trash and publishes 3D location. """
        try:
            # 1. Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            
            # 2. Run YOLO AI
            # classes=[39, 41] are usually Bottle/Cup in COCO dataset
            results = self.model(frame, conf=0.25, classes=[39, 41], device='cpu', verbose=False)

            for r in results:
                for box in r.boxes:
                    # Get center of the bounding box
                    u = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    v = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    
                    # 3. Check Depth (Distance)
                    if self.latest_depth_frame is not None:
                        try:
                            # Safety check for image bounds
                            if v >= self.latest_depth_frame.shape[0] or u >= self.latest_depth_frame.shape[1]:
                                continue

                            depth = self.latest_depth_frame[v, u]
                            
                            if not (np.isnan(depth) or np.isinf(depth)):
                                z_m = float(depth)
                                x_m = (u - self.cx) * z_m / self.fx
                                y_m = (v - self.cy) * z_m / self.fy
                                
                                # 4. Publish PoseStamped
                                trash_msg = PoseStamped()
                                trash_msg.header.stamp = self.get_clock().now().to_msg()
                                # FRAME ID: Must match what we found in tf2_monitor earlier
                                trash_msg.header.frame_id = "head_front_camera_color_optical_frame"
                                
                                trash_msg.pose.position.x = x_m
                                trash_msg.pose.position.y = y_m
                                trash_msg.pose.position.z = z_m
                                trash_msg.pose.orientation.w = 1.0 

                                self.trash_pub.publish(trash_msg)
                                self.get_logger().info(f'Trash Found! Z={z_m:.2f}m')
                        except Exception as e:
                            pass # Ignore sporadic math errors

            # 5. Visualize
            annotated_frame = results[0].plot()
            cv2.imshow("EcoBot_YOLO_View", annotated_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Vision Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = EcoBotVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()