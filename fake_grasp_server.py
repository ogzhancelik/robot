#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from std_srvs.srv import Trigger

class FakeGraspServer(Node):
    def __init__(self):
        super().__init__('fake_grasp_server')
        # Create the service your patrol node is looking for
        self.srv = self.create_service(Trigger, '/grasp_trash', self.grasp_callback)
        print("Fake Grasp Service Ready! Waiting for robot...")

    def grasp_callback(self, request, response):
        print("\n[Grasp Arm] Request received! Grasping trash...")
        
        # Simulate time taken to move the arm (e.g., 3 seconds)
        time.sleep(3)
        
        print("[Grasp Arm] Grasp complete. Returning success.")
        response.success = True
        response.message = "Trash collected successfully!"
        return response

def main():
    rclpy.init()
    node = FakeGraspServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()