#!/usr/bin/env python3

# test file for check subscription (/symptom_text)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SymptomSubscriber(Node):
    def __init__(self):
        super().__init__('symptom_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/symptom_text',
            self.listener_callback,
            10
        )
        self.get_logger().info("symptom_subscriber 노드가 시작되었습니다.")

    def listener_callback(self, msg):
        self.get_logger().info(f"받은 메시지: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = SymptomSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
