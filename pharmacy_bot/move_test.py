#!/usr/bin/env python3

import rclpy
import DR_init
import time
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
OFF, ON = 0, 1         
VELOCITY, ACC = 20, 20
DR_VEL_=20
DR_ACC_=20
def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("pjt_test", namespace=ROBOT_ID)

    DR_init.__dsr__node = node
    

    try:
        from DSR_ROBOT2 import (
                set_tool,
                set_tcp,
                movej,
                movel,
                ikin,
                set_digital_output,
                get_current_posx,
                set_desired_force,
                task_compliance_ctrl,
                check_force_condition,
                release_force,
                release_compliance_ctrl,
                get_tool_force,
                drl_script_stop,
                wait,
                DR_BASE,
                DR_TOOL,
                DR_AXIS_Z,
                amove_periodic,
                trans,
        )
        set_tool("TCP208mm")
        set_tcp("GripperSA_rg2_250509")

        def grip():
            set_digital_output(1,1)
            wait(1.5)
            set_digital_output(1,0)

        def relealse():
            set_digital_output(2,1)
            wait(1.5)
            set_digital_output(2,0)



        from DR_common2 import posx, posj

    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        return

    JReady = [0, 0, 90, 0, 90, 0]
    home_pose=[-140.48, -34.46, 146, 95.15, 49.70, -19.84]  # 홈 위치 조인트값

    set_tool("Tool Weight_2FG")
    set_tcp("2FG_TCP")

    while rclpy.ok():

        print("movej to home")
        movej(home_pose, vel=VELOCITY, acc=ACC)
        # print("movej to sapo1")
        break
        

    rclpy.shutdown()

if __name__ == "__main__":

    main()

