#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from wrench_based_impedance_control import Wrench_CartesianImpedanceControl
from acceleration_based_impedance_control import Acceleration_CartesianImpedanceControl

# move to initial position first
"""
for rhythmic motion, start position: 0,25,110,0,45,0
for discrete motion, start position: 0,0,90,0,90,0
"""
p1= posj(0,0,0,0,0,0)
#p1= posj(0,0,90,0,90,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
movej(p1, vel=40, acc=20)

time.sleep(2.0)

"""Wrench based Impedance Control"""
try:
    # Initialize ROS node first
    rospy.init_node('My_service_node')
    
    # Create control object
    task = Wrench_CartesianImpedanceControl(file_name="demo_for_APM")
    rospy.sleep(2.0)  # Give time for initialization

    # Start controller in a separate thread
    # controller_thread = Thread(target=task.run_controller, args=(1250.0, 180.0)) # translation stiff -> N/m, rotational stiffness -> Nm/rad 

    controller_thread = Thread(target=task.run_controller, args=(2000.0, 450.0)) # translation stiff -> N/m, rotational stiffness -> Nm/rad 
    controller_thread.daemon = True
    controller_thread.start()
    
    # Keep the main thread running for the plot
    while not rospy.is_shutdown():
        rospy.sleep(0.01)

except rospy.ROSInterruptException:
    pass

finally:
    task.save(name="dmp_performed_for_APM")  # save data for plotting
    pass


# """Acceleration based Impedance Control"""
# try:
#     # Initialize ROS node first
#     rospy.init_node('My_service_node')
    
#     # Create control object
#     task = Acceleration_CartesianImpedanceControl(file_name="demo_discrete")
#     rospy.sleep(2.0)  # Give time for initialization

#     # Start controller in a separate thread   # 150.0, 15.0 for close trajectory
#     controller_thread = Thread(target=task.run_controller_2, args=(500.0, 250.0)) # translation stiff -> N/m, rotational stiffness -> Nm/rad 
#     controller_thread.daemon = True
#     controller_thread.start()
    
#     # Keep the main thread running for the plot
#     while not rospy.is_shutdown():
#         rospy.sleep(0.01)

# except rospy.ROSInterruptException:
#     pass

# finally:
#     task.save(name="task_performed_discrete")  # save data for plotting
#     pass
