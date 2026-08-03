#! /usr/bin/python3
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from wrench_based_impedance_control import Wrench_CartesianImpedanceControl
from acceleration_based_impedance_control import Acceleration_CartesianImpedanceControl
from learn_dmp import PositionDMP, QuaternionDMP

# move to initial position first
# p1= posj(0,25,110,0,45,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
# p1= posj(0,15,120,0,-45,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
p1= posj(0,0,90,0,90,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
movej(p1, vel=40, acc=20)

time.sleep(2.0)

try:
    # Initialize ROS node first
    rospy.init_node('My_service_node')
    
    # Create control object
    task = Wrench_CartesianImpedanceControl(file_name="dmp_discrete")
    rospy.sleep(2.0)  # Give time for initialization

    # Start controller in a separate thread
    controller_thread = Thread(target=task.run_dmp, args=(1250.0, 450.0)) # translation stiff -> N/m, rotational stiffness -> Nm/rad 
    controller_thread.daemon = True
    controller_thread.start()
    
    # Keep the main thread running for the plot
    while not rospy.is_shutdown():
        task.plotter.app.processEvents()
        rospy.sleep(0.01)

except rospy.ROSInterruptException:
    pass

finally:
    task.save(name="dmp_performed_discrete_with_wrench")  # save data for plotting
    plt.close('all')  # Clean up plots on exit
    pass

"""#################################################################################################################"""

# def load_demo(name='demo'):
#     curr_dir=os.getcwd()
#     data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
#     time = data['time']
#     q_demo = data['q']    # shape: (N, 6), in rad
#     position_demo = 1000 * data['traj']   # shape: (N, 3), convert from m to mm
#     orientation_demo = data['ori']   # shape: (N, 4)
#     linear_velocity_demo = data['vel']   # shape: (N, 3), in m/s
#     angular_velocity_demo = data['omega']   # shape: (N, 3), in rad/s
#     N = position_demo.shape[0]   # no of sample points
#     return time, position_demo, orientation_demo, linear_velocity_demo, angular_velocity_demo, N

# # load demo data
# time, position_demo, orientation_demo, linear_velocity_demo, angular_velocity_demo, N = load_demo(name="demo_discrete")

# # Calculate time step between trajectory points (in seconds)
# traj_dt = 0.015  # 10Hz = 0.1s between points

# # Start DMPS
# position_dmp = PositionDMP(no_of_DMPs=3, no_of_basis_func=25, T=5.0, dt=traj_dt, K=100.0, alpha=1.0)
# quaternion_dmp = QuaternionDMP(no_of_basis_func=25, T=5.0, dt=traj_dt, K=100.0, alpha=1.0)

# # learn Weights based on position Demo
# X_demo = 0.001 * position_demo.T   # demo position data of shape (3, N) in m
# V_demo = linear_velocity_demo.T   # demo velocity data of shape (3, N) in m
# position_dmp.learn_dynamics(time=time, X_des=X_demo, dX_des=V_demo)
# position_dmp.reset_state()

# # learn Weights based on orientation Demo
# Q_demo = orientation_demo.T   # orientation data of shape (4, N)
# omega_demo = angular_velocity_demo.T   # demo velocity data of shape (3, N) in m
# quaternion_dmp.learn_dynamics(time=time, q_des=Q_demo, omega_des=omega_demo)
# quaternion_dmp.reset_state()

# X_goal = X_demo[:,[-1]]
# Q_goal = Q_demo[:,[-1]]

# X_track, dX_track = position_dmp.rollout(X_goal)
# Q_track, omega_track = quaternion_dmp.rollout(Q_goal)

# plt.subplot(2, 2, 1)
# plt.plot(range(N), X_demo[0,:], "--s", color="black", markevery=15)
# plt.plot(range(N), X_demo[1,:], "--o", color="black", markevery=15)
# plt.plot(range(N), X_demo[2,:], "--*", color="black", markevery=15)
# plt.plot(range(X_track.shape[1]), X_track[0,:], color="red")
# plt.plot(range(X_track.shape[1]), X_track[1,:], color="blue")
# plt.plot(range(X_track.shape[1]), X_track[2,:], color="green")
# plt.legend(['$x_{demo}$','$y_{demo}$','$z_{demo}$','$x_{imitated}$','$y_{imitated}$','$z_{imitated}$'])
# plt.xlabel("time")
# plt.ylabel("position")

# plt.subplot(2, 2, 2)
# plt.plot(range(N), V_demo[0,:], "--s", color="black", markevery=15)
# plt.plot(range(N), V_demo[1,:], "--o", color="black", markevery=15)
# plt.plot(range(N), V_demo[2,:], "--*", color="black", markevery=15)
# plt.plot(range(dX_track.shape[1]), dX_track[0,:], color="red")
# plt.plot(range(dX_track.shape[1]), dX_track[1,:], color="blue")
# plt.plot(range(dX_track.shape[1]), dX_track[2,:], color="green")
# plt.legend(['$x_{demo}$','$y_{demo}$','$z_{demo}$','$x_{imitated}$','$y_{imitated}$','$z_{imitated}$'])
# plt.xlabel("time")
# plt.ylabel("velocity")

# plt.subplot(2, 2, 3)
# plt.plot(range(N), Q_demo[0,:], "--D", color="black", markevery=15)
# plt.plot(range(N), Q_demo[1,:], "--s", color="black", markevery=15)
# plt.plot(range(N), Q_demo[2,:], "--o", color="black", markevery=15)
# plt.plot(range(N), Q_demo[3,:], "--*", color="black", markevery=15)
# plt.plot(range(Q_track.shape[1]), Q_track[0,:], color="red")
# plt.plot(range(Q_track.shape[1]), Q_track[1,:], color="blue")
# plt.plot(range(Q_track.shape[1]), Q_track[2,:], color="green")
# plt.plot(range(Q_track.shape[1]), Q_track[3,:], color="magenta")
# plt.legend(['$q1_{demo}$','$q2_{demo}$','$q3_{demo}$','$q0_{demo}$','$q1_{imitated}$','$q2_{imitated}$','$q3_{imitated}$','$q0_{imitated}$'])
# plt.xlabel("time")
# plt.ylabel("quaternion")

# plt.subplot(2, 2, 4)
# plt.plot(range(N), omega_demo[0,:], "--s", color="black", markevery=15)
# plt.plot(range(N), omega_demo[1,:], "--o", color="black", markevery=15)
# plt.plot(range(N), omega_demo[2,:], "--*", color="black", markevery=15)
# plt.plot(range(omega_track.shape[1]), omega_track[0,:], color="red")
# plt.plot(range(omega_track.shape[1]), omega_track[1,:], color="blue")
# plt.plot(range(omega_track.shape[1]), omega_track[2,:], color="green")
# plt.legend(['$\omega x_{demo}$','$\omega y_{demo}$','$\omega z_{demo}$','$\omega x_{imitated}$','$\omega y_{imitated}$','$\omega z_{imitated}$'])
# plt.xlabel("time")
# plt.ylabel("angular velocity")

# plt.show()
