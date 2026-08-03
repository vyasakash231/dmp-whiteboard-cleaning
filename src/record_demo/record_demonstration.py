#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from pose_transform import eul2quat


class DoosanRecord:
    def __init__(self):
        rospy.init_node('manual_control_node')
        rospy.on_shutdown(self.shutdown)
        
        # Wait for essential services
        rospy.wait_for_service('/dsr01a0509/system/set_robot_mode')
        rospy.wait_for_service('/dsr01a0509/force/task_compliance_ctrl')
        # rospy.wait_for_service('/dsr01a0509/force/set_stiffnessx')
        rospy.wait_for_service('/dsr01a0509/force/release_compliance_ctrl')
        
        # Create service proxies
        self.set_robot_mode = rospy.ServiceProxy('/dsr01a0509/system/set_robot_mode', SetRobotMode)
        self.task_compliance_ctrl = rospy.ServiceProxy('/dsr01a0509/force/task_compliance_ctrl', TaskComplianceCtrl)
        self.set_stiffness = rospy.ServiceProxy('/dsr01a0509/force/set_stiffnessx', SetStiffnessx)
        self.release_compliance = rospy.ServiceProxy('/dsr01a0509/force/release_compliance_ctrl', ReleaseComplianceCtrl)
        
        # Publishers
        self.stop_pub = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)
        
        # Subscribers
        self.state_sub = rospy.Subscriber('/dsr01a0509/state', RobotState, self.state_callback)

        # Set robot to manual mode
        self.set_robot_mode(0)  # 0 : ROBOT_MODE_MANUAL, (robot LED lights up blue) --> use it for recording demonstration

        # Get button state
        self.get_buttons_service = rospy.ServiceProxy('/dsr01a0509/system/get_buttons_state', GetButtonsState)
        
        self.gripper_close_width = 0
        self.gripper_open_width = 0.0
        self.gripper_sensitivity = 0.0

        # Default is [500, 500, 500, 100, 100, 100] -> Reducing these values will make the robot more compliant   
        self.stiffness = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        self.compliance = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    @property
    def button(self):
        return self.get_buttons_service().state[0]   # 0th button is used demo based learning
    
    @property
    def current_velocity(self):
        X_dot = np.zeros(6)
        X_dot[:3] = 0.001 * self.current_vel[:3]   # convert from mm/s to m/s
        X_dot[3:] = 0.0174532925 * self.current_vel[3:]  # convert from deg/s to rad/s  
        return X_dot
    
    def state_callback(self, msg):
        """Store complete info of robot and joint angle as current_posj in degrees"""
        self.q = 0.0174532925 * np.array(msg.current_posj)   # convert from deg to rad
        self.q_dot = 0.0174532925 * np.array(msg.current_velj)   # convert from deg/s to rad/s
        self.current_position = 0.001 * np.array(msg.current_posx)[:3]   # (x, y, z), converted from mm to m
        self.current_euler = 0.0174532925 * np.array(msg.current_posx)[3:]   # (a, b, c) follows Euler ZYZ notation, convert from deg to rad
        # self.current_quat = eul2quat(np.array(msg.current_posx)[3:])   # orientation in quaternion
        self.current_quat = Rotation.from_euler("ZYZ", np.array(msg.current_posx)[3:], degrees=True).as_quat()   # orientation in quaternion
        self.current_R = Rotation.from_euler("ZYZ", np.array(msg.current_posx)[3:], degrees=True).as_matrix()   # orientation in rotation matrix
        self.current_linear_vel = 0.001 * np.array(msg.current_velx)[:3]   # (Vx, Vy, Vz), convert from mm/s to m/s
        self.current_angular_vel = 0.0174532925 * np.array(msg.current_velx)[3:]   # (ωx, ωy, ωz), convert from deg/s to rad/s

    def time_based_resampling(self, demo_data, demo_times, target_length):
        """
        Resample based on time - most accurate for preserving temporal dynamics
        
        Args:
            demo_data: List of trajectories (each trajectory shape: (N_i, feature_dim))
            demo_times: List of time arrays for each demo
        
        Returns:
            resampled_data: List of trajectories with equal length
        """
        resampled_demos = []
        
        for i, (traj, times) in enumerate(zip(demo_data, demo_times)):
            if len(traj) < 2:
                continue
                
            # Ensure times is 1D array
            times = np.array(times).flatten()
            
            # Create uniform time vector
            t_start, t_end = times[0], times[-1]
            t_uniform = np.linspace(t_start, t_end, target_length)
            
            # Interpolate each dimension
            resampled_traj = np.zeros((target_length, traj.shape[1]))
            
            for dim in range(traj.shape[1]):
                # Use cubic interpolation if enough points, otherwise linear
                kind = 'cubic' if len(traj) >= 4 else 'linear'
                f = interp1d(times, traj[:, dim], kind=kind, bounds_error=False, fill_value='extrapolate')
                # Ensure the interpolated result is 1D
                interpolated_values = f(t_uniform)
                resampled_traj[:, dim] = np.array(interpolated_values).flatten()
                
            resampled_demos.append(resampled_traj)
        return resampled_demos
    
    def resample_demonstrations(self, target_length=100):
        """Resample all demonstrations to have equal number of points"""
        # Store original time data before resampling other data
        original_demo_time = self.demo_time.copy()

        self.demo_q = self.time_based_resampling(self.demo_q, self.demo_time, target_length)
        self.demo_q_dot = self.time_based_resampling(self.demo_q_dot, self.demo_time, target_length)
        self.demo_trajectory = self.time_based_resampling(self.demo_trajectory, self.demo_time, target_length)
        self.demo_orientation = self.time_based_resampling(self.demo_orientation, self.demo_time, target_length)
        self.demo_linear_velocity = self.time_based_resampling(self.demo_linear_velocity, self.demo_time, target_length)
        self.demo_angular_velocity = self.time_based_resampling(self.demo_angular_velocity, self.demo_time, target_length)

        # Now resample time itself to uniform length
        resampled_times = []
        for i, times in enumerate(original_demo_time):
            times_flat = np.array(times).flatten()
            if len(times_flat) < 2:
                rospy.logwarn(f"Demo {i+1} has insufficient time points: {len(times_flat)}")
                continue
            t_start, t_end = times_flat[0], times_flat[-1]
            t_uniform = np.linspace(t_start, t_end, target_length)
            resampled_times.append(t_uniform)
        
        self.demo_time = resampled_times
        
        # Convert to numpy arrays - use list comprehension for safety
        self.demo_q = np.array([demo for demo in self.demo_q])
        self.demo_q_dot = np.array([demo for demo in self.demo_q_dot])
        self.demo_trajectory = np.array([demo for demo in self.demo_trajectory])
        self.demo_orientation = np.array([demo for demo in self.demo_orientation])
        self.demo_linear_velocity = np.array([demo for demo in self.demo_linear_velocity])
        self.demo_angular_velocity = np.array([demo for demo in self.demo_angular_velocity])
        self.demo_time = np.array(self.demo_time)

        print("\n")
        rospy.loginfo(f"Resampled to {target_length} points each. Final shapes:")
        rospy.loginfo(f"Demo q: {self.demo_q.shape}")
        rospy.loginfo(f"Demo q_dot: {self.demo_q_dot.shape}")
        rospy.loginfo(f"Demo trajectories: {self.demo_trajectory.shape}")
        rospy.loginfo(f"Demo orientation: {self.demo_orientation.shape}")
        rospy.loginfo(f"Demo linear velocity: {self.demo_linear_velocity.shape}")
        rospy.loginfo(f"Demo angular velocity: {self.demo_angular_velocity.shape}")
        rospy.loginfo(f"Demo time: {self.demo_time.shape} \n")

    def free_move(self):
        self.set_stiffness(self.stiffness, 0, 0.0) 
        rospy.sleep(0.1)
        self.task_compliance_ctrl(self.compliance, 0, 0.0)  # time=0 for immediate effect
        rospy.loginfo(f"Compliance control enabled successfully \n")        

    def traj_record(self, trigger=0.001):  # trigger is 5mm     
        init_pose = self.current_position
        robot_perturbation = 0
        rospy.loginfo(f"Move robot to start recording.")

        # TO increase the amount of data collected, increase the frequency
        self.data_collection_freq = 60
        rate = rospy.Rate(self.data_collection_freq)   # 25Hz = 40ms, 50Hz = 20ms
        
        # observe small movement to start recoding
        while robot_perturbation < trigger:
            robot_perturbation = np.sqrt((self.current_position[0] - init_pose[0])**2 + (self.current_position[1] - init_pose[1])**2 + (self.current_position[2] - init_pose[2])**2)
        
        # At initialization
        self.recorded_q = self.q  # in rad
        self.recorded_q_dot = self.q_dot  # in rad/s
        self.recorded_trajectory = self.current_position  # in m
        # self.recorded_orientation = self.current_quat  # quaternions
        self.recorded_orientation = self.current_R[np.newaxis, :, :]  # Rotations
        self.recorded_linear_velocity = self.current_linear_vel  # in m/s
        self.recorded_angular_velocity = self.current_angular_vel  # in rad/s

        self.start_time = time.time()  # Record start time once
        self.recorded_time = np.array([0.0])  # Initialize with relative time 0

        # self.recorded_gripper = self.gripper_open_width

        print(self.current_R)
   
        while self.button:  # if the cockpit button is pressed
            # if self.gripper_width < (self.gripper_open_width - self.gripper_sensitivity):
            #     print("Close gripper")
            #     self.grip_value = 0   # Close the gripper
            # else:
            #     print("Open gripper")
            #     self.grip_value = self.gripper_open_width   # Open the gripper

            print(self.recorded_orientation.shape)
           
            self.recorded_q = np.vstack((self.recorded_q, self.q))  # shape: (N, 6) 
            self.recorded_q_dot = np.vstack((self.recorded_q_dot, self.q_dot))  # shape: (N, 6) 
            self.recorded_trajectory = np.vstack((self.recorded_trajectory, self.current_position))  # shape: (N, 3) 
            # self.recorded_orientation = np.vstack((self.recorded_orientation, self.current_quat))  # shape: (N, 4)
            self.recorded_orientation = np.concatenate((self.recorded_orientation, self.current_R[np.newaxis, :, :]), axis=0)  # shape: (N, 3, 3)
            self.recorded_linear_velocity = np.vstack((self.recorded_linear_velocity, self.current_linear_vel))  # shape: (N, 3)
            self.recorded_angular_velocity = np.vstack((self.recorded_angular_velocity, self.current_angular_vel))  # shape: (N, 3)
            
            # Record relative time since start
            current_time = time.time()
            relative_time = current_time - self.start_time
            self.recorded_time = np.vstack((self.recorded_time, relative_time))

            # self.recorded_gripper = np.vstack((self.recorded_gripper, self.grip_value))
            
            rate.sleep()

        rospy.loginfo("Ending trajectory recording.")

    def multiple_traject_record(self, no_of_demos=1.0):
        self.free_move()
        
        self.demo_q = []
        self.demo_q_dot = []
        self.demo_trajectory = []
        self.demo_orientation = []
        self.demo_linear_velocity = []
        self.demo_angular_velocity = []
        self.demo_time = []
        
        for i in range(no_of_demos):
            self.traj_record()  # record ith demo

            # since all the trajectories can have different no of sample points based the speed of demo, we'll store them as list not tensor
            self.demo_q.append(self.recorded_q)
            self.demo_q_dot.append(self.recorded_q_dot)
            self.demo_trajectory.append(self.recorded_trajectory)
            self.demo_orientation.append(self.recorded_orientation)
            self.demo_linear_velocity.append(self.recorded_linear_velocity)
            self.demo_angular_velocity.append(self.recorded_angular_velocity)
            self.demo_time.append(self.recorded_time)

            rospy.loginfo(f"Stored {i+1}th trajectory, reset robot state.")

            """reset state manually"""
            # while not self.button: 
            #     # wait while cockpit button is not pressed
            #     rospy.sleep(0.05)

            # while self.button:
            #     # wait while cockpit button is pressed
            #     rospy.sleep(0.05)

            """reset state automatically"""
            rospy.sleep(0.5)
            p1= posj(0,0,90,0,90,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
            movej(p1, vel=40, acc=20)
            rospy.sleep(0.5)

    def shutdown(self):
        """Cleanup when shutting down"""
        try:
            self.release_compliance()  # Release compliance control
            self.set_robot_mode(1)  # 1 : ROBOT_MODE_AUTONOMOUS, this will stop teaching mode (robot LED lights up in white)
            self.stop_pub.publish(stop_mode=STOP_TYPE_SLOW)  # Quick stop
            rospy.loginfo("Robot shutdown complete.")
        except Exception as e:
            rospy.logerr(f"Error during shutdown: {e}")
        return

    def save(self, name='demo'):
        curr_dir=os.getcwd()
        np.savez(curr_dir+ '/data/' + str(name) + '.npz',
                freq=self.data_collection_freq,
                q=self.demo_q,
                q_dot=self.demo_q_dot,
                traj=self.demo_trajectory,
                ori=self.demo_orientation,
                vel=self.demo_linear_velocity,
                omega=self.demo_angular_velocity,
                time=self.demo_time,
                #  grip=self.demo_gripper
                )
        rospy.loginfo(f"Successfully saved demonstration data to {name}.npz")

    def load(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.demo_q=data['q'],
        self.demo_trajectory = data['traj']
        self.demo_orientation = data['ori']
        self.demo_linear_velocity = data['vel']
        self.demo_angular_velocity = data['omega']
        # self.recorded_gripper = data['grip']

if __name__ == "__main__":
    # move to initial position first
    """
    for rhythmic motion, start position: 0,25,110,0,45,0
    for discrete motion, start position: 0,0,90,0,90,0
    """
    # p1= posj(0,15,120,0,-45,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
    # p1= posj(0,0,90,0,90,0)  # posj(q1, q2, q3, q4, q5, q6) This function designates the joint space angle in degrees
    p1 = np.rad2deg([0, -0.523598776, 1.5708, 0, 0.523599, 0]).tolist()
    movej(p1, vel=40, acc=20)
    
    time.sleep(1.0)

    try:
        controller = DoosanRecord()
        rospy.loginfo("Robot setup complete - Ready for manual demonstration")
        controller.multiple_traject_record(no_of_demos=1)
        # controller.resample_demonstrations(target_length=500)  # Resample to equal lengths
        controller.save(name="demo_for_dual_arm_paper")  # update file name before start recording
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
