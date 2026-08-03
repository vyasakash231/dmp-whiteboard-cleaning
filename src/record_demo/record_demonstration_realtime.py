#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot
from pose_transform import eul2quat


class DoosanRecord(Robot):
    def __init__(self):
        self.shutdown_flag = False 
        rospy.init_node('manual_control_node')
        rospy.on_shutdown(self.shutdown)
        
        # Get button state
        self.get_buttons_service = rospy.ServiceProxy('/dsr01a0509/system/get_buttons_state', GetButtonsState)
        
        # Control parameters
        self.data_collection_freq = 60  # Hz for data recording
        
        # Thread control
        self.control_thread = None
        self.control_active = False
        
        super().__init__()

    @property
    def button(self):
        try:
            return self.get_buttons_service().state[0]   # 0th button is used demo based learning
        except rospy.ServiceException as e:
            rospy.logwarn(f"Button service call failed: {e}")
            return False
    
    @property
    def current_velocity(self):
        X_dot = np.zeros(6)
        J = self.Robot_RT_State.jacobian_matrix
        X_dot = J @ self.q_dot[:,np.newaxis]
        return X_dot.reshape(-1)   # in [m/s, rad/s]
    
    def state_callback(self):
        """Store complete info of robot and joint angle as current_posj in degrees"""
        self.q = 0.0174532925 * self.Robot_RT_State.actual_joint_position_abs   # convert from deg to rad
        self.q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s
        self.current_position = 0.001 * self.Robot_RT_State.actual_tcp_position[:3]   # (x, y, z), converted from mm to m
        self.current_euler = 0.0174532925 * self.Robot_RT_State.actual_tcp_position[3:]   # (a, b, c) follows Euler ZYZ notation, convert from deg to rad
        self.current_quat = Rotation.from_euler("ZYZ", self.Robot_RT_State.actual_tcp_position[3:], degrees=True).as_quat()   # orientation in quaternion
        self.current_linear_vel = 0.001 * self.Robot_RT_State.actual_tcp_velocity[:3]   # linear velocity # (Vx, Vy, Vz), in m/s
        self.current_angular_vel = np.radians(self.Robot_RT_State.actual_tcp_velocity[3:])   # angular velocity (ωx, ωy, ωz) in rad/s
        self.current_wrench = self.Robot_RT_State.external_tcp_force   # in [N, Nm], w.r.t. base coordinates

    def start_gravity_compensation(self):
        """Start gravity compensation in a separate thread"""
        if not self.control_active:
            self.control_active = True
            self.control_thread = threading.Thread(target=self._gravity_control_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            rospy.loginfo("Gravity compensation started")

    def stop_gravity_compensation(self):
        """Stop gravity compensation"""
        self.control_active = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)
        rospy.loginfo("Gravity compensation stopped")

    def _gravity_control_loop(self):
        """Internal gravity compensation control loop"""
        rate = rospy.Rate(self.write_rate)  # Hz for gravity compensation (high frequency for smooth control)
        try:
            while self.control_active and not rospy.is_shutdown() and not self.shutdown_flag:
                self.state_callback()  # Update robot state at control frequency

                G_torques = self.Robot_RT_State.gravity_torque  # calculate gravitational torque in Nm
                
                # Send gravity compensation torques
                writedata = TorqueRTStream()
                writedata.tor = G_torques
                writedata.time = 0.0
                
                self.torque_publisher.publish(writedata)
                rate.sleep()

        except rospy.ROSInterruptException:
            pass
        except Exception as e:
            rospy.logerr(f"Gravity control error: {e}")

    def time_based_resampling(self, demo_data, demo_times, target_length):
        """
        Resample based on time - most accurate for preserving temporal dynamics
        
        Args:
            demo_data: List of trajectories (each trajectory shape: (N_i, feature_dim))
            demo_times: List of time arrays for each demo
            target_length: Target number of points after resampling
        
        Returns:
            resampled_data: List of trajectories with equal length
        """
        resampled_demos = []
        
        for i, (traj, times) in enumerate(zip(demo_data, demo_times)):
            if len(traj) < 2:
                rospy.logwarn(f"Demo {i+1} has insufficient points: {len(traj)}")
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
                kind = 'cubic'
                try:
                    f = interp1d(times, traj[:, dim], kind=kind, bounds_error=False, fill_value='extrapolate')
                    interpolated_values = f(t_uniform)
                    resampled_traj[:, dim] = np.array(interpolated_values).flatten()
                except Exception as e:
                    rospy.logwarn(f"Interpolation failed for demo {i+1}, dim {dim}: {e}")
                    # Fallback to linear interpolation
                    f = interp1d(times, traj[:, dim], kind='linear', bounds_error=False, fill_value='extrapolate')
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
        self.demo_wrench = self.time_based_resampling(self.demo_wrench, self.demo_time, target_length)

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
        self.demo_wrench = np.array([demo for demo in self.demo_wrench])
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

    def traj_record(self, trigger=0.005):  # trigger is 5mm     
        init_pose = self.current_position.copy()
        robot_perturbation = 0
        rospy.loginfo(f"Press and hold the robot button when ready to record.")

        rate = rospy.Rate(self.data_collection_freq)
        
        # Wait for movement to trigger recording
        while robot_perturbation < trigger and not rospy.is_shutdown():
            current_pos = self.current_position
            robot_perturbation = np.linalg.norm(current_pos - init_pose)
            rate.sleep()
        
        rospy.loginfo(f"Movement detected! Waiting for button press to start recording...")
        
        # Initialize recording arrays
        self.recorded_q = [self.q.copy()]
        self.recorded_q_dot = [self.q_dot.copy()]
        self.recorded_trajectory = [self.current_position.copy()]
        self.recorded_orientation = [self.current_quat.copy()]
        self.recorded_linear_velocity = [self.current_linear_vel.copy()]
        self.recorded_angular_velocity = [self.current_angular_vel.copy()]
        self.recorded_wrench = [self.current_wrench.copy()]

        self.start_time = time.time()
        self.recorded_time = [0.0]
   
        # Record while button is pressed
        while self.button and not rospy.is_shutdown():
            self.recorded_q.append(self.q.copy())
            self.recorded_q_dot.append(self.q_dot.copy())
            self.recorded_trajectory.append(self.current_position.copy())
            self.recorded_orientation.append(self.current_quat.copy())
            self.recorded_linear_velocity.append(self.current_linear_vel.copy())
            self.recorded_angular_velocity.append(self.current_angular_vel.copy())
            self.recorded_wrench.append(self.current_wrench.copy())
            
            # Record relative time since start
            current_time = time.time()
            relative_time = current_time - self.start_time
            self.recorded_time.append(relative_time)
            
            rate.sleep()

        # Convert lists to numpy arrays
        self.recorded_q = np.array(self.recorded_q)
        self.recorded_q_dot = np.array(self.recorded_q_dot)
        self.recorded_trajectory = np.array(self.recorded_trajectory)
        self.recorded_orientation = np.array(self.recorded_orientation)
        self.recorded_linear_velocity = np.array(self.recorded_linear_velocity)
        self.recorded_angular_velocity = np.array(self.recorded_angular_velocity)
        self.recorded_wrench = np.array(self.recorded_wrench)
        self.recorded_time = np.array(self.recorded_time)

    def multiple_traject_record(self, no_of_demos=1):
        """Record multiple demonstrations"""
        rospy.loginfo(f"Starting recording of {no_of_demos} demonstrations")
        
        # Initialize demo storage lists
        self.demo_q = []
        self.demo_q_dot = []
        self.demo_trajectory = []
        self.demo_orientation = []
        self.demo_linear_velocity = []
        self.demo_angular_velocity = []
        self.demo_wrench = []
        self.demo_time = []
        
        for i in range(no_of_demos):
            rospy.loginfo(f"\n=== Recording demonstration {i+1}/{no_of_demos} ===")
            
            # Wait for user to be ready
            input(f"Press Enter when ready to record demonstration {i+1}...")
            
            try:
                self.traj_record()  # record ith demo
                
                # Store the recorded data
                self.demo_q.append(self.recorded_q.copy())
                self.demo_q_dot.append(self.recorded_q_dot.copy())
                self.demo_trajectory.append(self.recorded_trajectory.copy())
                self.demo_orientation.append(self.recorded_orientation.copy())
                self.demo_linear_velocity.append(self.recorded_linear_velocity.copy())
                self.demo_angular_velocity.append(self.recorded_angular_velocity.copy())
                self.demo_wrench.append(self.recorded_wrench.copy())
                self.demo_time.append(self.recorded_time.copy())

                rospy.loginfo(f"Successfully stored demonstration {i+1}")
                
                """reset state automatically"""
                if i < no_of_demos - 1:  # Not the last demo
                    rospy.loginfo("Move robot to starting position for next demonstration")
                    rospy.sleep(2.0)  # Give time to reset
                    
            except Exception as e:
                rospy.logerr(f"Error recording demonstration {i+1}: {e}")
                continue

        rospy.loginfo(f"\nCompleted recording {len(self.demo_q)} demonstrations")

    def cleanup(self):
        """Cleanup function called on shutdown"""
        rospy.loginfo("Cleaning up...")
        self.stop_gravity_compensation()
        super().cleanup() if hasattr(super(), 'cleanup') else None

    def shutdown(self):
        """ROS shutdown callback"""
        rospy.loginfo("Shutdown signal received")
        self.shutdown_flag = True
        self.cleanup()

    def save(self, name='demo'):
        """Save demonstration data to file"""
        curr_dir = os.getcwd()
        filepath = os.path.join(curr_dir+ '/data/', f'{name}.npz')
        np.savez(filepath,
                freq=self.data_collection_freq,
                q=self.demo_q,
                q_dot=self.demo_q_dot,
                traj=self.demo_trajectory,
                ori=self.demo_orientation,
                vel=self.demo_linear_velocity,
                omega=self.demo_angular_velocity,
                wrench=self.demo_wrench,
                time=self.demo_time)
        
        rospy.loginfo(f"Successfully saved demonstration data to {filepath}")

    def load(self, name='demo'):
        """Load demonstration data from file"""
        curr_dir = os.getcwd()
        filepath = os.path.join(curr_dir, 'data', f'{name}.npz')
        data = np.load(filepath, allow_pickle=True)
        
        self.demo_q = data['q']
        self.demo_q_dot = data['q_dot']
        self.demo_trajectory = data['traj']
        self.demo_orientation = data['ori']
        self.demo_linear_velocity = data['vel']
        self.demo_angular_velocity = data['omega']
        self.demo_wrench = data['wrench']
        self.demo_time = data['time']
        
        rospy.loginfo(f"Successfully loaded demonstration data from {filepath}")


if __name__ == "__main__":
    """Main function to run the demonstration recording"""
    # Move to initial position first
    rospy.loginfo("Moving to initial position...")
    p1 = posj(0, 0, 90, 0, 90, 0)  # Joint space angles in degrees
    movej(p1, vel=40, acc=20)
    time.sleep(2.0)
    try:        
        # Initialize controller
        controller = DoosanRecord()
        rospy.loginfo("Robot setup complete - Ready for manual demonstration")

        # Start gravity compensation
        controller.start_gravity_compensation()
        rospy.sleep(0.5)  # Let gravity compensation stabilize

        controller.multiple_traject_record(no_of_demos=1)  # Record demonstrations
        controller.resample_demonstrations(target_length=500)  # Resample to equal lengths
        controller.save(name="demo_discrete_with_wrench")  # Save data
        
        rospy.loginfo("Demo recording completed successfully!")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Recording interrupted by user")
    except KeyboardInterrupt:
        rospy.loginfo("Recording interrupted by keyboard")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        if 'controller' in locals():
            controller.cleanup()