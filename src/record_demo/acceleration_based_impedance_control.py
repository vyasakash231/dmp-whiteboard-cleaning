#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import Robot
from learn_dmp import PositionDMP, QuaternionDMP
from pose_transform import make_quat_continuity


class Acceleration_CartesianImpedanceControl(Robot):
    def __init__(self, file_name='demo'):
        self.shutdown_flag = False  
        self.file_name = file_name
        
        self.Ko = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # Current robot state variables
        self.tau_J_d = np.zeros(6)  # Previous desired torque
        self.delta_tau_max = 1.0
        
        # Initial estimated frictional torque
        self.tau_f = np.zeros(self.n)

        self.impedance_force = np.zeros((self.n,1))

        super().__init__()

    def load_demo(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.data_collect_freq = data["freq"]
        self.q_demo = data['q']    # shape: (N, 6), in rad
        self.position_demo = 1000 * data['traj']   # shape: (N, 3), convert from m to mm
        self.orientation_demo = data['ori']   # shape: (N, 4)
        self.linear_velocity_demo = data['vel']   # shape: (N, 3), in m/s
        self.angular_velocity_demo = data['omega']   # shape: (N, 3), in rad/s
        # self.gripper = data['grip']
        self.N = self.position_demo.shape[0]   # no of sample points
        self.time = data['time']

    def store_data(self):
        pos = 0.001 * self.Robot_RT_State.actual_tcp_position[:3]    # in m
        # orient = self._eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # quaternions
        orient = Rotation.from_euler("ZYZ", self.Robot_RT_State.actual_tcp_position[3:], degrees=True).as_quat()   # quaternions
        self.record_trajectory = np.vstack((self.record_trajectory, pos))  # shape: (N, 3) 
        self.record_orientation = np.vstack((self.record_orientation, orient))  # shape: (N, 4)

        self.record_motor_torque = np.vstack((self.record_motor_torque, self.Robot_RT_State.actual_motor_torque))  # shape: (N, 6) 
        self.record_joint_torque = np.vstack((self.record_joint_torque, self.Robot_RT_State.actual_joint_torque))  # shape: (N, 6)
        self.record_imp_force = np.vstack((self.record_imp_force, self.impedance_force.reshape(-1)))  # shape: (N, 6)

    def start(self):
        """Initialize controller with current robot state"""
        # Load data
        self.load_demo(name=self.file_name)

        # define desired pose => [x, y, z] & [q1, q2, q3, q0]
        self.position_des_next = self.Robot_RT_State.actual_tcp_position[:3].copy()    # (x, y, z) in mm
        # self.orientation_des_next = self._eul2quat(self.Robot_RT_State.actual_tcp_position[3:].copy())    # Convert angles from Euler ZYZ (in degrees) to quaternion        
        self.orientation_des_next = Rotation.from_euler("ZYZ", self.Robot_RT_State.actual_tcp_position[3:].copy(), degrees=True).as_quat()    # Convert angles from Euler ZYZ (in degrees) to quaternion        

        # for plotting
        self.record_trajectory = 0.001 * self.Robot_RT_State.actual_tcp_position[:3]   # in m
        # self.record_orientation = self._eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # quaternions
        self.record_orientation = Rotation.from_euler("ZYZ", self.Robot_RT_State.actual_tcp_position[3:], degrees=True).as_quat()   # quaternions

        self.record_motor_torque = self.Robot_RT_State.actual_motor_torque  # in Nm
        self.record_joint_torque = self.Robot_RT_State.actual_joint_torque  # in Nm
        self.record_imp_force = np.zeros(self.n)
        rospy.loginfo("CartesianImpedanceController: Controller started")

    @property
    def current_velocity(self):
        X_dot = np.zeros(6)
        X_dot = self.J @ self.q_dot[:,np.newaxis]
        return X_dot.reshape(-1)   # in [m/s, rad/s]
        # X_dot[:3] = 0.001 * self.Robot_RT_State.actual_tcp_velocity[:3]
        # X_dot[3:] = 0.0174532925 * self.Robot_RT_State.actual_tcp_velocity[3:]
        # return X_dot
    
    @property
    def velocity_error(self):
        # Combine into a single 6D velocity vector
        desired_velocity = np.zeros(6)
        desired_velocity[:3] = self.desired_linear_vel  # in m/s
        desired_velocity[3:] = self.desired_angular_vel  # in rad/s 

        # EE-velocity in task-space
        current_velocity = self.current_velocity

        # Calculate velocity error (current - desired)
        error_dot = current_velocity - desired_velocity
        return error_dot

    @property
    def position_error(self):
        # actual robot flange position w.r.t. base coordinates: (x, y, z, a, b, c), where (a, b, c) follows Euler ZYZ notation [mm, deg]
        current_position = self.Robot_RT_State.actual_tcp_position[:3]   # (x, y, z) in mm
        return 0.001 * (current_position - self.position_des_next)   # convert from mm to m
    
    @property
    def orientation_error(self):
        # current_orientation = self._eul2quat(self.Robot_RT_State.actual_tcp_position[3:])   # Convert angles from Euler ZYZ (in degrees) to quaternion        
        current_orientation = Rotation.from_euler("ZYZ", self.Robot_RT_State.actual_tcp_position[3:], degrees=True).as_quat()   # Convert angles from Euler ZYZ (in degrees) to quaternion        

        if np.dot(current_orientation, self.orientation_des_next) < 0.0:
            current_orientation = -current_orientation
        
        current_rotation = Rotation.from_quat(current_orientation)  # convert to rotation matrix from quaternion, default order: [x,y,z,w]
        desired_rotation = Rotation.from_quat(self.orientation_des_next)  # convert to rotation matrix from quaternion, default order: [x,y,z,w]
        
        # # Compute the "difference" quaternion (q_error = q_current^-1 * q_desired)
        # """https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames"""
        error_rotation = current_rotation.inv() * desired_rotation
        error_quat = error_rotation.as_quat()  # default order: [x,y,z,w]
        
        # Extract x, y, z components of error quaternion
        rot_error = error_quat[:3]
        
        # Transform orientation error to base frame
        current_rotation_matrix = current_rotation.as_matrix()  # Assuming this returns a 3x3 rotation matrix
        rot_error = current_rotation_matrix @ rot_error[:, np.newaxis]
        return rot_error.reshape(-1)
    
    """inertia weighted pseudo-inverse of Jacobian matrix"""
    def J_pinv(self, Mq):
        # Mq_inv = np.linalg.inv(Mq)  # This was calculated based on LU-Decomposition which is numerically not very stable
        Mq_inv = self._svd_solve(Mq)  # SVD is more numerically stable when dealing with matrices that might be ill-conditioned
        Mx_inv = self.J @ (Mq_inv @ self.J.T)
        if abs(np.linalg.det(Mx_inv)) >= 1e-4:
            inertia_weighted_pseudo_inv = Mq_inv @ self.J.T @ self._svd_solve(Mx_inv)
        else:
            inertia_weighted_pseudo_inv = Mq_inv @ self.J.T @ np.linalg.pinv(Mx_inv, rcond=1e-5)   
        return inertia_weighted_pseudo_inv

    def set_compliance_parameters(self, translational_stiffness, rotational_stiffness):
        # Update stiffness matrix
        self.K_cartesian = np.eye(6)
        self.K_cartesian[:3, :3] *= translational_stiffness
        self.K_cartesian[3:, 3:] *= rotational_stiffness
        
        # Update damping matrix (critically damped)
        self.D_cartesian = np.eye(6)
        self.D_cartesian[:3, :3] *= 2.0 * np.sqrt(translational_stiffness)
        self.D_cartesian[3:, 3:] *= 1.0 * np.sqrt(rotational_stiffness)

    def saturate_torque(self, tau):
        # Now apply peak torque limits based on Doosan A0509 specs
        limit_factor = 0.95
        max_torque_limits = limit_factor * np.array([190.0, 190.0, 190.0, 40.0, 40.0, 40.0])  # Nm

        if tau.ndim == 2:
            tau = tau.reshape(-1)

        # Clip torque values to stay within limits (both positive and negative)
        tau = np.clip(tau, -max_torque_limits, max_torque_limits)
        return tau

    def plot_data(self):
        pass

    """Impedance control from the Appendix section of the paper, Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def run_controller(self, K_trans, K_rot):
        self.start()
        self.set_compliance_parameters(K_trans, K_rot)
        error = np.zeros(6)

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate
        
        # Calculate time step between trajectory points (in seconds)
        traj_dt = 1.0/self.data_collect_freq  #  freq is same as demo collection frequency

        # Track start time for trajectory indexing
        start_time = rospy.Time.now().to_sec()

        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                # Calculate elapsed time and determine trajectory index
                current_time = rospy.Time.now().to_sec()
                elapsed_time = current_time - start_time
                current_idx = int(elapsed_time / traj_dt)
                
                # Ensure index is within bounds
                if current_idx >= self.N:
                    current_idx = self.N - 1
                    if current_idx == self.N - 1 and np.linalg.norm(self.position_error)*1000 < 10.0:  # if error is less then 10mm break
                        rospy.loginfo("Trajectory complete and position error < 10mm")
                        break

                # Update desired position and orientation
                self.position_des_next = self.position_demo[current_idx,:]  # (x, y, z) in mm
                self.orientation_des_next = self.orientation_demo[current_idx,:]

                # Get desired velocity from recorded data
                self.desired_linear_vel = self.linear_velocity_demo[current_idx,:]  # Already in m/s
                self.desired_angular_vel = self.angular_velocity_demo[current_idx,:]  # Already in rad/s

                self.q = 0.0174532925 * self.Robot_RT_State.actual_joint_position_abs   # convert from deg to rad
                self.q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s

                M = self.Robot_RT_State.mass_matrix
                C = self.Robot_RT_State.coriolis_matrix
                self.J,_,_ = self.kinematic_model.Jacobian(self.q)
                J_dot,_,_ = self.kinematic_model.Jacobian_dot(self.q, self.q_dot)

                # define EE-Position & Orientation error in task-space
                error[:3] = self.position_error
                error[3:] = self.orientation_error

                # define EE-Velocitt error in task-space
                velocity_error = self.velocity_error

                # Cartesian PD control with damping
                commanded_cartesian_acc = self.K_cartesian @ error[:, np.newaxis] + self.D_cartesian @ velocity_error[:, np.newaxis]   # eqn (51-52) from the paper
                joint_acc = - self.J_pinv(M) @ (commanded_cartesian_acc - J_dot @ self.q_dot[:,np.newaxis])  # eqn (49) from the paper

                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # Compute desired torque -> eqn (50) from the paper
                tau_d = M @ joint_acc + C @ self.q_dot[:,np.newaxis] + G_torque[:, np.newaxis]
                
                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d)
                
                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()   # target motor torque [Nm]
                writedata.time = traj_dt    # target time [sec]
                self.torque_publisher.publish(writedata)

                # store actual data for plotting
                self.store_data()

                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()


    def run_dmp(self, K_trans, K_rot):
        self.start()
        self.set_compliance_parameters(K_trans, K_rot)
        error = np.zeros(6)

        rate = rospy.Rate(self.write_rate)  # 1000 Hz control rate

        # Calculate time step between trajectory points (in seconds)
        traj_dt = 0.005  # 10Hz = 0.1s between points  -->  freq is different from demo collection freq

        # Start DMPS
        position_dmp = PositionDMP(no_of_DMPs=3, no_of_basis_func=25, T=10, dt=traj_dt, K=750.0, alpha=1.0)
        quaternion_dmp = QuaternionDMP(no_of_basis_func=25, T=10, dt=traj_dt, K=500.0, alpha=1.0)

        # learn Weights based on position Demo
        X_demo = 0.001 * self.position_demo.T   # demo position data of shape (3, N) in m
        V_demo = self.linear_velocity_demo.T   # demo velocity data of shape (3, N) in m/s
        position_dmp.learn_dynamics(time=self.time, X_des=X_demo, dX_des=V_demo)

        # learn Weights based on orientation Demo
        Q_demo = self.orientation_demo.T   # orientation data of shape (4, N)
        omega_demo = self.angular_velocity_demo.T   # demo velocity data of shape (3, N) in rad/s
        quaternion_dmp.learn_dynamics(time=self.time, q_des=Q_demo, omega_des=omega_demo)

        # Goal State
        X_goal = X_demo[:,[-1]] # + np.array([[0.0],[0.05],[0.1]])  # in m
        Q_goal = Q_demo[:,[-1]]

        # initial joint position and velocity
        self.q = 0.0174532925 * self.Robot_RT_State.actual_joint_position_abs   # convert from deg to rad
        self.q_dot = 0.0174532925 * self.Robot_RT_State.actual_joint_velocity_abs   # convert from deg/s to rad/s

        try:
            while not rospy.is_shutdown() and not self.shutdown_flag:
                X, X_dot = position_dmp.step(X_goal, tau=1.0)   # run and record timestep
                self.position_des_next = 1000 * X.reshape(-1)   # convert from m to mm
                self.desired_linear_vel = X_dot.reshape(-1)   # in m/s

                q, omega = quaternion_dmp.step(Q_goal, tau=1.0)   # run and record timestep
                self.orientation_des_next = q.reshape(-1)
                self.desired_angular_vel = omega.reshape(-1)  # in rad/s

                M = self.Robot_RT_State.mass_matrix
                C = self.Robot_RT_State.coriolis_matrix
                self.J,_,_ = self.kinematic_model.Jacobian(self.q)
                J_dot,_,_ = self.kinematic_model.Jacobian_dot(self.q, self.q_dot)

                # define EE-Position & Orientation error in task-space
                error[:3] = self.position_error  # in m
                error[3:] = self.orientation_error

                # define EE-Velocitt error in task-space
                velocity_error = self.velocity_error

                # Cartesian PD control with damping
                commanded_cartesian_acc = self.K_cartesian @ error[:, np.newaxis] + self.D_cartesian @ velocity_error[:, np.newaxis]   # eqn (51-52) from the paper
                joint_acc = - self.J_pinv(M) @ (commanded_cartesian_acc - J_dot @ self.q_dot[:,np.newaxis])  # eqn (49) from the paper

                # compute gravitational torque in Nm
                G_torque = self.Robot_RT_State.gravity_torque 

                # Compute desired torque -> eqn (50) from the paper
                tau_d = M @ joint_acc + C @ self.q_dot[:,np.newaxis] + G_torque[:, np.newaxis]

                # Saturate torque to avoid limit breach
                tau_d = self.saturate_torque(tau_d)
                
                writedata = TorqueRTStream()
                writedata.tor = tau_d.tolist()   # target motor torque [Nm]
                writedata.time = 0.0    # target time [sec]
                self.torque_publisher.publish(writedata)

                # store actual data for plotting
                self.store_data()

                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        finally:
            self.cleanup()

    def save(self, name='task_performed'):
        curr_dir=os.getcwd()
        np.savez(curr_dir + '/data/' + str(name) + '.npz',
                traj=self.record_trajectory,
                ori=self.record_orientation,
                motor_torque=self.record_motor_torque,
                joint_torque=self.record_joint_torque,
                imp_force=self.record_imp_force
                #  grip=self.recorded_gripper
                )
