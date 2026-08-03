#! /usr/bin/python3
import os, sys
from math import *
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from common_utils import Robot_KM
from pose_transform import quat2mat, make_quat_continuity


class PlotData:
    def __init__(self, demo_file_name, task_performed_name=None):
        # load demo data
        self.load_demo(name=demo_file_name)

        # Check if the file of performed task already exists, if it exists load the data.
        curr_dir=os.getcwd()
        if os.path.exists(curr_dir + f'/data/{task_performed_name}' + '.npz'):
            print("task_performed already exists.")
            self.load_task_performed(name=task_performed_name)

        # Modified-DH Parameters 
        self.DOF = 6
        alpha = np.array([0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2])   
        a = np.array([0, 0, 0.409, 0, 0, 0])  # data from parameter data-sheet (in meters)
        d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])  # data from parameter data-sheet (in meters)
        d_nn = np.array([[0.0], [0.0], [0.0]])  # TCP coord in end-effector frame
        DH_params="modified"

        self.KM = Robot_KM(self.DOF, alpha, a, d, d_nn, DH_params)
        self.store_FK_data()
        self.store_rotation_matrices()

        # Setup the scale factor for orientation vectors
        self.orientation_scale = 0.15

    def load_demo(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.N_demo = data['q'].shape[0]   # no of sample points

        if self.N_demo == 1:
            self.q_demo = data['q'].squeeze(0)    # shape: (N_sample, 6), in rad
            self.position_demo = data['traj'].squeeze(0)   # shape: (N_sample, 3), in m
            self.orientation_demo = data['ori'].squeeze(0)   # shape: (N_sample, 4)
            self.linear_velocity_demo = data['vel'].squeeze(0)   # shape: (N_sample, 3), in m/s
            self.angular_velocity_demo = data['omega'].squeeze(0)   # shape: (N_sample, 3), in rad/s
            # self.gripper = data['grip']

            # Important to make quaternions continuous 
            self.orientation_demo = make_quat_continuity(self.orientation_demo)

            self.q1 = self.orientation_demo[:,0] 
            self.q2 = self.orientation_demo[:,1] 
            self.q3 = self.orientation_demo[:,2] 
            self.q0 = self.orientation_demo[:,3] 
        else:
            self.q_demo = data['q']    # shape: (N_demo, N_sample, 6), in rad
            self.position_demo = data['traj']   # shape: (N_demo, N_sample, 3), in m
            self.orientation_demo = data['ori']   # shape: (N_demo, N_sample, 4)
            self.linear_velocity_demo = data['vel']   # shape: (N_demo, N_sample, 3), in m/s
            self.angular_velocity_demo = data['omega']   # shape: (N_demo, N_sample, 3), in rad/s
            # self.gripper = data['grip']

            # Important to make quaternions continuous 
            self.orientation_demo = make_quat_continuity(self.orientation_demo)

            self.q1 = self.orientation_demo[:,0] 
            self.q2 = self.orientation_demo[:,1] 
            self.q3 = self.orientation_demo[:,2] 
            self.q0 = self.orientation_demo[:,3] 

    def load_task_performed(self, name='task_performed'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.position = data['traj']   # shape: (N, 3), in m
        self.orientation = data['ori']   # shape: (N, 4)
        self.motor_torque = data['motor_torque']   # shape: (N, 3), in Nm
        self.joint_torque = data['joint_torque']   # shape: (N, 3), in Nm
        self.impedance_force = data['imp_force']   # shape: (N, 3), in [N, Nm]
        # self.gripper = data['grip']
       
        # Important to make quaternions continuous 
        self.orientation = -make_quat_continuity(self.orientation)

        self.N = self.position.shape[0]   # no of sample points

    def store_FK_data(self):
        X_coord, Y_coord, Z_coord = [], [], []

        if self.N_demo == 1:
            # Single demo: iterate through all time steps
            n_timesteps = self.q_demo.shape[0]
            print(f"Computing FK for {n_timesteps} time steps")
            
            for i in range(n_timesteps):
                X, Y, Z = self.KM.taskspace_coord(self.q_demo[i,:])
                X_coord.append(X)
                Y_coord.append(Y)
                Z_coord.append(Z)
        else:
            # Multiple demos: iterate through demos
            print(f"Computing FK for {self.N_demo} demonstrations")
            for i in range(self.N_demo):
                # For multiple demos, you might need to handle this differently
                # depending on your data structure
                X, Y, Z = self.KM.taskspace_coord(self.q_demo[i,:])
                X_coord.append(X)
                Y_coord.append(Y)
                Z_coord.append(Z)

        self.X_coord = np.array(X_coord)
        self.Y_coord = np.array(Y_coord)
        self.Z_coord = np.array(Z_coord)
        print(f"FK data shape: X={self.X_coord.shape}, Y={self.Y_coord.shape}, Z={self.Z_coord.shape}")

    def store_rotation_matrices(self):
        self.rotation_matrices = []
        
        if self.N_demo == 1:
            # Single demo: iterate through all time steps
            n_timesteps = self.orientation_demo.shape[0]
            print(f"Computing rotation matrices for {n_timesteps} time steps")
            
            for i in range(n_timesteps):
                rot_mat = quat2mat(self.orientation_demo[i])
                self.rotation_matrices.append(rot_mat)
        else:
            # Multiple demos
            print(f"Computing rotation matrices for {self.N_demo} demonstrations")
            for i in range(self.N_demo):
                rot_mat = quat2mat(self.orientation_demo[i])
                self.rotation_matrices.append(rot_mat)

        # Draw unit sphere for reference (vector part of unit quaternion lies within this)
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        self.sphere_x = np.outer(np.cos(u), np.sin(v))
        self.sphere_y = np.outer(np.sin(u), np.sin(v))
        self.sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        print(f"Stored {len(self.rotation_matrices)} rotation matrices")

    """#####################################################################################################################"""

    def setup_demo_plot(self):
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 3)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,2]))
        return self.create_animation_1(self.update_plot_1)
    
    def setup_performed_task_plot(self):
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 3)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,2]))
        return self.create_animation_2(self.update_plot_2)
    
    """#####################################################################################################################"""

    def create_animation_1(self, update_func):
        # For single demo, we need to animate through the time steps, not the demo count
        if self.N_demo == 1:
            # Use the number of time steps in the single demonstration
            n_timesteps = self.position_demo.shape[0]
            print(f"Single demo with {n_timesteps} time steps")
            
            if n_timesteps > 400:
                frames = np.linspace(0, n_timesteps-1, 250, dtype=int)
                interval = 50
                print(f"Using {len(frames)} subsampled frames")
            else:
                frames = range(n_timesteps)
                interval = 100
                print(f"Using all {n_timesteps} frames")
        else:
            # Multiple demos - this was your original logic
            print(f"Multiple demos: {self.N_demo}")
            if self.N_demo > 400:
                frames = np.linspace(0, self.N_demo-1, 250, dtype=int)
                interval = 50
            else:
                frames = range(self.N_demo)
                interval = 100

        print(f"Frame range: {min(frames)} to {max(frames)}")
        print(f"Animation interval: {interval}ms")
        
        anim = FuncAnimation(self.fig, update_func, frames=frames, interval=interval, blit=False, repeat=False)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.tight_layout()
        return anim

    def create_animation_2(self, update_func):
        # This one uses self.N which should be correct already
        print(f"Task performed data points: {self.N}")
        print(f"Position shape: {self.position.shape}")
        
        if self.N > 400:
            frames = np.linspace(0, self.N-1, 250, dtype=int)
            interval = 50
            print(f"Using {len(frames)} subsampled frames")
        else:
            frames = range(self.N)
            interval = 100
            print(f"Using all {self.N} frames")

        print(f"Frame range: {min(frames)} to {max(frames)}")
        print(f"Animation interval: {interval}ms")
        
        anim = FuncAnimation(self.fig, update_func, frames=frames, interval=interval, blit=False, repeat=False)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        # plt.tight_layout()
        return anim
        
    """#####################################################################################################################"""


    def update_plot_1(self, frame):
        self._plot_1(self.axs[0], frame)  # position data
        self._plot_2(self.axs[1], frame)  # orientation data
        self._plot_3(self.axs[2], frame)  # orientation data
        self._set_3d_plot_properties(self.axs[0], x_lim=[-1.0, 1.0], y_lim=[-1.0, 1.0], z_lim=[-0.1, 1.0])
        self._set_3d_plot_properties(self.axs[1],  x_lim=[-1.0, 1.0], y_lim=[-1.0, 1.0], z_lim=[-1.0, 1.0])
        return self.axs

    def update_plot_2(self, frame):
        self._plot_4(self.axs[0], frame)  # position data
        self._plot_5(self.axs[1], frame)  # orientation data
        self._plot_6(self.axs[2], frame)  # orientation data
        self._set_3d_plot_properties(self.axs[0], x_lim=[-0.25, 1.0], y_lim=[-0.5, 0.5], z_lim=[-0.5, 1.0])
        self._set_3d_plot_properties(self.axs[1],  x_lim=[-1.0, 1.0], y_lim=[-1.0, 1.0], z_lim=[-1.0, 1.0])
        return self.axs

    """#####################################################################################################################"""
    
    def _plot_1(self, ax, k):
        ax.clear()
        ax.plot(self.position_demo[:k+1,0], self.position_demo[:k+1,1], self.position_demo[:k+1,2], "k", linewidth=1.5)

        for j in range(self.DOF):
            ax.plot(self.X_coord[k,j:j+2], self.Y_coord[k,j:j+2], self.Z_coord[k,j:j+2], '-', linewidth=9-j)  # Links
            ax.plot(self.X_coord[k,j], self.Y_coord[k,j], self.Z_coord[k,j], 'ko', markersize=9-j)   # Joints

        # Also plot orientation vectors in this plot
        self._plot_orientation_vectors(ax, k)

    def _plot_orientation_vectors(self, ax, k):
        position = self.position_demo[k]
        
        # Use precomputed rotation matrix for better performance
        rotation_matrix = self.rotation_matrices[k]
        
        # Scale the orientation vectors
        scale = self.orientation_scale
        
        # Create a single quiver plot for all axes to improve performance
        origins = np.tile(position, (3, 1))
        
        # Create directions for all three axes
        directions = np.zeros((3, 3))
        directions[0] = rotation_matrix[:, 0] * scale  # X axis
        directions[1] = rotation_matrix[:, 1] * scale  # Y axis
        directions[2] = rotation_matrix[:, 2] * scale  # Z axis
        
        # Colors for the three axes
        colors = ['r', 'g', 'b']
        
        # Plot all three axes
        for i in range(3):
            ax.quiver(origins[i, 0], origins[i, 1], origins[i, 2], directions[i, 0], directions[i, 1], directions[i, 2], color=colors[i], alpha=0.5, linewidth=2)

    def _plot_2(self, ax, k):
        """
        When the w component changes significantly during a smooth rotation, the normalized vector part 
        (x,y,z) will move inside the 3D unit sphere. This is perfectly normal and actually expected for 
        certain types of rotations, especially those that pass near the "identity rotation" (w ≈ 1, x,y,z ≈ 0)
        The path appearing inside the sphere doesn't indicate an error in your quaternions, it's just an artifact of the projection method. 
        As long as your original quaternions are normalized in 4D, they represent valid rotations.
        """
        ax.clear()

        # Plot unit sphere
        ax.plot_surface(self.sphere_x, self.sphere_y, self.sphere_z, color="lightblue", alpha=0.3)

        # Normalize quaternions to ensure they're on the unit hypersphere
        q_norms = np.sqrt(self.orientation_demo[:k+1,0]**2 + self.orientation_demo[:k+1,1]**2 + self.orientation_demo[:k+1,2]**2 + self.orientation_demo[:k+1,3]**2)
        
        qx_norm = self.orientation_demo[:k+1,0] / q_norms
        qy_norm = self.orientation_demo[:k+1,1] / q_norms
        qz_norm = self.orientation_demo[:k+1,2] / q_norms
            
        # Plot the trajectory line
        ax.plot(qx_norm, qy_norm, qz_norm, 'k-', linewidth=3.0)

        # Draw reference unit vectors
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.5, length=0.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, length=0.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.5, length=0.5, normalize=True)

    def _plot_3(self, ax, k):
        ax.clear()

        # Plot quaternion components over time
        time_indices = np.arange(k+1)
        
        # Plot quaternion components as trajectories
        ax.plot(time_indices, self.orientation_demo[:k+1,3], 'r-', linewidth=2, label='qw')
        ax.plot(time_indices, self.orientation_demo[:k+1,0], 'g-', linewidth=2, label='qx')  
        ax.plot(time_indices, self.orientation_demo[:k+1,1], 'b-', linewidth=2, label='qy')
        ax.plot(time_indices, self.orientation_demo[:k+1,2], 'k-', linewidth=2, label='qz')
        
        # Mark the current points
        if k > 0:
            ax.plot(k, self.orientation_demo[k,3], 'ro', markersize=8)
            ax.plot(k, self.orientation_demo[k,0], 'go', markersize=8)
            ax.plot(k, self.orientation_demo[k,1], 'bo', markersize=8)
            ax.plot(k, self.orientation_demo[k,2], 'ko', markersize=8)
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Quaternion Components')
        ax.set_title('Quaternion Components Over Time')
        ax.set_ylim(-1.0, 1.2)
        ax.grid(True)

    def _plot_4(self, ax, k):
        ax.clear()
        ax.plot(self.position_demo[:,0], self.position_demo[:,1], self.position_demo[:,2], "k--", linewidth=1.5)
        ax.plot(self.position[:k+1,0], self.position[:k+1,1], self.position[:k+1,2], "r", linewidth=1.25)

    def _plot_5(self, ax, k):
        ax.clear()

        # Plot unit sphere
        ax.plot_surface(self.sphere_x, self.sphere_y, self.sphere_z, color="lightblue", alpha=0.3)

        # Normalize quaternions to ensure they're on the unit hypersphere
        q_demo_norms = np.sqrt(self.orientation_demo[:,0]**2 + self.orientation_demo[:,1]**2 + self.orientation_demo[:,2]**2 + self.orientation_demo[:,3]**2)
       
        qx_demo_norm = -self.orientation_demo[:,0] / q_demo_norms
        qy_demo_norm = -self.orientation_demo[:,1] / q_demo_norms
        qz_demo_norm = -self.orientation_demo[:,2] / q_demo_norms
            
        # Plot the trajectory line
        ax.plot(qx_demo_norm, qy_demo_norm, qz_demo_norm, 'k-', linewidth=2.0)

        # Draw reference unit vectors
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.5, length=0.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, length=0.5, normalize=True)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.5, length=0.5, normalize=True)
    
        # Normalize quaternions to ensure they're on the unit hypersphere
        q_norms = np.sqrt(self.orientation[:k+1,0]**2 + self.orientation[:k+1,1]**2 + self.orientation[:k+1,2]**2 + self.orientation[:k+1,3]**2)
        
        qx_norm = self.orientation[:k+1,0] / q_norms
        qy_norm = self.orientation[:k+1,1] / q_norms
        qz_norm = self.orientation[:k+1,2] / q_norms
            
        # Plot the trajectory line
        ax.plot(qx_norm, qy_norm, qz_norm, 'r-', linewidth=1.5)

    def _plot_6(self, ax, k):
        ax.clear()

        # Plot quaternion components over time
        time_indices = np.arange(k+1)
        
        # Plot quaternion components as trajectories
        ax.plot(time_indices, self.impedance_force[:k+1,0], 'r-', linewidth=3.5, label='Fx')
        ax.plot(time_indices, self.impedance_force[:k+1,1], 'g-', linewidth=3.5, label='Fy')  
        ax.plot(time_indices, self.impedance_force[:k+1,2], 'b-', linewidth=3.5, label='Fz')
        ax.plot(time_indices, self.impedance_force[:k+1,3], 'k-', linewidth=3.5, label='Mx')
        ax.plot(time_indices, self.impedance_force[:k+1,4], 'c-', linewidth=3.5, label='My')
        ax.plot(time_indices, self.impedance_force[:k+1,5], 'm-', linewidth=3.5, label='Mz')

        ax.set_xlabel('Time Index', fontsize=20)
        ax.set_ylabel('Impedance Force', fontsize=20)
        ax.set_title('Impedance Force Over Time: $K_{c} \Delta X + D_{c} \Delta \dot{X}$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend()
        # ax.grid(True)

    """#####################################################################################################################"""

    def _set_3d_plot_properties(self, ax, x_lim, y_lim, z_lim):
        # ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_zlim(z_lim[0], z_lim[1])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

    def show(self):
        plt.show()

if __name__ == "__main__":
    plotter = PlotData(demo_file_name="demo_for_APM", task_performed_name="dmp_performed_for_APM")
    
    # anim = plotter.setup_demo_plot()
    # plotter.show()

    anim = plotter.setup_performed_task_plot()
    plotter.show()