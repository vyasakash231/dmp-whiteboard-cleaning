#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from .robot_RT_state import RT_STATE
from .robot_kinematic_model import Robot_KM
from .utils import *

class Robot(ABC):
    n = 6  # No of joints

    # Modified-DH Parameters (same as conventional/standard DH parameters)
    alpha = np.array([0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2])   
    a = np.array([0, 0, 0.409, 0, 0, 0])  # data from parameter data-sheet (in meters)
    d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])  # data from parameter data-sheet (in meters)

    # 195mm is for AG95 gripper 
    d_nn = np.array([[0.0], [0.0], [0.195]])  # TCP coord in end-effector frame (in meters)
    DH_params="modified"

    def __init__(self):
        self.kinematic_model = Robot_KM(self.n, self.alpha, self.a, self.d, self.d_nn, self.DH_params)

        self.is_rt_connected = False

        self.Robot_RT_State = RT_STATE()
        
        # Initialize RT control services
        self.initialize_rt_service_proxies()
        
        self.my_publisher = rospy.Publisher('/dsr01a0509/stop', RobotStop, queue_size=10)
        
        # Real-time data Publisher --> rospy.Publisher(topic_name, message_type, queue_size)
        self.speedj_publisher = rospy.Publisher('/dsr01a0509/speedj_rt_stream', SpeedJRTStream, queue_size=10)    # SpeedJRTStream -> Topic message that controls the joint velocity from an external controller.
        self.servoj_publisher = rospy.Publisher('/dsr01a0509/servoj_rt_stream', ServoJRTStream, queue_size=10)    # ServoJRTStream -> Topic message that controls the joint position from an external controller.   
        
        self.servol_publisher = rospy.Publisher('/dsr01a0509/servol_rt_stream', ServoLRTStream, queue_size=10)

        # For 3 kHz data, 100ms would be about 300 messages
        self.torque_publisher = rospy.Publisher('/dsr01a0509/torque_rt_stream', TorqueRTStream, queue_size=300)    # TorqueRTStream -> Topic message that controls the motor torque from an external controller.

        self.RT_observer_client = rospy.ServiceProxy('/dsr01a0509/realtime/read_data_rt', ReadDataRT)    # This function reads the real-time output data from the robot controller.

        self.read_rate = 3000  # in Hz (0.333 ms)
        self.write_rate = 1000  # in Hz (1 ms)

        self.read_thread = Thread(target=self.read_data_rt_client)
        self.read_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        self.read_thread.start()

        rospy.on_shutdown(self.cleanup)

    def initialize_rt_service_proxies(self):
        try:
            service_timeout = 3.0
            services = [
                ('/dsr01a0509/system/set_robot_mode', SetRobotMode),
                ('/dsr01a0509/realtime/connect_rt_control', ConnectRTControl),   # This service connects to robot controller via Real-time External Control.
                ('/dsr01a0509/realtime/set_rt_control_input', SetRTControlInput),    # This service set the input data (external controller →  robot controller) communication configuration supported by real-time external control.
                ('/dsr01a0509/realtime/set_rt_control_output', SetRTControlOutput),    # This service set the output data (robot controller →  external controller) communication configuration supported by real-time external control.
                ('/dsr01a0509/realtime/start_rt_control', StartRTControl),    # Starts sending/receiving the set input/output data.
                ('/dsr01a0509/realtime/stop_rt_control', StopRTControl),     # Finishes sending/receiving the set input/output data.
                ('/dsr01a0509/realtime/disconnect_rt_control', DisconnectRTControl),    # This service disconnects real-time external control.
            ]

            # Wait for all services with timeout
            for service_name, _ in services:
                try:
                    rospy.wait_for_service(service_name, timeout=service_timeout)
                except rospy.ROSException as e:
                    rospy.logerr(f"Service {service_name} not available: {e}")
                    raise

            # Create service proxies
            self.set_robot_mode = rospy.ServiceProxy(services[0][0], services[0][1])
            self.connect_rt_control = rospy.ServiceProxy(services[1][0], services[1][1])
            self.set_rt_control_input = rospy.ServiceProxy(services[2][0], services[2][1])
            self.set_rt_control_output = rospy.ServiceProxy(services[3][0], services[3][1])
            self.start_rt_control = rospy.ServiceProxy(services[4][0], services[4][1])
            self.stop_rt_control = rospy.ServiceProxy(services[5][0], services[5][1])
            self.disconnect_rt_control = rospy.ServiceProxy(services[6][0], services[6][1])

            self.joint_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velj_rt', SetVelJRT)   # The global joint speed is set in (deg/sec)
            self.joint_acc_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accj_rt', SetAccJRT)  # The global joint acceleration is set in (deg/sec^2)
            self.ee_vel_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_velx_rt', SetVelXRT)   # This function sets the global task velocity in (mm/sec, deg/s)
            self.ee_acc_limits = rospy.ServiceProxy('/dsr01a0509/realtime/set_accx_rt', SetAccXRT)

            self.connect_to_rt_control()

        except Exception as e:
            rospy.logerr(f"Failed to initialize RT control services: {e}")
            sys.exit(1)

    def connect_to_rt_control(self):
        try:
            mode_req = SetRobotModeRequest()
            mode_req.robot_mode = ROBOT_MODE_AUTONOMOUS
            robot_mode = self.set_robot_mode(mode_req)
            if not robot_mode.success:
                raise Exception("Failed to set robot mode")
            rospy.loginfo("Robot set to autonomous mode")

            connect_req = ConnectRTControlRequest()
            connect_req.ip_address = "192.168.137.100"
            connect_req.port = 12347   
            connect_response = self.connect_rt_control(connect_req)
            if not connect_response.success:
                raise Exception("Failed to connect RT control")
            
            set_output_req = SetRTControlOutputRequest()
            set_output_req.period = 0.005   # Communication Period (sec). Range: 0.001~1 [sec]
            set_output_req.loss = 4    # In succession, if the input data or the servo control command is lost due to over the set count, the real-time control connection is disconnected.
            set_output_response = self.set_rt_control_output(set_output_req)
            if not set_output_response.success:
                raise Exception("Failed to set RT control output")

            start_response = self.start_rt_control(StartRTControlRequest())
            if not start_response.success:
                raise Exception("Failed to start RT control")
                        
            self.is_rt_connected = True
            rospy.loginfo("Successfully connected to RT control")

        except Exception as e:
            rospy.logerr(f"Failed to establish RT control connection: {e}")
            self.cleanup()
            sys.exit(1)

    def cleanup(self):
        """Improved cleanup function with better error handling"""
        if self.shutdown_flag:  # Prevent multiple cleanup calls
            return
        
        self.shutdown_flag = True
        rospy.loginfo("Initiating cleanup process...")

        try:
            # Send stop command first
            stop_msg = RobotStop()
            stop_msg.stop_mode = 1  # STOP_TYPE_QUICK
            self.my_publisher.publish(stop_msg)
            rospy.sleep(0.1)  # Give time for stop command to process

            if self.is_rt_connected:
                try:
                    # Stop RT control with timeout
                    stop_future = self.stop_rt_control(StopRTControlRequest())
                    rospy.sleep(0.5)
                    
                    # Disconnect RT control
                    if not rospy.is_shutdown():  # Only try to disconnect if ROS isn't shutting down
                        self.disconnect_rt_control(DisconnectRTControlRequest())
                    
                    self.is_rt_connected = False
                    rospy.loginfo("RT control cleanup completed successfully")
                
                except (rospy.ServiceException, rospy.ROSException) as e:
                    rospy.logwarn(f"Non-critical error during cleanup: {e}")
                    # Continue cleanup process despite errors
        
        except Exception as e:
            rospy.logerr(f"Critical error during cleanup: {e}")
            
        finally:
            rospy.loginfo("Cleanup process finished")

    def read_data_rt_client(self):
        rate = rospy.Rate(self.read_rate)  # 3000 Hz

        while not rospy.is_shutdown() and not self.shutdown_flag:
            try:
                if not self.is_rt_connected:
                    rate.sleep()
                    continue

                request_data = ReadDataRTRequest()
                response_data = self.RT_observer_client(request_data)

                # Plot Data
                self.data = response_data.data
                self.plot_data()
                
                # Store Real-Time data
                self.Robot_RT_State.store_data(response_data.data)
                 
            except (rospy.ServiceException, rospy.ROSException) as e:
                if not self.shutdown_flag:  # Only log if we're not shutting down
                    rospy.logwarn(f"Service call failed: {e}") 
            rate.sleep()

    #@abstractmethod   # Force child classes to implement this method
    def plot_data(self):
        pass  

    def _svd_solve(self, A):
        return svd_solve(A)
    
    def _euler2mat(self, euler_angles):
        # return euler2mat(euler_angles)
        R_doosan = Rotation.from_euler("ZYZ", euler_angles, degrees=True).as_matrix()
        return R_doosan
    
    def _mat2quat(self, M):
        return mat2quat(M)
    
    def _eul2quat(self, euler_angles):
        # rmat = euler2mat(euler_angles)
        # M = np.asarray(rmat).astype(np.float32)
        # q = mat2quat(M)
        """Capital Letters - Extrinsic Rotations: The axes of rotation are fixed in the global frame of reference."""
        q = Rotation.from_euler("ZYZ", euler_angles, degrees=True).as_quat()  # [x,y,z,w]
        return q
    
    def _quat2eul(self, q):
        t1 = math.atan2(q[0], q[1])
        t2 = math.atan2(q[2], q[3])
        z1 = t2 - t1 
        y1 = 2*math.acos(math.sqrt(q[3]*q[3] + q[2]*q[2]))
        z2 = t2 + t1  
        eulerZYZ = [np.rad2deg(z1), np.rad2deg(y1), np.rad2deg(z2)]
        return eulerZYZ

    def _quat_slerp(self, q1, q2, fraction):
        return quat_slerp(q1, q2, fraction)