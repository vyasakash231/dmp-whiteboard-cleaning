import math
import numpy as np

def euler2mat(euler_angles):  # euler_angles in degrees
    z1_angle, y_angle, z2_angle = np.radians(euler_angles)  # convert deg to rad

    # Rotation matrices for individual axes
    Rz1 = np.array([
        [math.cos(z1_angle), -math.sin(z1_angle), 0],
        [math.sin(z1_angle), math.cos(z1_angle), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [math.cos(y_angle), 0, math.sin(y_angle)],
        [0, 1, 0],
        [-math.sin(y_angle), 0, math.cos(y_angle)]
    ])
    
    Rz2 = np.array([
        [math.cos(z2_angle), -math.sin(z2_angle), 0],
        [math.sin(z2_angle), math.cos(z2_angle), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations in ZYZ order
    """
    * The rotation order (Z1 * Y * Z2) is typically referred to as the "intrinsic" ZYZ rotation sequence
    * The rotation order (Z2 * Y * Z1) is typically referred to as the "extrinsic" ZYZ rotation sequence

    The key difference is that intrinsic rotations are performed relative to the object's current orientation, 
    while extrinsic rotations are performed relative to the fixed global coordinate system.
    """
    R = Rz1 @ Ry @ Rz2
    return R

def mat2quat(rmat):
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    # symmetric matrix K
    K = np.array([
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
                ])
    K /= 3.0

    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]

def eul2quat(euler_angles):
    rmat = euler2mat(euler_angles)
    M = np.asarray(rmat).astype(np.float32)
    q = mat2quat(M)
    return q.tolist()  # (x,y,z,w)

def make_quat_continuity(quats):
    for i in range(1, quats.shape[0]):
        if np.dot(quats[i-1,:], quats[i,:]) < 0:  # Angle > 90 degrees
            quats[i,:] = -quats[i,:]  # Flip to maintain continuity
    return quats

def quat2mat(quaternion):
    x, y, z, w = quaternion
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-10:
        return np.identity(3)
    
    quaternion /= norm
    x, y, z, w = quaternion
    
    # Form rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def mat2euler(R):
    # Handle special case when R[2,2] is close to 1 (gimbal lock)
    if abs(R[2, 2]) > 0.9999:
        # Gimbal lock case
        z1 = math.atan2(R[0, 1], R[0, 0])
        y = 0.0
        z2 = 0.0  # arbitrary, we set to 0
    else:
        # General case
        z1 = math.atan2(R[1, 2], R[0, 2])
        y = math.acos(R[2, 2])
        z2 = math.atan2(R[2, 1], -R[2, 0])
    
    # Convert to degrees
    euler_angles = np.array([z1, y, z2]) * 180.0 / math.pi
    return euler_angles

def quat2euler(quaternion):
    R = quat2mat(quaternion)
    euler = mat2euler(R)
    return euler