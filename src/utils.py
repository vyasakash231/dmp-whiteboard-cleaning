from math import *
import numpy as np

EPS = np.finfo(float).eps * 4.0


def rk4_step(Yn, A, B, dt):
    def f(Y):
        return (A @ Y) + B

    k1 = f(Yn)
    k2 = f(Yn + 0.5 * dt * k1)
    k3 = f(Yn + 0.5 * dt * k2)
    k4 = f(Yn + dt * k3)
    
    dY = (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return dY

def forward_euler(Yn, A, B, dt):
    def f(Y):
        return (A @ Y) + B
    return f(Yn)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def weight_Func(m,n,q_range,q,epsilon):
    const = 400
    We = np.zeros((m,m))
    for i in range(0,m):
        We[i,i] = 50

    Wc = np.zeros((n,n))
    for i in range(0,n):
        if q[i] < q_range[i,0]:
            Wc[i,i] = const
        elif q_range[i,0] <= q[i] <= (q_range[i,0] + epsilon[i]):
            Wc[i,i] = (const/2)*(1 + cos(pi*((q[i] - q_range[i,0])/epsilon[i])))
        elif (q_range[i,0] + epsilon[i]) < q[i] < (q_range[i,1] - epsilon[i]):
            Wc[i,i] = 0
        elif (q_range[i,1] - epsilon[i]) <= q[i] <= q_range[i,1]:
            Wc[i,i] = (const/2)*(1 + cos(pi*((q_range[i,1] - q[i])/epsilon[i])))
        else:
            Wc[i,i] = const

    Wv = np.zeros((n,n))
    for i in range(0,n):
        Wv[i,i] = 0.5
    return We, Wc, Wv

def cost_func(n,K,q,q_range,m):
    # Initiate
    c = np.zeros((n,))
    b = np.zeros((n,))
    del_phi_del_q = np.zeros((n,1))
    q_c = np.mean(q_range,axis = 1); # column vector containing the mean of each row
    del_q = q_range[:,1] - q_range[:,0]; # Total working range of each joint

    for i in range(0,n):
        if q[i] >= q_c[i]:
            c[i] = pow((K[i,i]*((q[i] - q_c[i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q[i] - q_c[i])/del_q[i])),m-1)
        elif q[i] < q_c[i]:
            c[i] = pow((K[i,i]*((q_c[i] - q[i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q_c[i] - q[i])/del_q[i])),(m-1))

    L = np.sum(c)

    for j in range(0,n):
        if q[j] >= q_c[j]:
            del_phi_del_q[j] = pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])
        elif q[j] < q_c[j]:
            del_phi_del_q[j] = -pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])

    v = -del_phi_del_q
    return v

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def svd_solve(A):
    U, s, V_transp = np.linalg.svd(A)

    # Option-1
    S_inv = np.diag(s**-1)
    
    # Option-2, Handle small singular values
    # s_inv = np.zeros_like(s)
    # for i in range(len(s)):
    #     if s[i] > threshold:
    #         s_inv[i] = 1.0 / s[i]
    #     else:
    #         s_inv[i] = 0.0  # Or apply damping: s[i]/(s[i]^2 + lambda^2)
    
    # # Reconstruct inverse
    # S_inv = np.zeros_like(M)
    # for i in range(len(s)):
    #     S_inv[i,i] = s_inv[i]

    # A^-1 = V * S^-1 * U^T
    A_inv = V_transp.T @ S_inv @ U.T   # V = V_transp.T
    return A_inv

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def skew4x4(z):
    mat = np.zeros((4,4))

    mat[0,1:] = -z  # [-ωx,-ωy,-ωz]
    mat[1:,0] = z  # [ωx,ωy,ωz]
    mat[1,2] = z[2]  # ωz
    mat[1,3] = -z[1]  # -ωy
    mat[2,1] = -z[2]  # -ωz
    mat[2,3] = z[0]  # ωx
    mat[3,1] = z[1]  # ωy
    mat[3,2] = -z[0]  # -ωx
    return mat

def skew3x3(z):
    mat = np.zeros((3,3))

    if z.ndim == 2:
        z = z.reshape(-1)

    mat[0,1] = -z[2]  # ωz
    mat[0,2] = z[1]  # -ωy
    mat[1,0] = z[2]  # -ωz
    mat[1,2] = -z[0]  # ωx
    mat[2,0] = -z[1]  # ωy
    mat[2,1] = z[0]  # -ωx
    return mat

def euler2mat(euler_angles):  # euler_angles in degrees
    """
    Convert Euler ZYZ rotation angles to a 3D rotation matrix.
    
    Args:
    z1_angle (float): First rotation angle around Z-axis in radians
    y_angle (float): Rotation angle around Y-axis in radians
    z2_angle (float): Second rotation angle around Z-axis in radians
    
    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    z1_angle, y_angle, z2_angle = np.radians(euler_angles)

    # Rotation matrices for individual axes
    Rz1 = np.array([
        [cos(z1_angle), -sin(z1_angle), 0],
        [sin(z1_angle), cos(z1_angle), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [cos(y_angle), 0, sin(y_angle)],
        [0, 1, 0],
        [-sin(y_angle), 0, cos(y_angle)]
    ])
    
    Rz2 = np.array([
        [cos(z2_angle), -sin(z2_angle), 0],
        [sin(z2_angle), cos(z2_angle), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations in ZYZ order
    """
    * The rotation order (Z1 * Y * Z2) is typically referred to as the "intrinsic" ZYZ rotation sequence
    * The rotation order (Z2 * Y * Z1) is typically referred to as the "extrinsic" ZYZ rotation sequence

    The key difference is that intrinsic rotations are performed relative to the object's current orientation, 
    while extrinsic rotations are performed relative to the fixed global coordinate system.
    """
    R = Rz1 @ (Ry @ Rz2)
    return R

def mat2quat(rmat):
    M = np.asarray(rmat).astype(np.float64)[:3, :3]

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
                [m00 - m11 - m22, np.float64(0.0), np.float64(0.0), np.float64(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float64(0.0), np.float64(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float64(0.0)],
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
    M = np.asarray(rmat).astype(np.float64)
    q = mat2quat(M)
    return q

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    EPS = np.finfo(float).eps * 4.0
    
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    
    d = np.dot(q0, q1)
    
    if abs(abs(d) - 1.0) < EPS:
        return q0
    
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = acos(np.clip(d, -1, 1))
    
    if abs(angle) < EPS:
        return q0
    
    isin = 1.0 / sin(angle)
    q0 *= sin((1.0 - fraction) * angle) * isin
    q1 *= sin(fraction * angle) * isin
    q0 += q1
    return q0
    