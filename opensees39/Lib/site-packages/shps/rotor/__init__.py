import numpy as np
"""
Quaternion.getVector()
Quaternion.getVersor()
Quaternion.getCayley()
Quaternion.getMatrix()
Quaternion.getEuler()

rm rmat_to_
rv rvec_to_
qu quat_to_
qv qvec_to_
cv cvec_to_
"""

Eye3 = np.eye(3)

def normalized(a, axis=-1, order=2):
    """
    Return normalized vector.

    :param a: vector
    :param order: normalization order
    :returns: normalized input
    """
    l2 = np.linalg.norm(a, ord=order)
    return a / l2


def hat(vector, **kwds):
    """Get the skew symmetric matrix of a rotational vector.

    Parameters
    ----------
    vector: numpy.ndarray
        the axial vector.

    Returns
    -------
    matrix:
        the skew symmetric matrix representing the rotational vector.

    """
    vector = np.atleast_1d(vector).flatten()

    return  np.array([[ 0.0,  -vector[2],  vector[1]],
                      [ vector[2],   0.0, -vector[0]],
                      [-vector[1], vector[0],   0.0]], **kwds)


def vee(matrix, **kwds):
    """Get the rotational vector representing a skew symmetric matrix.

    Parameters
    ----------
    matrix: numpy.ndarray
        the skew symmetric matrix.

    Returns
    -------
    vector: numpy.ndarray
        the axial vector.

    """

    return np.array([
                   matrix[2, 1] ,
                   matrix[0, 2] ,
                   matrix[1, 0] 
    ], **kwds)


def exp(axial):
    """Evaluate the exponential of a skew symmetric matrix given its axial vector.

    Parameters
    ----------
    axial: np.ndarray
        the rotational vector.

    Returns
    -------
    rotation_matrix:
        the rotation matrix.

    """
    Skw = hat(axial)
    angle = np.linalg.norm(axial)

    if angle == 0.0:
        return np.identity(3)

    else:
        return Eye3 + np.sin(angle)/angle * Skw + (1 - np.cos(angle)) / (angle ** 2) * Skw @ Skw


def log(matrix):
    """Evaluate the logarithm on SO3 and return the axial vector representation.

    Parameters
    ----------
    matrix: np.ndarray
        the rotation matrix.

    Returns
    -------
    axial:
        axial vector of the logarithm.

    """
    cosalpha = (np.trace(matrix) - 1) / 2

    # in initial state, cos(alpha) may equal to 1.000000002 
    # due to some numerical problems, which is not acceptable 
    # for arccos function. The error is here filtered for this reason.
    if np.abs(cosalpha - 1.0) < 1e-8:
        cosalpha = 1.0

    if np.abs(cosalpha + 1.0) < 1e-8:
        cosalpha = -1.0

    alpha = np.arccos(cosalpha)

    if alpha == 0.0:
        Skw = 1/2 * (matrix - matrix.T)

    else:
        Skw = alpha / \
            (2 * np.sin(alpha)) * (matrix - matrix.T)

    return vee(Skw)

def dlog(vector):
    """Compute the differential of the logarithm.

    Parameters
    ----------
    vector: numpy.ndarray
        the rotational vector.

    Returns
    -------
    numpy.ndarray:
        the differential of the logarithm.

    """
    angle = np.linalg.norm(vector)
    Skw = hat(vector)

    if angle == 0.:
        return np.identity(3)

    else:
        return ((angle / 2) / np.tan(angle / 2)) * Eye3 + (
            1 - (angle / 2) / np.tan(angle / 2))/angle**2 * vector@vector.T - 0.5 * Skw


def simo_dyn_linmap(Th: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(Th) / np.sqrt(2)

    eta, _ = ber(angle)

    if angle == 0:
        return np.identity(3)

    return Eye3 - 0.5*Th + eta * Th @ Th


def dexp(rotational_vector):
    """Compute the exponential differential.

    Parameters
    ----------
    rotational_vector: numpy.ndarray
        the rotation vector.

    Returns
    -------
    T_s: numpy.ndarray
        the exponential differential.

    """
    angle = np.linalg.norm(rotational_vector)
    Skw = hat(rotational_vector)

    if angle == 0.:
        return np.eye(3)

    else:
        return Eye3 + (1 - np.cos(angle)) / angle**2 * Skw + \
            (angle - np.sin(angle)) / angle**3 * Skw @ Skw


# ax2qu
def vector_to_versor(rv):
    angle = np.linalg.norm(rv)
    a = np.array(rv)
    if angle != 0.0:
        a = a / angle

    sn = np.sin(angle/2.0)
    cs = np.cos(angle/2.0)
    return np.array([
           a[0] * sn,
           a[1] * sn,
           a[2] * sn,
                  cs
    ])


def versor_to_vector(q):
    qn = normalized(q)
    qv = np.array(qn[:3])
    n  = np.linalg.norm(qv)
    rv = np.zeros(shape=(3))

    if n == 0.0:
        return np.zeros((3,1))

    angle = 2.0 * np.arctan2(n, qn[3])
    if angle > 1.8 * np.pi:
        angle = angle - 2.0*np.pi
    if angle < -1.8 * np.pi:
        angle = angle + 2.0*np.pi
    rv = angle * qv / n
    return rv


def matrix_to_versor(R):
    return spurrier(R)


def versor_to_matrix(q):
    qc = normalized(q)
    S  = hat(qc[:3])
    S2 = S @ S
    return Eye3 + 2.0*qc[3]*S + 2.0*S2


def spurrier(R):
    """
    Form a quaternion representation of the matrix R by Spurrier's algorithm.
    """
    tr = R[0,0] + R[1,1] + R[2,2]

    M = max(tr, R[0,0], R[1,1], R[2,2])
    q = np.zeros(shape=(4))

    if M == tr:
        q[3] = 0.5 * np.sqrt(1.0 + tr)
        q[0] = 0.25 *(R[2,1] - R[1,2]) / q[3]
        q[1] = 0.25 *(R[0,2] - R[2,0]) / q[3]
        q[2] = 0.25 *(R[1,0] - R[0,1]) / q[3]

    elif M == R[0,0]:
        q[0] = np.sqrt(0.5 * R[0,0] + 0.25*(1.0 - tr))
        q[3] = 0.25 *(R[2,1] - R[1,2]) / q[0]
        q[1] = 0.25 *(R[1,0] + R[0,1]) / q[0]
        q[2] = 0.25 *(R[2,0] + R[0,2]) / q[0]

    elif M == R[1,1]:
        q[1] = np.sqrt(0.5 * R[1,1] + 0.25*(1.0 - tr))
        q[3] = 0.25 *(R[0,2] - R[2,0]) / q[1]
        q[2] = 0.25 *(R[2,1] + R[1,2]) / q[1]
        q[0] = 0.25 *(R[0,1] + R[1,0]) / q[1]

    elif M == R[2,2]:
        q[2] = np.sqrt(0.5 * R[2,2] + 0.25*(1.0 - tr))
        q[3] = 0.25 *(R[1,0] - R[0,1]) / q[2]
        q[0] = 0.25 *(R[0,2] + R[2,0]) / q[2]
        q[1] = 0.25 *(R[1,2] + R[2,1]) / q[2]

    return q/np.linalg.norm(q)


def hamp(ql: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """
    quaternion product
    """
    q = np.zeros(shape=(4))
    q[0] = (ql[3]*qr[0] + ql[0]*qr[3] +
            ql[1]*qr[2] - ql[2]*qr[1])
    q[1] = (ql[3]*qr[1] - ql[0]*qr[2] +
            ql[1]*qr[3] + ql[2]*qr[0])
    q[2] = (ql[3]*qr[2] + ql[0]*qr[1] -
            ql[1]*qr[0] + ql[2]*qr[3])
    q[3] = (ql[3]*qr[3] - ql[0]*qr[0] -
            ql[1]*qr[1] - ql[2]*qr[2])

    return q


def conjugate_quat(q: np.ndarray) -> np.ndarray:
    c = np.array(q)
    c[:3] = -1.0 * c[:3]
    return c


def inverse_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q) ** 2
    i = conjugate_quat(q)
    return i / n


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector with quaternion
    """
    qv = np.zeros(4)
    qv[:3] = np.array(v)
    qv[3] = 0.0
    qinv = inverse_quat(q)
    return hamp(hamp(q, qv), qinv)[:3]


def rotate2(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Rotate a 3x3 matrix with quaternion
    """
    p = np.zeros_like(M)
    for i in range(3):
        p[:,i] = rotate(q, M[:,i])
    return p


def ber(angle):
    """Compute eta and mu.

    Parameters
    ----------
    angle: float
        the angle of the rotation.

    Returns
    -------
    The first coefficient eta: float.
    The second coefficient mu: float.

    """
    if angle == 0.:
        eta = 1/12 + angle**2/720 + angle**4/30240 + angle**6/1209600;
        mu  = 1/360

    else:
        sn  = np.sin(angle)
        eta = (2 * sn - angle * (1 + np.cos(angle))) / (2 * angle ** 2 * sn)
#       eta = (1.0 - 0.5*angle*tan(0.5*np.pi - 0.5*angle)) / angle**2

        mu  = (angle * (angle + sn) - 8 * np.sin(angle / 2) ** 2) / \
              (4 * angle ** 4 * np.sin(angle / 2) ** 2)

    return eta, mu

