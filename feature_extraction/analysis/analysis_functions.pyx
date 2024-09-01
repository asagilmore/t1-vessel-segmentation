import numpy as np
cimport numpy as cnp
from numpy.core.multiarray import ndarray
from scipy.ndimage import convolve
from libc.math cimport sqrt
from skimage.morphology import skeletonize

def _get_edge_array(binary_map):
    kernel = np.ones((3,3,3)).astype(int)
    kernel[1,1,1] = 0
    kernel[0,0,0] = 0
    kernel[0,0,2] = 0
    kernel[0,2,2] = 0
    kernel[2,2,2] = 0
    kernel[0,2,0] = 0
    kernel[2,2,0] = 0
    kernel[2,0,0] = 0
    kernel[2,0,2] = 0

    neighbor_count = convolve(binary_map.astype(int), kernel, mode="constant", cval=1)

    if neighbor_count.shape != binary_map.shape:
        raise ValueError("something went wrong here my bad")


    edge_array = neighbor_count < 18

    return edge_array & binary_map.astype(bool)

cdef class Point:
    cdef public int x
    cdef public int y
    cdef public int z
    cdef public float distance

cdef _generate_distance_lut(int max_distance):
    cdef int x, y, z
    cdef float distance
    cdef list distance_lut = []
    cdef Point point

    for x in range(0, max_distance+1):
        for y in range(0, max_distance+1):
            for z in range(0, max_distance+1):
                distance = sqrt(x**2 + y**2 + z**2)
                point = Point()
                point.x = x
                point.y = y
                point.z = z
                point.distance = distance
                distance_lut.append(point)

    # Sort the list by distance
    distance_lut.sort(key=lambda p: p.distance)

    return distance_lut


def get_distance_map(binary_map, max_distance):
    edge_array = _get_edge_array(binary_map)
    skeleton = skeletonize(binary_map)
    skeleton_points = np.nonzero(skeleton)
    skeleton_points = np.vstack(skeleton_points).T.astype(np.uint32)
    print(f'{len(skeleton_points)} skeleton points')

    return _get_distance_map(skeleton_points, edge_array, max_distance)


cdef float _get_distance(int px, int py, int pz, cnp.ndarray[cnp.uint8_t, ndim=3] edge_array, list distance_lut):
    cdef int dim1, dim2, dim3
    cdef int x, y, z
    dim1 = edge_array.shape[0]
    dim2 = edge_array.shape[1]
    dim3 = edge_array.shape[2]

    for point in distance_lut:
        for sign_x in [-1, 1]:
            for sign_y in [-1, 1]:
                for sign_z in [-1, 1]:
                    x = px + sign_x * point.x
                    y = py + sign_y * point.y
                    z = pz + sign_z * point.z
                    if x >= 0 and x < dim1 and y >= 0 and y < dim2 and z >= 0 and z < dim3 and edge_array[x, y, z] > 0:
                        return (point.distance * 2) + 1.0 # add 1 because points with 1 pixel width have distance 1
    return -1.0


cdef _get_distance_map(cnp.ndarray[cnp.uint32_t, ndim=2] points, cnp.ndarray[cnp.uint8_t, ndim=3] edge_array, int max_distance):
    cdef:
        Py_ssize_t rows = edge_array.shape[0]
        Py_ssize_t cols = edge_array.shape[1]
        Py_ssize_t slices = edge_array.shape[2]
    cdef cnp.ndarray[cnp.float32_t, ndim=3] distance_map
    distance_map = np.zeros((rows, cols, slices), dtype=np.float32)

    cdef list distance_lut = _generate_distance_lut(max_distance)
    cdef int i

    for i in range(points.shape[0]):
        distance_map[points[i, 0], points[i, 1], points[i, 2]] = _get_distance(points[i, 0], points[i, 1], points[i, 2], edge_array, distance_lut)

    return distance_map