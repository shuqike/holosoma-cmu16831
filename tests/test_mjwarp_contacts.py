"""Guard the dependency's contact interpolation against world-scale cancellation."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
gjk = pytest.importorskip("mujoco_warp._src.collision_gjk")
pytestmark = pytest.mark.mujoco_warp


@wp.kernel
def affine_coordinates(points: wp.array2d[wp.vec3], result: wp.array[wp.vec3]):
    i = wp.tid()
    result[i] = gjk._tri_affine_coord(points[i, 0], points[i, 1], points[i, 2], points[i, 3])


def test_contact_affine_coordinates_preserve_partition_and_geometry():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required to reproduce the float32 contact arithmetic")
    rows = []
    expected = []
    # All dominant projections, reversed windings, interior and exterior points.
    # Float64 NumPy solve is the oracle for the exact float32 input vertices.
    for offset in (0.0, 36.8, -79.9, 119.9, 1000.0):
        for permutation in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for winding in ((0, 1, 2), (0, 2, 1)):
                for weights in ((0.2, 0.3, 0.5), (-0.2, 0.6, 0.6), (0.0, 1.0, 0.0)):
                    triangle = np.array([[0, 0, 0.04], [0.1, 0.1, 0], [0.1, 0, 0.005]])
                    triangle = (triangle[list(winding)][:, list(permutation)] + offset).astype(np.float32)
                    p = (np.asarray(weights) @ triangle.astype(np.float64)).astype(np.float32)
                    a, b, c = triangle.astype(np.float64)
                    normal = np.cross(b - a, c - a)
                    axes = [i for i in range(3) if i != np.argmax(np.abs(normal))]
                    uv = np.linalg.solve(np.stack((b - a, c - a), axis=1)[axes], (p - a)[axes])
                    rows.append(np.vstack((triangle, p)))
                    expected.append([1 - uv.sum(), *uv])
    output = wp.empty(len(rows), dtype=wp.vec3, device="cuda:0")
    wp.launch(affine_coordinates, len(rows), [wp.array(np.array(rows), dtype=wp.vec3, device="cuda:0"), output])
    actual = output.numpy()
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=2e-7, rtol=0)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@wp.kernel
def inflate_contacts(separations: wp.array[float], distances: wp.array[float], witnesses: wp.array2d[wp.vec3]):
    i = wp.tid()
    z = separations[i]
    result = gjk.GJKResult()
    result.x1 = wp.vec3(0.0, 0.0, 0.0)
    result.x2 = wp.vec3(0.0, 0.0, z)
    result.dist = z
    result.dim = 2
    result.simplex_index1 = wp.vec4i(-2, -3, 0, 0)
    terrain = gjk.Geom()
    prism = gjk.mat63()
    prism[0] = wp.vec3(-0.2, -0.2, -0.1)
    prism[1] = wp.vec3(0.2, -0.2, -0.1)
    prism[2] = wp.vec3(0.0, 0.2, -0.1)
    prism[3] = wp.vec3(-0.2, -0.2, 0.0)
    prism[4] = wp.vec3(0.2, -0.2, 0.0)
    prism[5] = wp.vec3(0.0, 0.2, 0.0)
    terrain.hfprism = prism
    capsule = gjk.Geom()
    capsule.pos = wp.vec3(0.0, 0.0, z + 0.1)
    capsule.rot = wp.identity(n=3, dtype=float)
    capsule.size = wp.vec3(0.0, 0.1, 0.0)  # GJK has already removed the radius.
    dist, x1, x2 = gjk._inflate(result, terrain, capsule, gjk.GeomType.HFIELD, gjk.GeomType.CAPSULE, 0.0, 0.008)
    distances[i] = dist
    witnesses[i, 0] = x1
    witnesses[i, 1] = x2


def test_hfield_inflation_preserves_signed_distance_and_closest_capsule_endpoint():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required")
    z = np.array([0.007, 0.008, 0.009], dtype=np.float32)
    distances = wp.empty(3, dtype=float, device="cuda:0")
    witnesses = wp.empty((3, 2), dtype=wp.vec3, device="cuda:0")
    wp.launch(inflate_contacts, 3, [wp.array(z, device="cuda:0"), distances, witnesses])
    np.testing.assert_allclose(distances.numpy(), z - 0.008, atol=1e-8, rtol=0)
    points = witnesses.numpy()
    np.testing.assert_allclose(points[:, 0], 0.0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(points[:, 1, 2], z - 0.008, atol=1e-8, rtol=0)


@wp.kernel
def epa_contact(pt: gjk.Polytope, distance: wp.array[float], witnesses: wp.array[wp.vec3]):
    terrain = gjk.Geom()
    prism = gjk.mat63()
    prism[3] = pt.vert[0]
    prism[4] = pt.vert[2]
    prism[5] = pt.vert[4]
    terrain.hfprism = prism
    capsule = gjk.Geom()
    capsule.pos = wp.vec3(0.0, 0.0, 0.298)
    capsule.rot = wp.identity(n=3, dtype=float)
    capsule.size = wp.vec3(0.008, 0.1, 0.0)
    x1, x2, depth = gjk._epa_witness(pt, terrain, capsule, gjk.GeomType.HFIELD, gjk.GeomType.CAPSULE, 0)
    distance[0] = depth
    witnesses[0] = x1
    witnesses[1] = x2


def test_hfield_epa_preserves_converged_face_instead_of_reselecting_capsule_support():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required")
    # A horizontal top face at z=.2, with the capsule's bottom at z=.19.
    # EPA's closest Minkowski face is z=.01, so penetration is exactly .01.
    pt = gjk.Polytope()
    pt.vert = wp.array(
        [[-0.2, -0.2, 0.2], [0, 0, 0.19], [0.2, -0.2, 0.2], [0, 0, 0.19], [0, 0.2, 0.2], [0, 0, 0.19]],
        dtype=wp.vec3,
        device="cuda:0",
    )
    pt.vert_index = wp.array([-1, -1, -2, -1, -3, -1], dtype=int, device="cuda:0")
    pt.face = wp.array([(1 << 10) | (2 << 20)], dtype=int, device="cuda:0")
    pt.face_pr = wp.array([[0, 0, 0.01]], dtype=wp.vec3, device="cuda:0")
    pt.face_norm2 = wp.array([0.0001], dtype=float, device="cuda:0")
    pt.nvert = 3
    pt.nface = 1
    distance = wp.empty(1, dtype=float, device="cuda:0")
    witnesses = wp.empty(2, dtype=wp.vec3, device="cuda:0")
    wp.launch(epa_contact, 1, [pt, distance, witnesses])
    np.testing.assert_allclose(distance.numpy(), [-0.01], atol=1e-7, rtol=0)
    np.testing.assert_allclose(witnesses.numpy(), [[0, 0, 0.2], [0, 0, 0.19]], atol=1e-7, rtol=0)
