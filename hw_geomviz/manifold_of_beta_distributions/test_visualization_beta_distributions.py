"""Unit tests for visualization."""

matplotlib.use("Agg")  # NOQA
import random
import matplotlib
import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
import tests.conftest

from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import (
    SpecialOrthogonal,
    _SpecialOrthogonalMatrices,
)


from geomstats.information_geometry.beta import BetaDistributions

from hw_geomviz.manifold_of_beta_distributions.Beta_Visualization import Beta




class TestVisualizationBeta(tests.conftest.TestCase):
    def setup_method(self):
        self.n_samples = 10
        self.Beta = BetaDistributions()
        self.beta_viz = Beta()

    def test_plot_beta(self):
        points = gs.random.rand(2,2)
        self.beta_viz.plot(points)

    def test_scatter_beta(self):
        num_points = random.randint(2,50)
        points = gs.random.rand(num_points,2)
        self.beta_viz.plot(points)


    @staticmethod
    def test_tutorial_matplotlib():
        visualization.tutorial_matplotlib()

    def test_plot_points_so3(self):
        points = self.SO3_GROUP.random_uniform(self.n_samples)
        visualization.plot(points, space="SO3_GROUP")

    def test_plot_points_se3(self):
        points = self.SE3_GROUP.random_point(self.n_samples)
        visualization.plot(points, space="SE3_GROUP")

    def test_draw_pre_shape_2d(self):
        self.KS.draw()

    def test_draw_points_pre_shape_2d(self):
        points = self.S32.random_point(self.n_samples)
        visualization.plot(points, space="S32")
        points = self.M32.random_point(self.n_samples)
        visualization.plot(points, space="M32")
        self.KS.clear_points()

    def test_draw_curve_pre_shape_2d(self):
        self.KS.draw()
        base_point = self.S32.random_point()
        vec = self.S32.random_point()
        tangent_vec = self.S32.to_tangent(vec, base_point)
        times = gs.linspace(0.0, 1.0, 1000)
        speeds = gs.array([-t * tangent_vec for t in times])
        points = self.S32.total_space_metric.exp(speeds, base_point)
        self.KS.add_points(points)
        self.KS.draw_curve()
        self.KS.clear_points()

    def test_draw_vector_pre_shape_2d(self):
        self.KS.draw()
        base_point = self.S32.random_point()
        vec = self.S32.random_point()
        tangent_vec = self.S32.to_tangent(vec, base_point)
        self.KS.draw_vector(tangent_vec, base_point)

    def test_convert_to_spherical_coordinates_pre_shape_2d(self):
        points = self.S32.random_point(self.n_samples)
        coords = self.KS.convert_to_spherical_coordinates(points)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        result = x**2 + y**2 + z**2
        expected = 0.25 * gs.ones(self.n_samples)
        self.assertAllClose(result, expected)

    def test_rotation_pre_shape_2d(self):
        theta = gs.random.rand(1)[0]
        phi = gs.random.rand(1)[0]
        rot = self.KS.rotation(theta, phi)
        result = _SpecialOrthogonalMatrices(3).belongs(rot)
        expected = True
        self.assertAllClose(result, expected)

    def test_draw_pre_shape_3d(self):
        self.KD.draw()

    def test_draw_points_pre_shape_3d(self):
        points = self.S33.random_point(self.n_samples)
        visualization.plot(points, space="S33")
        points = self.M33.random_point(self.n_samples)
        visualization.plot(points, space="M33")
        self.KD.clear_points()

    def test_draw_curve_pre_shape_3d(self):
        self.KD.draw()
        base_point = self.S33.random_point()
        vec = self.S33.random_point()
        tangent_vec = self.S33.to_tangent(vec, base_point)
        tangent_vec = 0.5 * tangent_vec / self.S33.total_space_metric.norm(tangent_vec)
        times = gs.linspace(0.0, 1.0, 1000)
        speeds = gs.array([-t * tangent_vec for t in times])
        points = self.S33.total_space_metric.exp(speeds, base_point)
        self.KD.add_points(points)
        self.KD.draw_curve()
        self.KD.clear_points()

    def test_draw_vector_pre_shape_3d(self):
        self.KS.draw()
        base_point = self.S32.random_point()
        vec = self.S32.random_point()
        tangent_vec = self.S32.to_tangent(vec, base_point)
        self.KS.draw_vector(tangent_vec, base_point)

    def test_convert_to_planar_coordinates_pre_shape_3d(self):
        points = self.S33.random_point(self.n_samples)
        coords = self.KD.convert_to_planar_coordinates(points)
        x = coords[:, 0]
        y = coords[:, 1]
        radius = x**2 + y**2
        result = [r <= 1.0 for r in radius]
        self.assertTrue(gs.all(result))

    def test_plot_points_s1(self):
        points = self.S1.random_uniform(self.n_samples)
        visualization.plot(points, space="S1")

    def test_plot_points_s2(self):
        points = self.S2.random_uniform(self.n_samples)
        visualization.plot(points, space="S2")

    def test_plot_points_h2_poincare_disk(self):
        points = self.H2.random_point(self.n_samples)
        visualization.plot(points, space="H2_poincare_disk")

    def test_plot_points_h2_poincare_half_plane_ext(self):
        points = self.H2.random_point(self.n_samples)
        visualization.plot(
            points, space="H2_poincare_half_plane", coords_type="extrinsic"
        )

    def test_plot_points_h2_poincare_half_plane_none(self):
        points = self.H2_half_plane.random_point(self.n_samples)
        visualization.plot(points, space="H2_poincare_half_plane")

    def test_plot_points_h2_poincare_half_plane_hs(self):
        points = self.H2_half_plane.random_point(self.n_samples)
        visualization.plot(
            points, space="H2_poincare_half_plane", coords_type="half_space"
        )

    def test_plot_points_h2_klein_disk(self):
        points = self.H2.random_point(self.n_samples)
        visualization.plot(points, space="H2_klein_disk")

    @staticmethod
    def test_plot_points_se2():
        points = SpecialEuclidean(n=2, point_type="vector").random_point(4)
        visu = visualization.SpecialEuclidean2(points, point_type="vector")
        ax = visu.set_ax()
        visu.draw_points(ax)

    def test_plot_points_spd2(self):
        one_point = self.spd.random_point()
        visualization.plot(one_point, space="SPD2")

        points = self.spd.random_point(4)
        visualization.plot(points, space="SPD2")

    def test_compute_coordinates_spd2(self):
        point = gs.eye(2)
        ellipsis = visualization.Ellipses(k_sampling_points=4)
        x, y = ellipsis.compute_coordinates(point)
        self.assertAllClose(x, gs.array([1, 0, -1, 0, 1]))
        self.assertAllClose(y, gs.array([0, 1, 0, -1, 0]))

    @staticmethod
    def teardown_method():
        plt.close()
