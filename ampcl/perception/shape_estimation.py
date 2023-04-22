import cv2
import numpy as np
import open3d as o3d


class ConvexHullModel:
    def __init__(self):
        self.footprint = None
        self.centroid = None
        self.dimension_h = 0

    def estimate(self, cluster):

        self.centroid = np.mean(cluster, axis=0)
        max_z = np.max(cluster[:, 2], axis=0)
        min_z = np.min(cluster[:, 2], axis=0)
        self.dimension_h = max_z - min_z

        # only use 2D information
        cluster = cluster[:, :2]
        self.footprint = cv2.convexHull(cluster.astype(np.float32)).squeeze()

    def generate_centroid_o3d(self, radius=0.2, color=(1, 0, 0)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        sphere.translate(self.centroid)
        sphere.paint_uniform_color(color)
        return sphere

    def generate_polygon_o3d(self, edge_color=(1, 0, 0)):
        """
        according to the footprint and centroid, generate a 3D-polygon (open3d object)
        :param edge_color:
        :return:
        """
        z_upper_boundary = np.zeros((self.footprint.shape[0]), dtype=np.float32) \
                           + self.dimension_h / 2 + self.centroid[2]
        z_lower_boundary = np.zeros((self.footprint.shape[0]), dtype=np.float32) - self.dimension_h / 2 \
                           + self.centroid[2]

        z_upper_boundary = np.hstack((self.footprint, z_upper_boundary[:, np.newaxis]))
        z_lower_boundary = np.hstack((self.footprint, z_lower_boundary[:, np.newaxis]))

        points = np.vstack((z_upper_boundary, z_lower_boundary))

        # build line
        lines = []
        for i in range(self.footprint.shape[0] - 1):
            lines.append([i, (i + 1)])
        lines.append([self.footprint.shape[0] - 1, 0])

        for i in range(self.footprint.shape[0], points.shape[0] - 1):
            lines.append([i, (i + 1)])
        lines.append([points.shape[0] - 1, self.footprint.shape[0]])

        for i in range(self.footprint.shape[0] - 1):
            lines.append([i, (i + self.footprint.shape[0])])

        # equal as: colors = [color for _ in range(len(lines))]
        colors = np.expand_dims(edge_color, axis=0).repeat(len(lines), axis=0)

        # build o3d line
        polygon_o3d = o3d.geometry.LineSet()
        polygon_o3d.lines = o3d.utility.Vector2iVector(lines)
        polygon_o3d.colors = o3d.utility.Vector3dVector(colors)
        polygon_o3d.points = o3d.utility.Vector3dVector(points)

        return polygon_o3d
