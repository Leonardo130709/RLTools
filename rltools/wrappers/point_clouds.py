from .base import Wrapper
import numpy as np
from dm_env import specs
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums


class PointCloudWrapper(Wrapper):
    def __init__(self, env, pn_number=1000, render_kwargs=None,
                 static_camera=True, as_pixels=False, downsample=1):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=240, width=320)
        assert all(map(lambda k: k in self.render_kwargs, ('camera_id', 'height', 'width')))
        self.pn_number = pn_number

        self.scene_option = wrapper.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_STATIC] = 0

        self.static_camera = static_camera
        self.as_pixels = as_pixels
        self.downsample = downsample
        self._partial_sum = None
        self._inverse_matrix = self.inverse_matrix() if static_camera else None

    def observation(self, timestamp):
        depth_map = self.physics.render(depth=True, **self.render_kwargs,
                                        scene_option=self.scene_option)
        point_cloud = self._get_point_cloud(depth_map)
        if self.as_pixels:
            # TODO: decide if segmentation or another mask is needed
            return point_cloud
        point_cloud = np.reshape(point_cloud, (-1, 3))
        segmentation_mask = self._segmentation_mask()
        # TODO: fix orientation so mask can be used
        mask = self._mask(point_cloud)  # additional mask if needed
        selected_points = point_cloud[segmentation_mask & mask][::self.downsample]
        return self._to_fixed_number(selected_points).astype(np.float32)

    def inverse_matrix(self):
        # one could reuse the matrix if a camera remains static
        cam_id, height, width = map(self.render_kwargs.get, ('camera_id', 'height', 'width'))
        fov = self.physics.model.cam_fovy[cam_id]
        rotation = self.physics.data.cam_xmat[cam_id].reshape(3, 3)
        cx = (width - 1)/2.
        cy = (height - 1)/2.
        f_inv = 2.*np.tan(np.deg2rad(fov)/2.)/height
        inv_mat = np.array([
            [f_inv, 0, -cx*f_inv],
            [0, f_inv, -f_inv*cy],
            [0, 0, 1.]
        ])
        return rotation.T@inv_mat

    def _segmentation_mask(self):
        seg = self.physics.render(segmentation=True, **self.render_kwargs,
                                  scene_option=self.scene_option)
        model_id, obj_type = np.split(seg, 2, -1)
        return (obj_type != -1).flatten()

    def _to_fixed_number(self, pc):
        n = len(pc)
        if n < self.pn_number:
            return np.pad(pc, ((0, self.pn_number - n), (0, 0)), mode='edge')
        else:
            return np.random.permutation(pc)[:self.pn_number]

    def _get_point_cloud(self, depth_map):
        # cam_id = self.render_kwargs['camera_id']
        inv_mat = self._inverse_matrix if self.static_camera else self.inverse_matrix()

        if not self.static_camera or self._partial_sum is None:
            width = self.render_kwargs.get('width', 320)
            height = self.render_kwargs.get('height', 240)
            grid = 1. + np.mgrid[:height, :width]
            self._partial_sum = self.dot_product(inv_mat[:, :-1], grid)

        residual_sum = self.dot_product(inv_mat[:, -1:], depth_map[np.newaxis])
        return self._partial_sum + residual_sum

    def _mask(self, point_cloud):
        """ Heuristic to cut outliers """
        # threshold = np.quantile(point_cloud[..., 2], .99)
        return point_cloud[..., 2] < 10.

    def observation_spec(self):
        return specs.Array(shape=(self.pn_number, 3), dtype=np.float32, name='point_cloud')

    @staticmethod
    def _dot_product(x, y):
        return np.einsum('ij, jhw->hwi', x, y)
