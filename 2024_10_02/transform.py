import numpy as np
from typing import List, Tuple
import numpy as np
from PIL import Image
import pandas as pd
from typing import List, Tuple
import pandas as pd


class PerspectiveTransform:
    def __init__(
        self,
        src: List[Tuple[float, float]],
        dst: List[Tuple[float, float]],
    ) -> None:
        self.M = self._get_perspective_projection_matrix(src, dst)

    def _get_perspective_projection_matrix(
        self,
        src: List[Tuple[float, float]],
        dst: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Each input is a 4x2 matrix of (x,y) coordinates of the corners of a rectangle.
        """
        # Convert the input to numpy arrays
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)

        assert src.shape == dst.shape, "dimension mismatch"
        assert src.shape[0] == 4, "dimension mismatch"

        # Construct the homogeneous system of equations
        A = np.zeros((8, 8))
        b = np.zeros((8, 1))

        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            A[i * 2] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            A[i * 2 + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            b[i * 2] = u
            b[i * 2 + 1] = v

        # Solve the system of equations using least squares
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        x = np.append(x, 1)

        # Reshape the solution to a 3x3 matrix
        M = x.reshape((3, 3))

        return M

    def transform(
        self,
        x: List[float],
        y: List[float],
        decimals: int = 8,
    ) -> np.ndarray:
        """
        Applies the perspective transformation defined by the 3x3 matrix self.M to a set of points.

        Parameters
        ----------
        - points (np.ndarray): A Nx2 array of N points in Cartesian coordinates, with shape (N, 2). # 要是左下, 右下, 右上, 左上
        - decimals (int): Number of decimal places to round the transformed points to (default: 8).

        Returns
        -------
        - transformed_points (np.ndarray): A Nx2 array of N transformed points in Cartesian coordinates,
        with shape (N, 2).

        Example:
        >>> src = [[296.8, 658.2], [988.6, 659], [843.6, 274], [438.6, 273.2]]
        >>> dst = [[0, 0], [61, 0], [61, 134], [0, 134]]
        >>> transformer = PerspectiveTransform(src, dst)
        >>> x = [100, 150, 200, 250]
        >>> y = [100, 200, 150, 250]
        >>> transformed_points = transformer.transform(x, y)
        >>> print(transformed_points)
            [[-88.53033602 284.00775591]
            [-54.72805006 183.77898745]
            [-55.11171215 228.12637239]
            [-30.93211887 148.15418828]]
        """
        assert len(x) == len(y), "dimension mismatch"

        # Convert the input to a numpy array
        points = np.array([x, y]).T
        points = np.array(points, dtype=np.float32)

        # Check that the input has the correct shape
        assert points.shape[1] == 2, "dimension mismatch"

        # Add a column of ones to the points to turn them into homogeneous coordinates
        points_homogeneous = np.hstack((points, np.ones((len(points), 1))))

        # Apply the perspective transformation to the points
        transformed_points_homogeneous = np.dot(self.M, points_homogeneous.T)

        # Convert the transformed points back to Cartesian coordinates
        transformed_points = (
            transformed_points_homogeneous[:2] / transformed_points_homogeneous[2]
        ).T
        transformed_points[abs(transformed_points) < 10**-decimals] = 0.0
        return np.around(transformed_points, decimals)


if __name__ == "__main__":
    A_x, B_x, C_x, D_x, A_y, B_y, C_y, D_y = [], [], [], [], [], [], [], []
    H_x, H_y = [], []
    match = pd.read_csv("./match.csv")
    set = pd.read_csv("./set.csv")
    rally = pd.read_csv("./rally.csv")
    shot = pd.read_csv("./shot.csv")
    rtns = []
    for idx, row in match.iterrows():
        M = [
            [row["upleft_x"], row["upleft_y"]],
            [row["upright_x"], row["upright_y"]],
            [row["downleft_x"], row["downleft_y"]],
            [row["downright_x"], row["downright_y"]],
        ]
        source_points = np.float32(M)
        # 定義變換後圖像上相應的四個點的座標
        target_points = np.float32(
            [
                [0, 134],
                [61, 134],
                [0, 0],
                [61, 0],
            ]
        )

        transformer = PerspectiveTransform(source_points, target_points)
        set_list = set[set["match_id"].eq(row["match_id"])]["set_id"].tolist()

        rally_list = rally[rally["set_id"].isin(set_list)]["rally_id"].tolist()
        shot_list = shot[shot["rally_id"].isin(rally_list)]["shot_id"].tolist()

        shot_ = shot[shot["shot_id"].isin(shot_list)]
        player_A_x, player_A_y = list(shot_["player_A_x"]), list(shot_["player_A_y"])
        player_B_x, player_B_y = list(shot_["player_B_x"]), list(shot_["player_B_y"])
        player_C_x, player_C_y = list(shot_["player_C_x"]), list(shot_["player_C_y"])
        player_D_x, player_D_y = list(shot_["player_D_x"]), list(shot_["player_D_y"])
        hit_x, hit_y = list(shot_["hit_x"]), list(shot_["hit_y"])
        return_x, return_y = list(shot_["return_x"]), list(shot_["return_y"])
        A = transformer.transform(player_A_x, player_A_y)
        player_A_x, player_A_y = A[:, 0], A[:, 1]
        B = transformer.transform(player_B_x, player_B_y)
        player_B_x, player_B_y = B[:, 0], B[:, 1]
        C = transformer.transform(player_C_x, player_C_y)
        player_C_x, player_C_y = C[:, 0], C[:, 1]
        D = transformer.transform(player_D_x, player_D_y)
        player_D_x, player_D_y = D[:, 0], D[:, 1]
        H = transformer.transform(hit_x, hit_y)
        hit_x, hit_y = H[:, 0], H[:, 1]
        R = transformer.transform(return_x, return_y)
        return_x, return_y = R[:, 0], R[:, 1]

        shot_["player_A_x"] = player_A_x
        shot_["player_A_y"] = player_A_y
        shot_["player_B_x"] = player_B_x
        shot_["player_B_y"] = player_B_y
        shot_["player_C_x"] = player_C_x
        shot_["player_C_y"] = player_C_y
        shot_["player_D_x"] = player_D_x
        shot_["player_D_y"] = player_D_y

        shot_["hit_x"] = hit_x
        shot_["hit_y"] = hit_y
        shot_["return_x"] = return_x
        shot_["return_y"] = return_y
        rtns.append(shot_)
    data = pd.concat(rtns)
    data.to_csv("./convert_shot.csv", index=False)
