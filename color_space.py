import numpy as np
from numpy import ndarray


def rgb_to_xyz(images: ndarray) -> ndarray:
	"""
	Converts RGB images to XYZ color space.

	:param images: The input RGB images (B, H, W, C).
	:return: The input images in XYZ color space (B, H, W, C).
	"""

	magic_constants = np.array([
		[0.4124564566, 0.3575760779, 0.1804374833],
		[0.2126725044, 0.7151521552, 0.0721743070],
		[0.0193338847, 0.1191920250, 0.9503040953]
	])

	images_xyz = images @ magic_constants.T
	return images_xyz


def xyz_to_rgb(images: ndarray) -> ndarray:
	"""
	Converts XYZ images to RGB color space.

	:param images: The input XYZ images (B, H, W, C).
	:return: The input images in RGB color space (B, H, W, C).
	"""

	magic_constants = np.array([
		[3.2404521753, -1.5371373864, -0.4985322193],
		[-0.9692637934, 1.8760095563, 0.0415570448],
		[0.0556432260, -0.2040257870, 1.0572250515]
	])

	images_rgb = images @ magic_constants.T
	return images_rgb


def xyz_to_lab(images: ndarray) -> ndarray:
	"""
	Converts XYZ images to LAB color space.
	:param images: The input XYZ images (B, H, W, C).
	:return: The input images in LAB color space (B, H, W, C).
	"""

	magic_constants = np.array([0.950456, 1.0, 1.088754])
	images /= magic_constants

	mask = images > 0.008856
	images[mask] = np.power(images[mask], 1.0 / 3.0)

	# TODO: Benchmark what is faster. Also with a real world example and not just microbenchmark, please.
	# lightness = images[..., 1].copy()
	# lightness[mask[..., 1]] = 116.0 * lightness[mask[..., 1]] - 16.0
	# lightness[~mask[..., 1]] *= 903.3
	mask_light = mask[..., 1]
	lightness = np.empty_like(images[..., 1])
	lightness[mask] = 116.0 * images[..., 1][mask_light] - 16.0
	lightness[~mask] = 903.3 * images[..., 1][~mask_light]

	images[~mask] = 7.787 * images[~mask] + 16.0 / 116.0

	images = np.stack(
		[
			lightness,
			500.0 * (images[..., 0] - images[..., 1]),
			200.0 * (images[..., 1] - images[..., 2])
		], axis=-1)

	return images


def lab_to_xyz(images: ndarray) -> ndarray:
	"""
	Converts LAB images to XYZ color space.
	:param images: The input LAB images (B, H, W, C).
	:return: The input images in XYZ color space (B, H, W, C).
	"""

	y = images[..., 0] / 903.3
	mask = y > 0.008856
	y[mask] = ((images[..., 0][mask] + 16.0) / 116.0) ** 3.0

	f_of_y = y.copy()
	f_of_y[mask] = np.power(y[mask], 1.0 / 3.0)
	f_of_y[~mask] = 7.787 * y[~mask] + 16.0 / 116.0

	f_of_x = images[..., 1] / 500.0, + f_of_y
	x = (f_of_x - 16.0 / 116.0) / 7.787
	# mask = x > 0.206893
	mask = x > 0.008856
	x[mask] = np.power(f_of_x[mask], 3.0)

	f_of_z = -(images[..., 2] / 200.0 - f_of_y)
	z = (f_of_z - 16.0 / 116.0) / 7.787
	# mask = z > 0.206893
	mask = z > 0.008856
	z[mask] = np.power(f_of_z[mask], 3.0)

	xyz = np.stack([x, y, z], axis=-1)
	magic_constants = np.array([0.950456, 1.0, 1.088754])
	xyz *= magic_constants

	return xyz


def rgb_to_lab(images: ndarray) -> ndarray:
	"""
	Converts RGB images to LAB color space.

	:param images: The input RGB images (B, H, W, C).
	:return: The input images in LAB color space (B, H, W, C).
	"""

	return xyz_to_lab(rgb_to_xyz(images))


def lab_to_rgb(images: ndarray) -> ndarray:
	"""
	Converts LAB images to RGB color space.

	:param images: The input LAB images (B, H, W, C).
	:return: The input images in RGB color space (B, H, W, C).
	"""

	return xyz_to_rgb(lab_to_xyz(images))

# TODO: Test these functions
