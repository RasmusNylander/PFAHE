import numpy as np
from numpy import ndarray


def rgb_to_xyz(images: ndarray) -> ndarray:
	"""
	Converts RGB images to XYZ color space.

	:param images: The input RGB images (B, H, W, C).
	:return: The input images in XYZ color space (B, H, W, C).
	"""

	# images = images.astype(np.float32) / 255.0
	images = images.copy()
	mask = images > 0.04045

	images[mask] = np.power((images[mask] + 0.055) / 1.055, 2.4)
	images[~mask] /= 12.92

	images *= 100.0

	magic_constants = np.array([
		[0.412453, 0.357580, 0.180423],
		[0.212671, 0.715160, 0.072169],
		[0.019334, 0.119193, 0.950227]
	])

	images_xyz = images @ magic_constants.T

	return images_xyz


def xyz_to_rgb(images: ndarray) -> ndarray:
	"""
	Converts XYZ images to RGB color space.

	:param images: The input XYZ images (B, H, W, C).
	:return: The input images in RGB color space (B, H, W, C).
	"""

	images = images / 100.0

	magic_constants = np.array([
		[3.24048134, -1.53715152, -0.49853633],
		[-0.96925495, 1.87599, 0.04155593],
		[0.05564664, -0.20404134, 1.05731107]
	])

	images_rgb = images @ magic_constants.T

	mask = images_rgb > 0.0031308
	images_rgb[mask] = 1.055 * np.power(images_rgb[mask], 1.0 / 2.4) - 0.055
	images_rgb[~mask] *= 12.92

	return np.clip(images_rgb, 0.0, 1.0)


def xyz_to_lab(images: ndarray) -> ndarray:
	"""
	Converts XYZ images to LAB color space.
	:param images:  The input XYZ images (B, H, W, C).
	:return:  The input images in LAB color space (B, H, W, C).
	"""

	magic_constants = np.array([0.950456, 1.0, 1.088754])
	images /= magic_constants

	mask = images > 0.008856
	images[mask] = np.power(images[mask], 1.0 / 3.0)
	images[~mask] = 7.787 * images[~mask] + 16.0 / 116.0

	images = np.stack(
		[
			116.0 * images[..., 1] - 16.0,
			500.0 * (images[..., 0] - images[..., 1]),
			200.0 * (images[..., 1] - images[..., 2])
		], axis=-1)

	return images


def lab_to_xyz(images: ndarray) -> ndarray:
	"""
	Converts LAB images to XYZ color space.
	:param images:  The input LAB images (B, H, W, C).
	:return:  The input images in XYZ color space (B, H, W, C).
	"""

	y = (images[..., 0] + 16.0) / 116.0
	xyz = np.stack([
		images[..., 1] / 500.0 + y,
		y,
		y - images[..., 2] / 200.0
	], axis=-1)

	# mask = xyz > 0.206893
	mask = xyz > 0.008856
	xyz[mask] = np.power(xyz[mask], 3.0)
	xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787

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
