import numpy as np
import cv2
import os
import random
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class RandomCirclesNoise(ImageOnlyTransform):
    """
    Adds evenly spaced transparent, dark circles on the image with slight random offsets.

    Parameters:
        num_circles (int): Number of circles to add (total in the grid).
        min_radius (int): Minimum radius of the circles.
        max_radius (int): Maximum radius of the circles.
        alpha (float): Transparency of the circles (0 = fully transparent, 1 = fully opaque).
        offset (int): Maximum range of random offset from the center of the grid cell.
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        num_circles=50,
        min_radius=5,
        max_radius=20,
        alpha=0.3,
        color=(0, 0, 0),
        offset=10,
        always_apply=False,
        p=1,
    ):
        super(RandomCirclesNoise, self).__init__(always_apply, p)
        self.num_circles = num_circles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.alpha = alpha
        self.color = color  # circle color
        self.offset = offset  # maximum random offset

    def apply(self, image, **params):
        """
        Applies the effect of evenly spaced circles on the image.

        Parameters:
            image (numpy array): Input image.

        Returns:
            numpy array: Image with evenly spaced transparent circles overlaid.
        """

        # Create a copy of the image to overlay the circles
        overlay = image.copy()

        # Calculate the number of columns and rows based on the number of circles and image size
        grid_size = int(np.sqrt(self.num_circles))
        step_x = image.shape[1] // grid_size
        step_y = image.shape[0] // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate the center of the grid cell
                center_x = j * step_x + step_x // 2
                center_y = i * step_y + step_y // 2

                # Add a random offset in the range of -offset to +offset
                x = center_x + np.random.randint(-self.offset, self.offset)
                y = center_y + np.random.randint(-self.offset, self.offset)

                # Random radius in the defined range
                if self.min_radius == self.max_radius:
                    radius = self.min_radius
                else:
                    radius = np.random.randint(self.min_radius, self.max_radius)

                # Draw a dark circle on the mask
                cv2.circle(overlay, (x, y), radius, self.color, -1)

        # Apply the final mask of circles on the image with transparency
        noisy_image = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return noisy_image


class RandomHorizontalStripes(ImageOnlyTransform):
    """
    Adds randomly placed horizontal stripes on the image, simulating the effect of horizontal smudges.

    Parameters:
        num_stripes (int): Number of horizontal stripes.
        min_thickness (int): Minimal thickness of the stripes.
        max_thickness (int): Maximal thickness of the stripes.
        alpha (float): Transparency of the stripes (0 = fully transparent, 1 = fully opaque).
        color (tuple): Color of the stripes in RGB format (default is black).
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        num_stripes=20,
        min_thickness=2,
        max_thickness=10,
        alpha=0.3,
        color=(0, 0, 0),
        always_apply=False,
        p=1,
    ):
        super(RandomHorizontalStripes, self).__init__(always_apply, p)
        self.num_stripes = num_stripes
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.alpha = alpha
        self.color = color  # Stripe color

    def apply(self, image, **params):
        """
        Applies the effect of horizontal stripes on the image.

        Parameters:
            image (numpy array): Obraz wejściowy.

        Return:
            numpy array: Image with randomly placed horizontal stripes.
        """
        # Create copy of the image to overlay the stripes
        overlay = image.copy()
        height, width = image.shape[:2]

        for _ in range(self.num_stripes):
            # Random position in the vertical direction and thickness of the stripe
            y = np.random.randint(0, height)
            thickness = np.random.randint(self.min_thickness, self.max_thickness)

            # Draw a horizontal stripe on the overlay
            cv2.rectangle(overlay, (0, y), (width, y + thickness), self.color, -1)

        # Apply the final mask of stripes on the image with transparency
        striped_image = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return striped_image


class RandomHorizontalLines(ImageOnlyTransform):
    """
    Adds randomly placed horizontal lines on the image, simulating the effect of horizontal smudges.

    Parameters:
        num_lines (int): Number of horizontal lines.
        min_length (int): Minimal length of the lines.
        max_length (int): Maximal length of the lines.
        min_thickness (int): Minimal thickness of the lines.
        max_thickness (int): Maximal thickness of the lines.
        alpha (float): Transparency of the lines (0 = fully transparent, 1 = fully opaque).
        color (tuple): Color of the lines in RGB format (default is gray).
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        num_lines=10,
        min_length=20,
        max_length=100,
        min_thickness=2,
        max_thickness=2,
        alpha=0.2,
        color=(200, 200, 200),
        always_apply=False,
        p=1,
    ):
        super(RandomHorizontalLines, self).__init__(always_apply, p)
        self.num_lines = num_lines
        self.min_length = min_length
        self.max_length = max_length
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.alpha = alpha
        self.color = color  # Line color

    def apply(self, image, **params):
        """
        Applies the effect of horizontal lines on the image.

        Parameters:
            image (numpy array): Input image.

        Return:
            numpy array: Image with randomly placed horizontal lines.
        """
        # Create copy of the image to overlay the lines
        overlay = image.copy()
        height, width = image.shape[:2]

        for _ in range(self.num_lines):
            # Random position in the vertical direction, length and thickness of the line
            y = np.random.randint(0, height)
            length = np.random.randint(self.min_length, self.max_length)
            thickness = np.random.randint(self.min_thickness, self.max_thickness)

            # Random start and end of the line in the horizontal direction
            start_x = np.random.randint(0, width - length)
            end_x = start_x + length

            # Draw a horizontal line on the overlay
            cv2.line(overlay, (start_x, y), (end_x, y), self.color, thickness)

        # Apply the final mask of lines on the image with transparency
        lined_image = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return lined_image


class RandomHorizontalRain(ImageOnlyTransform):
    """
    Adds randomly placed horizontal lines on the image, simulating the effect of horizontal rain.

    Parameters:
        line_length (int): Length of the horizontal lines.
        line_thickness (int): Thickness of the horizontal lines.
        line_color (tuple[int, int, int]): Color of the horizontal lines in RGB format.
        blur_value (int): Blur value for achieving soft lines.
        brightness_coefficient (float): Brightness coefficient for adjusting the brightness of the lines.
        num_lines (int): Number of horizontal lines.
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        line_length: int = 50,
        line_thickness: int = 2,
        line_color: tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 5,
        brightness_coefficient: float = 0.8,
        num_lines: int = 50,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.line_length = line_length
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.num_lines = num_lines

    def apply(
        self, img: np.ndarray, horizontal_lines: list[tuple[int, int]], **params
    ) -> np.ndarray:
        overlay = img.copy()

        for x, y in horizontal_lines:
            end_x = min(x + self.line_length, img.shape[1])
            cv2.line(overlay, (x, y), (end_x, y), self.line_color, self.line_thickness)

        # Blur and adjust brightness
        if self.blur_value > 0:
            overlay = cv2.GaussianBlur(overlay, (self.blur_value, self.blur_value), 0)

        blended_img = cv2.addWeighted(
            overlay,
            self.brightness_coefficient,
            img,
            1 - self.brightness_coefficient,
            0,
        )
        return blended_img

    def get_params_dependent_on_data(self, params, data):
        height, width = params["shape"][:2]
        horizontal_lines = [
            (
                np.random.randint(0, width - self.line_length),
                np.random.randint(0, height),
            )
            for _ in range(self.num_lines)
        ]
        return {"horizontal_lines": horizontal_lines}

    def get_transform_init_args_names(self):
        return (
            "line_length",
            "line_thickness",
            "line_color",
            "blur_value",
            "brightness_coefficient",
            "num_lines",
        )


class RandomDis(ImageOnlyTransform):
    """
    Adds predefined textures from images in a directory onto the base image.

    Parameters:
    - images_path (str): Path to the directory containing spot images.
    - frequency (tuple): Frequency of spots (min, max). Defaults to (1, 1000).
    - id_ (int): ID of the spot image to use. Random if None.
    - size (tuple): Size of the spot (min, max). Randomly generated if None.
    - opacity (tuple): Opacity of the spot (min, max) or single float value. Random if None.
    """

    def __init__(
        self,
        images_path="src/imgs",
        frequency: tuple = None,
        id_: int = None,
        size: tuple = None,
        opacity: tuple = None,
    ):
        super().__init__(always_apply=True)
        self.images_path = images_path
        self.__images = self.__load_images()
        self.frequency = self.__parse_frequency(frequency)
        self.id_ = id_
        self.size = size
        self.opacity = opacity

    def __parse_frequency(self, frequency):
        if frequency is not None and not isinstance(frequency, tuple):
            raise ValueError("Frequency must be a tuple")
        return (
            random.randint(frequency[0], frequency[1])
            if frequency
            else random.randint(1, 1000)
        )

    def __load_images(self):
        images = []
        for img in os.listdir(self.images_path):
            image = cv2.imread(
                os.path.join(self.images_path, img), cv2.IMREAD_UNCHANGED
            )
            images.append(image)
        return images

    def show_images(self):
        for idx, img in enumerate(self.__images):
            cv2.imshow(f"image id {idx}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def __generate_spot_size(self, frequency, width, height):
        inverse_frequency = 1000 - frequency
        normalized_frequency = inverse_frequency / 1000
        base_size = int(normalized_frequency * min(width, height) // 3)
        random_factor = random.uniform(0.5, 1.1)
        return max(5, int(base_size * random_factor))

    def __rotate_spot(self, spot, angle):
        (h, w) = spot.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            spot,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    def __apply_spot_on_image(self, spot, opacity, x, y, image):
        spot_height, spot_width = spot.shape[:2]
        if spot.shape[2] == 4:
            alpha_channel = spot[:, :, 3] * opacity
            rgb_channels = spot[:, :, :3]
        else:
            raise ValueError("Spot image must have an alpha channel.")

        alpha_mask = alpha_channel / 255.0
        roi = image[y : y + spot_height, x : x + spot_width]

        for c in range(3):
            roi[:, :, c] = (
                rgb_channels[:, :, c] * alpha_mask + roi[:, :, c] * (1 - alpha_mask)
            ).astype(np.uint8)

        image[y : y + spot_height, x : x + spot_width] = roi
        return image

    def apply(self, image: np.ndarray, **params):
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        height, width = image.shape[:2]
        if (
            self.size is not None
            and isinstance(self.size, tuple)
            and max(self.size) > min(height, width)
        ):
            raise ValueError("Size is too big")

        image_stained = image.copy()
        id_was_none, size_was_none, opacity_was_none = (
            self.id_ is None,
            self.size is None,
            self.opacity is None,
        )

        for _ in range(self.frequency):
            self.size = self.size or self.__generate_spot_size(
                self.frequency, width, height
            )
            if isinstance(self.size, tuple):
                self.size = random.randint(self.size[0], self.size[1])

            self.id_ = (
                self.id_
                if self.id_ is not None
                else random.randint(0, len(self.__images) - 1)
            )
            x, y = random.randint(0, width - self.size), random.randint(
                0, height - self.size
            )

            self.opacity = self.opacity or random.uniform(0.1, 0.9)
            if isinstance(self.opacity, tuple):
                self.opacity = random.uniform(self.opacity[0], self.opacity[1])

            spot = cv2.resize(self.__images[self.id_].copy(), (self.size, self.size))
            spot = self.__rotate_spot(spot, random.uniform(0, 360))
            image_stained = self.__apply_spot_on_image(
                spot, self.opacity, x, y, image_stained
            )

            if id_was_none:
                self.id_ = None
            if size_was_none:
                self.size = None
            if opacity_was_none:
                self.opacity = None

        return image_stained


class Checks(ImageOnlyTransform):
    """
    Adds a checkered pattern to the image with specified size and opacity.

    Parameters:
    - size (int or str): Size of the checks. Accepts an integer or 'width'/'height' to scale based on image dimensions.
    - opacity (float): Opacity level of the checks. Random if not specified.
    - x (int): Scaling factor when size is 'width' or 'height'.
    """

    def __init__(self, size=None, opacity=None, x=10, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.size = size
        self.opacity = opacity
        self.x = x

    def __generate_check_size(self, width: int, height: int) -> int:
        """
        Generates the size of each check in the pattern based on initialized parameters.
        """
        if isinstance(self.size, int):
            return self.size
        elif self.size == "width":
            return width // self.x
        elif self.size == "height":
            return height // self.x
        else:
            return random.randint(10, min(width, height) // 5)

    def __apply_opacity(self, image: np.ndarray, opacity: float) -> np.ndarray:
        """
        Creates an overlay with the specified opacity level.
        """
        black_square = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        black_square[:, :, 3] = int(opacity * 255)
        return black_square

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies a checkered pattern overlay to the input image.

        Parameters:
        - image (np.ndarray): Input image to which the pattern will be applied.

        Returns:
        - np.ndarray: Image with checkered overlay.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        height, width = image.shape[:2]
        check_size = self.__generate_check_size(width, height)

        opacity = self.opacity if self.opacity is not None else random.uniform(0.1, 0.9)

        checks = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(0, height, check_size):
            for x in range(0, width, check_size):
                if (x // check_size + y // check_size) % 2 == 0:
                    checks[y : y + check_size, x : x + check_size, 3] = 0
                else:
                    checks[y : y + check_size, x : x + check_size, :3] = 0
                    checks[y : y + check_size, x : x + check_size, 3] = int(
                        opacity * 255
                    )

        image_with_checks = image.copy()
        alpha_mask = checks[:, :, 3] / 255.0
        for c in range(3):
            image_with_checks[:, :, c] = (
                checks[:, :, c] * alpha_mask
                + image_with_checks[:, :, c] * (1 - alpha_mask)
            ).astype(np.uint8)

        return image_with_checks


class Movie(ImageOnlyTransform):
    """
    Applies a film grain effect to an image.

    Parameters:
    - intensity (float): Intensity of the grain effect (0-1).
    """

    def __init__(self, intensity=0.5, always_apply=True):
        super().__init__(always_apply=always_apply)
        if not (0 <= intensity <= 1):
            raise ValueError("Intensity must be between 0 and 1.")
        self.intensity = intensity

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies the film grain effect to the input image.

        Parameters:
        - image (np.ndarray): Input image to which the effect will be applied.

        Returns:
        - np.ndarray: Image with film grain effect.
        """
        height, width = image.shape[:2]

        noise = np.random.normal(0, 255 * self.intensity, (height, width)).astype(
            np.uint8
        )

        if len(image.shape) == 3:
            noise = cv2.merge([noise, noise, noise])

        image_with_grain = cv2.addWeighted(
            image, 1 - self.intensity, noise, self.intensity, 0
        )

        return image_with_grain


class ColorReduce(ImageOnlyTransform):
    """
    Reduces the color levels in an image to create a quantized effect.

    Parameters:
    - levels (int): Number of color levels for quantization. Higher values retain more color detail.
    """

    def __init__(self, levels=4, always_apply=True):
        super().__init__(always_apply=always_apply)
        if levels < 1:
            raise ValueError("Levels must be at least 1.")
        self.levels = levels

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies color reduction by quantizing color levels.

        Parameters:
        - image (np.ndarray): Input image to be processed.

        Returns:
        - np.ndarray: Color-reduced image.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array.")

        factor = 256 // self.levels

        reduced_image = (image // factor) * factor + factor // 2

        return reduced_image


class Reflections(ImageOnlyTransform):
    """
    Adds random reflections to an image.

    Parameters:
    - intensity (float): Intensity of the reflections (0-1).
    - max_reflections (int): Maximum number of reflections to add.
    """

    def __init__(self, intensity=0.2, max_reflections=5, always_apply=True):
        super().__init__(always_apply=always_apply)
        if not (0 <= intensity <= 1):
            raise ValueError("Intensity must be between 0 and 1.")
        self.intensity = intensity
        self.max_reflections = max_reflections

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Adds random reflections to the input image.

        Parameters:
        - image (np.ndarray): Input image to which reflections will be applied.

        Returns:
        - np.ndarray: Image with added reflections.
        """
        height, width = image.shape[:2]
        image_with_reflections = image.copy()

        for _ in range(random.randint(1, self.max_reflections)):
            radius = random.randint(10, min(width, height) // 5)
            x, y = random.randint(0, width - radius), random.randint(0, height - radius)
            overlay = np.zeros_like(image_with_reflections, dtype=np.uint8)

            cv2.circle(overlay, (x, y), radius, (255, 255, 255), -1)

            image_with_reflections = cv2.addWeighted(
                image_with_reflections, 1 - self.intensity, overlay, self.intensity, 0
            )

        return image_with_reflections


class FocusDrift(ImageOnlyTransform):
    """
    Applies a selective blur effect to detected face areas in an image.

    Parameters:
    - intensity (float): Intensity of the blur (0-1).
    - segmentation_model: A segmentation model instance (e.g., Segformer). Defaults to Segformer if None.
    - face_class (int): The class label for faces in segmentation output.
    """

    def __init__(
        self, intensity=0.9, segmentation_model=None, face_class=1, always_apply=True
    ):
        super().__init__(always_apply=always_apply)
        self.intensity = intensity
        self.face_class = face_class
        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

        # Load the default segmentation model if none provided
        if segmentation_model is None:
            self.segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
        else:
            self.segmentation_model = segmentation_model

    def apply(self, image, **params):
        """
        Applies the selective blur effect to detected faces in the input image.

        Parameters:
        - image (np.ndarray): Input image on which to apply the blur effect.

        Returns:
        - np.ndarray: Image with blurred face areas.
        """
        mask = self.segment_face(image)

        if mask is None:
            print("Face not detected.")
            return image

        height, width = image.shape[:2]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 3:
            mask = cv2.merge([mask, mask, mask])

        blurred_face = cv2.GaussianBlur(image, (15, 15), self.intensity * 10)

        image_with_blur = np.where(mask == 0, blurred_face, image)

        return image_with_blur

    def segment_face(self, image):
        """
        Generates a face mask using the segmentation model.

        Parameters:
        - image (np.ndarray): Input image for face segmentation.

        Returns:
        - np.ndarray: Binary mask for the face.
        """
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.segmentation_model(**inputs)

        logits = outputs.logits  # [batch_size, num_labels, height, width]
        mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        face_mask = (mask == self.face_class).astype(np.uint8) * 255

        if np.sum(face_mask) == 0:
            print("No face detected for class:", self.face_class)
            return None

        return face_mask


class GeoWarping(ImageOnlyTransform):
    """
    Applies small geometric distortions to the image.

    Parameters:
    - intensity (int): Maximum displacement for the distortions.
    """

    def __init__(self, intensity=5, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.intensity = intensity

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies small geometric distortions to the input image.

        Parameters:
        - image (np.ndarray): Input image to be distorted.

        Returns:
        - np.ndarray: Image with geometric distortion.
        """
        height, width = image.shape[:2]
        image_warped = image.copy()

        for _ in range(random.randint(1, 5)):
            # Select random control points and displacements
            x, y = random.randint(0, width), random.randint(0, height)
            dx, dy = random.randint(-self.intensity, self.intensity), random.randint(
                -self.intensity, self.intensity
            )
            pts1 = np.float32([[x, y], [x + 1, y], [x, y + 1]])
            pts2 = np.float32([[x + dx, y + dy], [x + 1 + dx, y + dy], [x, y + 1 + dy]])

            # Create the affine transformation matrix
            matrix = cv2.getAffineTransform(pts1, pts2)
            image_warped = cv2.warpAffine(
                image_warped,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        return image_warped


import random
import numpy as np
import cv2
from albumentations import ImageOnlyTransform


class GeoWarping(ImageOnlyTransform):
    """
    Applies small geometric distortions to the image.

    Parameters:
    - intensity (int): Maximum displacement for the distortions.
    """

    def __init__(self, intensity=3, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.intensity = intensity  # Zmniejszenie domyślnego poziomu intensywności

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies small geometric distortions to the input image.

        Parameters:
        - image (np.ndarray): Input image to be distorted.

        Returns:
        - np.ndarray: Image with geometric distortion.
        """
        height, width = image.shape[:2]
        image_warped = image.copy()

        # Zmniejszenie liczby iteracji i zakresu zniekształceń
        for _ in range(random.randint(1, 3)):  # Zmniejszenie liczby iteracji
            x, y = random.randint(width/2, width - 1), random.randint(height/2, height - 1)
            dx = random.randint(
                -self.intensity // 2, self.intensity // 2
            )  # Zmniejszenie zakresu
            dy = random.randint(-self.intensity // 2, self.intensity // 2)

            pts1 = np.float32([[x, y], [x + 1, y], [x, y + 1]])
            pts2 = np.float32([[x + dx, y + dy], [x + 1 + dx, y + dy], [x, y + 1 + dy]])

            matrix = cv2.getAffineTransform(pts1, pts2)
            image_warped = cv2.warpAffine(
                image_warped,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        return image_warped


class Compression(ImageOnlyTransform):
    """
    Simulates social media compression artifacts by reducing JPEG quality.

    Parameters:
    - quality (int): JPEG quality level (1-100). Lower values produce more artifacts.
    """

    def __init__(self, quality=30, always_apply=True):
        super().__init__(always_apply=always_apply)
        if not (1 <= quality <= 100):
            raise ValueError("Quality must be between 1 and 100.")
        self.quality = quality

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Compresses the image to simulate social media compression artifacts.

        Parameters:
        - image (np.ndarray): Input image to be compressed.

        Returns:
        - np.ndarray: Compressed and decompressed image.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, compressed_img = cv2.imencode(".jpg", image, encode_param)
        decompressed_img = cv2.imdecode(compressed_img, cv2.IMREAD_COLOR)

        return decompressed_img


class Sharpen(ImageOnlyTransform):
    """
    Applies sharpening to an image to enhance edges.

    Parameters:
    - intensity (float): Intensity of sharpening. Higher values increase sharpening strength.
    """

    def __init__(self, intensity=1.0, always_apply=True):
        super().__init__(always_apply=always_apply)
        self.intensity = intensity

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies sharpening to the input image.

        Parameters:
        - image (np.ndarray): Input image to be sharpened.

        Returns:
        - np.ndarray: Sharpened image.
        """
        kernel = np.array([[0, -1, 0], [-1, 5 + self.intensity, -1], [0, -1, 0]])

        sharpened_image = cv2.filter2D(image, -1, kernel)

        return sharpened_image


class BaseRain(ImageOnlyTransform):
    """
    Base class for different types of rain effects.
    """

    def __init__(
        self,
        slant_range=(-10, 10),
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=7,
        brightness_coefficient=0.7,
        always_apply=False,
        p=1,
    ):
        super(BaseRain, self).__init__(always_apply, p)
        self.slant_range = slant_range
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient

    def apply_rain(self, image, drop_length, num_drops):
        rain_transform = A.RandomRain(
            slant_range=self.slant_range,
            drop_length=drop_length,
            drop_width=self.drop_width,
            drop_color=self.drop_color,
            blur_value=self.blur_value,
            brightness_coefficient=self.brightness_coefficient,
            rain_type="default",
            p=1.0,
        )

        # Override the number of drops by directly setting the number of drops as calculated
        augmented = rain_transform(image=image)
        return augmented["image"]


class DrizzleRain(BaseRain):
    """
    Light drizzle effect with small, sparse rain drops.
    """

    def apply(self, image, **params):
        height, width = image.shape[:2]
        area = height * width
        drop_length = 10
        num_drops = area // 770
        return self.apply_rain(image, drop_length, num_drops)


class HeavyRain(BaseRain):
    """
    Heavy rain effect with medium-sized, more frequent rain drops.
    """

    def apply(self, image, **params):
        height, width = image.shape[:2]
        drop_length = 30
        num_drops = (width * height) // 600
        return self.apply_rain(image, drop_length, num_drops)


class TorrentialRain(BaseRain):
    """
    Torrential rain effect with large, dense rain drops.
    """

    def apply(self, image, **params):
        height, width = image.shape[:2]
        area = height * width
        drop_length = 60
        num_drops = area // 500
        return self.apply_rain(image, drop_length, num_drops)


class RandomCheckerboard(ImageOnlyTransform):
    """
    Adds a semi-transparent checkerboard pattern over the image.

    Parameters:
        square_size (int): Size of each square in the checkerboard pattern.
        alpha (float): Transparency of the checkerboard pattern (0 = fully transparent, 1 = fully opaque).
        color (tuple): Color of the checkerboard squares in RGB format (default is black).
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        square_size=10,
        alpha=0.5,
        color=(255, 255, 255),
        always_apply=False,
        p=1.0,
    ):
        super(RandomCheckerboard, self).__init__(always_apply, p)
        self.square_size = square_size
        self.alpha = alpha
        self.color = color

    def apply(self, image, **params):
        """
        Applies the checkerboard pattern on the image.

        Parameters:
            image (numpy array): Input image.

        Returns:
            numpy array: Image with checkerboard pattern overlaid.
        """
        # Create an empty checkerboard mask with the same shape as the input image
        checkerboard = np.zeros_like(image, dtype=np.uint8)

        # Fill the checkerboard pattern
        for i in range(0, checkerboard.shape[0], self.square_size):
            for j in range(0, checkerboard.shape[1], self.square_size):
                if (i // self.square_size) % 2 == (j // self.square_size) % 2:
                    checkerboard[i : i + self.square_size, j : j + self.square_size] = (
                        self.color
                    )

        # Blend the checkerboard mask with the original image
        # Blend using alpha transparency, so that the checkerboard pattern overlays the image
        checkerboard_image = cv2.addWeighted(
            image, 1 - self.alpha, checkerboard, self.alpha, 0
        )
        return checkerboard_image


class RandomCheckerboardInCheckerboard(ImageOnlyTransform):
    """
    Adds a semi-transparent checkerboard pattern over the image, where each square in the checkerboard contains
    a smaller checkerboard pattern.

    Parameters:
        square_size (int): Size of each main square in the checkerboard pattern.
        inner_square_size (int): Size of each inner square in the sub-checkerboard pattern.
        alpha (float): Transparency of the checkerboard pattern (0 = fully transparent, 1 = fully opaque).
        color (tuple): Color of the checkerboard squares in RGB format.
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        square_size=20,
        inner_square_size=5,
        alpha=0.5,
        color=(255, 255, 255),
        always_apply=False,
        p=1,
    ):
        super(RandomCheckerboardInCheckerboard, self).__init__(always_apply, p)
        self.square_size = square_size
        self.inner_square_size = inner_square_size
        self.alpha = alpha
        self.color = color

    def apply(self, image, **params):
        """
        Applies the checkerboard in checkerboard pattern on the image.

        Parameters:
            image (numpy array): Input image.

        Returns:
            numpy array: Image with checkerboard pattern overlaid.
        """
        # Create an empty checkerboard mask with the same shape as the input image
        checkerboard = np.zeros_like(image, dtype=np.uint8)

        # Fill the main checkerboard pattern
        for i in range(0, checkerboard.shape[0], self.square_size):
            for j in range(0, checkerboard.shape[1], self.square_size):
                if (i // self.square_size) % 2 == (j // self.square_size) % 2:
                    # Fill the larger square with a smaller checkerboard pattern
                    for m in range(
                        i,
                        min(i + self.square_size, checkerboard.shape[0]),
                        self.inner_square_size,
                    ):
                        for n in range(
                            j,
                            min(j + self.square_size, checkerboard.shape[1]),
                            self.inner_square_size,
                        ):
                            if ((m - i) // self.inner_square_size) % 2 == (
                                (n - j) // self.inner_square_size
                            ) % 2:
                                checkerboard[
                                    m : m + self.inner_square_size,
                                    n : n + self.inner_square_size,
                                ] = self.color

        # Blend the checkerboard mask with the original image
        checkerboard_image = cv2.addWeighted(
            image, 1 - self.alpha, checkerboard, self.alpha, 0
        )
        return checkerboard_image


class RandomDottedCheckerboard(ImageOnlyTransform):
    """
    Adds a semi-transparent dotted checkerboard pattern over the image, where each larger square
    contains a grid of smaller dots.

    Parameters:
        square_size (int): Size of each main square in the checkerboard pattern.
        dot_spacing (int): Spacing between dots in the sub-pattern inside each square.
        dot_radius (int): Radius of each dot.
        alpha (float): Transparency of the checkerboard pattern (0 = fully transparent, 1 = fully opaque).
        color (tuple): Color of the dots in RGB format.
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        square_size=20,
        dot_spacing=5,
        dot_radius=1,
        alpha=0.5,
        color=(255, 255, 255),
        always_apply=False,
        p=1.0,
    ):
        super(RandomDottedCheckerboard, self).__init__(always_apply, p)
        self.square_size = square_size
        self.dot_spacing = dot_spacing
        self.dot_radius = dot_radius
        self.alpha = alpha
        self.color = color

    def apply(self, image, **params):
        """
        Applies the dotted checkerboard pattern on the image.

        Parameters:
            image (numpy array): Input image.

        Returns:
            numpy array: Image with dotted checkerboard pattern overlaid.
        """
        # Create an empty checkerboard mask with the same shape as the input image
        checkerboard = np.zeros_like(image, dtype=np.uint8)

        # Fill the main checkerboard pattern with dots
        for i in range(0, checkerboard.shape[0], self.square_size):
            for j in range(0, checkerboard.shape[1], self.square_size):
                if (i // self.square_size) % 2 == (j // self.square_size) % 2:
                    # Place dots in a grid inside each "checkerboard" square
                    for m in range(
                        i,
                        min(i + self.square_size, checkerboard.shape[0]),
                        self.dot_spacing,
                    ):
                        for n in range(
                            j,
                            min(j + self.square_size, checkerboard.shape[1]),
                            self.dot_spacing,
                        ):
                            cv2.circle(
                                checkerboard, (n, m), self.dot_radius, self.color, -1
                            )

        # Blend the checkerboard mask with the original image
        dotted_checkerboard_image = cv2.addWeighted(
            image, 1 - self.alpha, checkerboard, self.alpha, 0
        )
        return dotted_checkerboard_image


class RandomDotGrid(ImageOnlyTransform):
    """
    Adds a semi-transparent dot grid pattern over the image, where dots are evenly spaced across the entire image.

    Parameters:
        dot_spacing (int): Spacing between dots in pixels.
        dot_radius (int): Radius of each dot.
        alpha (float): Transparency of the dot pattern (0 = fully transparent, 1 = fully opaque).
        color (tuple): Color of the dots in RGB format.
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        dot_spacing=10,
        dot_radius=1,
        alpha=0.5,
        color=(0, 0, 0),
        always_apply=False,
        p=1,
    ):
        super(RandomDotGrid, self).__init__(always_apply, p)
        self.dot_spacing = dot_spacing
        self.dot_radius = dot_radius
        self.alpha = alpha
        self.color = color

    def apply(self, image, **params):
        """
        Applies the dot grid pattern on the image.

        Parameters:
            image (numpy array): Input image.

        Returns:
            numpy array: Image with dot grid pattern overlaid.
        """
        # Create an empty mask with the same shape as the input image
        dot_grid = np.zeros_like(image, dtype=np.uint8)

        # Fill the mask with dots evenly spaced across the entire image
        for i in range(0, dot_grid.shape[0], self.dot_spacing):
            for j in range(0, dot_grid.shape[1], self.dot_spacing):
                cv2.circle(dot_grid, (j, i), self.dot_radius, self.color, -1)

        # Blend the dot grid mask with the original image
        dotted_image = cv2.addWeighted(image, 1 - self.alpha, dot_grid, self.alpha, 0)
        return dotted_image


class RandomGridOverlay(ImageOnlyTransform):
    """
    Adds a semi-transparent grid overlay to the image, with evenly spaced grid lines.

    Parameters:
        grid_spacing (int): Spacing between grid lines in pixels.
        line_thickness (int): Thickness of each grid line.
        alpha (float): Transparency of the grid (0 = fully transparent, 1 = fully opaque).
        color (tuple): Color of the grid lines in RGB format.
        always_apply (bool): Apply the transformation always.
        p (float): Probability of applying the transformation.
    """

    def __init__(
        self,
        grid_spacing=20,
        line_thickness=1,
        alpha=0.5,
        color=(0, 0, 0),
        always_apply=False,
        p=1,
    ):
        super(RandomGridOverlay, self).__init__(always_apply, p)
        self.grid_spacing = grid_spacing
        self.line_thickness = line_thickness
        self.alpha = alpha
        self.color = color

    def apply(self, image, **params):
        """
        Applies the grid overlay pattern on the image.

        Parameters:
            image (numpy array): Input image.

        Returns:
            numpy array: Image with grid overlay.
        """
        # Create an empty grid overlay with the same shape as the input image
        grid_overlay = np.zeros_like(image, dtype=np.uint8)

        # Draw vertical grid lines
        for x in range(0, grid_overlay.shape[1], self.grid_spacing):
            cv2.line(
                grid_overlay,
                (x, 0),
                (x, grid_overlay.shape[0]),
                self.color,
                self.line_thickness,
            )

        # Draw horizontal grid lines
        for y in range(0, grid_overlay.shape[0], self.grid_spacing):
            cv2.line(
                grid_overlay,
                (0, y),
                (grid_overlay.shape[1], y),
                self.color,
                self.line_thickness,
            )

        # Blend the grid overlay with the original image
        grid_image = cv2.addWeighted(image, 1 - self.alpha, grid_overlay, self.alpha, 0)
        return grid_image


class DirtyWindow(ImageOnlyTransform):
    """
    Applies a "dirty window" effect to the image by overlaying smudges and spots.

    Parameters:
    - dirt_density (float): Density of dirt (0-1). Higher values increase the number of smudges and spots.
    - dirt_intensity (float): Intensity of the dirt effect (0-1). Higher values make the spots more opaque.
    - max_dirt_size (int): Maximum size of individual dirt spots or smudges.
    """

    def __init__(
        self,
        dirt_density=0.5,
        dirt_intensity=0.5,
        max_dirt_size=20,
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.dirt_density = dirt_density
        self.dirt_intensity = dirt_intensity
        self.max_dirt_size = max_dirt_size

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Adds dirt-like smudges and spots to the input image.

        Parameters:
        - image (np.ndarray): Input image to which the dirty effect will be applied.

        Returns:
        - np.ndarray: Image with "dirty window" effect applied.
        """
        height, width = image.shape[:2]
        dirty_image = image.copy()

        num_spots = int(self.dirt_density * 100)

        for _ in range(num_spots):
            dirt_size = random.randint(5, self.max_dirt_size)
            x, y = random.randint(0, width - dirt_size), random.randint(
                0, height - dirt_size
            )

            spot = np.zeros((dirt_size, dirt_size, 3), dtype=np.uint8)
            spot = cv2.circle(
                spot,
                (dirt_size // 2, dirt_size // 2),
                dirt_size // 2,
                (255, 255, 255),
                -1,
            )

            spot_alpha = random.uniform(self.dirt_intensity * 0.5, self.dirt_intensity)
            spot = cv2.GaussianBlur(
                spot, (dirt_size // 2 * 2 + 1, dirt_size // 2 * 2 + 1), 0
            )
            spot_alpha_mask = (spot[:, :, 0] / 255.0) * spot_alpha

            for c in range(3):
                dirty_image[y : y + dirt_size, x : x + dirt_size, c] = (
                    spot[:, :, c] * spot_alpha_mask
                    + dirty_image[y : y + dirt_size, x : x + dirt_size, c]
                    * (1 - spot_alpha_mask)
                ).astype(np.uint8)

        return dirty_image


class VintageFilmEffect(ImageOnlyTransform):
    """
    Applies a vintage film effect with noise, scratches, and fading.

    Parameters:
    - noise_intensity (float): Intensity of the grain/noise (0-1).
    - blur_intensity (int): Kernel size for blurring (higher values create stronger blur).
    - fade_intensity (float): Fading effect, reducing contrast (0-1).
    - scratch_density (float): Density of scratches (0-1).
    """

    def __init__(
        self,
        noise_intensity=0.5,
        blur_intensity=3,
        fade_intensity=0.2,
        scratch_density=0.1,
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.noise_intensity = noise_intensity
        self.blur_intensity = blur_intensity
        self.fade_intensity = fade_intensity
        self.scratch_density = scratch_density

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies the vintage film effect to the input image.

        Parameters:
        - image (np.ndarray): Input image to which the effect will be applied.

        Returns:
        - np.ndarray: Image with vintage film effect applied.
        """

        noise = np.random.normal(0, 255 * self.noise_intensity, image.shape).astype(
            np.uint8
        )
        noisy_image = cv2.addWeighted(
            image, 1 - self.noise_intensity, noise, self.noise_intensity, 0
        )

        blurred_image = cv2.GaussianBlur(
            noisy_image, (self.blur_intensity * 2 + 1, self.blur_intensity * 2 + 1), 0
        )

        faded_image = cv2.addWeighted(
            blurred_image,
            1 - self.fade_intensity,
            np.full_like(blurred_image, 128),
            self.fade_intensity,
            0,
        )

        height, width = faded_image.shape[:2]
        scratch_image = faded_image.copy()
        num_scratch_lines = int(self.scratch_density * 100)

        for _ in range(num_scratch_lines):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = x1 + random.randint(-20, 20), y1 + random.randint(-20, 20)
            color = (random.randint(200, 255),) * 3  # Light color for scratches
            cv2.line(scratch_image, (x1, y1), (x2, y2), color, 1)

        vintage_image = cv2.addWeighted(faded_image, 0.9, scratch_image, 0.1, 0)

        return vintage_image


class WindowTexture(ImageOnlyTransform):
    """
    Applies a dirty overlay effect by blending a given texture onto an image with transparency options.

    Parameters:
    - texture_path (str): Path to the dirt texture image.
    - texture_intensity (float): Blending intensity of the texture overlay (0-1). Higher values make the texture more prominent.
    - remove_light (bool): If True, light pixels in the texture will be made transparent.
    - remove_dark (bool): If True, dark pixels in the texture will be made transparent.
    - light_threshold (int): Threshold for light colors (0-255). Colors above this are treated as light.
    - dark_threshold (int): Threshold for dark colors (0-255). Colors below this are treated as dark.
    - opacity (float): Opacity level for the texture overlay (0-1). Controls overall transparency of the texture.
    - blur_amount (int): Amount of Gaussian blur applied to the texture before overlaying. Higher values increase blur.
    """

    def __init__(
        self,
        texture_path,
        texture_intensity=0.7,
        remove_light=False,
        remove_dark=False,
        light_threshold=240,
        dark_threshold=50,
        opacity=1.0,
        blur_amount=0,
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.texture_path = texture_path
        self.texture_intensity = texture_intensity
        self.remove_light = remove_light
        self.remove_dark = remove_dark
        self.light_threshold = light_threshold
        self.dark_threshold = dark_threshold
        self.opacity = opacity
        self.blur_amount = blur_amount

        self.texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
        if self.texture is None:
            raise ValueError(f"Could not load texture from path: {texture_path}")

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        height, width = image.shape[:2]

        texture_resized = cv2.resize(
            self.texture, (width, height), interpolation=cv2.INTER_LINEAR
        )

        if self.blur_amount > 0:
            texture_resized = cv2.GaussianBlur(
                texture_resized, (self.blur_amount * 2 + 1, self.blur_amount * 2 + 1), 0
            )

        texture_with_alpha = cv2.cvtColor(texture_resized, cv2.COLOR_BGR2BGRA)

        gray_texture = cv2.cvtColor(texture_resized, cv2.COLOR_BGR2GRAY)

        if self.remove_light:
            _, light_mask = cv2.threshold(
                gray_texture, self.light_threshold, 255, cv2.THRESH_BINARY_INV
            )
            texture_with_alpha[:, :, 3] = cv2.bitwise_and(
                texture_with_alpha[:, :, 3], light_mask
            )

        if self.remove_dark:
            _, dark_mask = cv2.threshold(
                gray_texture, self.dark_threshold, 255, cv2.THRESH_BINARY
            )
            texture_with_alpha[:, :, 3] = cv2.bitwise_and(
                texture_with_alpha[:, :, 3], dark_mask
            )

        texture_with_alpha[:, :, 3] = (
            texture_with_alpha[:, :, 3] * self.opacity
        ).astype(np.uint8)

        output_image = image.copy()
        alpha_mask = texture_with_alpha[:, :, 3] / 255.0  # Normalize alpha to range 0-1

        for c in range(3):  # Apply to R, G, B channels
            output_image[:, :, c] = (
                texture_with_alpha[:, :, c] * alpha_mask * self.texture_intensity
                + output_image[:, :, c] * (1 - alpha_mask * self.texture_intensity)
            ).astype(np.uint8)

        return output_image


class WatermarkEffect(ImageOnlyTransform):
    """
    Adds a text watermark to the image with different styles.

    Parameters:
    - text (str): The text to be used as the watermark.
    - opacity (float): Transparency level of the watermark (0-1). Higher values make the watermark more visible.
    - style (str): The watermarking style ('line_by_line', 'random', 'center').
    - line_spacing (float): For 'line_by_line' style, the spacing between lines as a percentage of image height.
    - density (int): For 'random' style, the number of watermark instances on the image.
    """

    def __init__(
        self,
        text,
        opacity=0.3,
        style="center",
        line_spacing=0.1,
        density=10,
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.text = text
        self.opacity = opacity
        self.style = style
        self.line_spacing = line_spacing  # Used only for line_by_line style
        self.density = density  # Used only for random style

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        height, width = image.shape[:2]

        watermark_image = image.copy()
        watermark_image = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2BGRA)

        if self.style == "line_by_line":
            # Style 1: Line by line from top to bottom with specified spacing
            font_scale = (
                0.05 * height / 20
            )  # Dynamically scale font based on image height
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_height = int(self.line_spacing * height)

            for y in range(0, height, line_height):
                (text_width, text_height), _ = cv2.getTextSize(
                    self.text, font, font_scale, thickness=2
                )
                x = (width - text_width) // 2  # Center-align the text horizontally
                overlay = np.zeros_like(watermark_image, dtype=np.uint8)
                cv2.putText(
                    overlay,
                    self.text,
                    (x, y + text_height),
                    font,
                    font_scale,
                    (255, 255, 255, int(255 * self.opacity)),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                alpha_overlay = overlay[:, :, 3] / 255.0
                for c in range(3):
                    watermark_image[:, :, c] = (
                        overlay[:, :, c] * alpha_overlay
                        + watermark_image[:, :, c] * (1 - alpha_overlay)
                    ).astype(np.uint8)

        elif self.style == "random":
            # Style 2: Random placement with density parameter
            font = cv2.FONT_HERSHEY_SIMPLEX
            for _ in range(self.density):
                font_scale = random.uniform(0.02 * width / 100, 0.1 * width / 100)
                (text_width, text_height), _ = cv2.getTextSize(
                    self.text, font, font_scale, thickness=2
                )
                x = random.randint(0, width - text_width)
                y = random.randint(text_height, height - text_height)
                overlay = np.zeros_like(watermark_image, dtype=np.uint8)
                cv2.putText(
                    overlay,
                    self.text,
                    (x, y),
                    font,
                    font_scale,
                    (255, 255, 255, int(255 * self.opacity)),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                alpha_overlay = overlay[:, :, 3] / 255.0
                for c in range(3):
                    watermark_image[:, :, c] = (
                        overlay[:, :, c] * alpha_overlay
                        + watermark_image[:, :, c] * (1 - alpha_overlay)
                    ).astype(np.uint8)

        elif self.style == "center":
            # Style 3: Centered watermark with full width of the image
            font_scale = 0.1 * width / 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), _ = cv2.getTextSize(
                self.text, font, font_scale, thickness=2
            )
            x = (width - text_width) // 2
            y = (height + text_height) // 2
            overlay = np.zeros_like(watermark_image, dtype=np.uint8)
            cv2.putText(
                overlay,
                self.text,
                (x, y),
                font,
                font_scale,
                (255, 255, 255, int(255 * self.opacity)),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            alpha_overlay = overlay[:, :, 3] / 255.0
            for c in range(3):
                watermark_image[:, :, c] = (
                    overlay[:, :, c] * alpha_overlay
                    + watermark_image[:, :, c] * (1 - alpha_overlay)
                ).astype(np.uint8)

        output_image = cv2.cvtColor(watermark_image, cv2.COLOR_BGRA2BGR)
        return output_image


class BaseRain(ImageOnlyTransform):
    """
    Base class for different types of rain effects.
    """

    def __init__(
        self,
        slant_range=(-10, 10),
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=7,
        brightness_coefficient=0.7,
        drop_length=20,
        always_apply=False,
        p=1,
    ):
        super(BaseRain, self).__init__(always_apply, p)
        self.slant_range = slant_range
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.drop_length = drop_length

    def apply_rain(self, image, num_drops):
        rain_transform = A.RandomRain(
            slant_range=self.slant_range,
            drop_length=self.drop_length,
            drop_width=self.drop_width,
            drop_color=self.drop_color,
            blur_value=self.blur_value,
            brightness_coefficient=self.brightness_coefficient,
            rain_type="default",
            p=1.0,
        )

        # Apply the rain effect
        augmented = rain_transform(image=image)
        return augmented["image"]


class DrizzleRain(BaseRain):
    """
    Light drizzle effect with small, sparse rain drops.
    """

    def __init__(self, drop_length=10, **kwargs):
        super(DrizzleRain, self).__init__(drop_length=drop_length, **kwargs)

    def apply(self, image, **params):
        height, width = image.shape[:2]
        area = height * width
        num_drops = area // 770
        return self.apply_rain(image, num_drops)


class HeavyRain(BaseRain):
    """
    Heavy rain effect with medium-sized, more frequent rain drops.
    """

    def __init__(self, drop_length=30, **kwargs):
        super(HeavyRain, self).__init__(drop_length=drop_length, **kwargs)

    def apply(self, image, **params):
        height, width = image.shape[:2]
        num_drops = (width * height) // 600
        return self.apply_rain(image, num_drops)


class TorrentialRain(BaseRain):
    """
    Torrential rain effect with large, dense rain drops.
    """

    def __init__(self, drop_length=60, **kwargs):
        super(TorrentialRain, self).__init__(drop_length=drop_length, **kwargs)

    def apply(self, image, **params):
        height, width = image.shape[:2]
        area = height * width
        num_drops = area // 500
        return self.apply_rain(image, num_drops)
