import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A

from src.albutortion import (
    RandomCirclesNoise,
    RandomHorizontalStripes,
    RandomHorizontalLines,
    RandomHorizontalRain,
    RandomDis,
    Checks,
    RandomDottedCheckerboard,
    RandomCheckerboard,
    RandomCheckerboardInCheckerboard,
    RandomDotGrid,
    RandomGridOverlay,
    DrizzleRain,
    HeavyRain,
    TorrentialRain,
    DirtyWindow,
    VintageFilmEffect,
    WindowTexture,
    WatermarkEffect,
    Movie,
    ColorReduce,
    Reflections,
    FocusDrift,
    GeoWarping,
    Compression,
    Sharpen,
)

TEXTURE_DIR = "textures"
textures = [
    os.path.join(TEXTURE_DIR, file)
    for file in os.listdir(TEXTURE_DIR)
    if file.endswith((".jpg", ".png"))
]

TEXTURE_TRANSFORMATIONS = [
    (
        f"WindowTexture_{os.path.splitext(os.path.basename(texture))[0]}",
        WindowTexture(texture_path=texture, opacity=0.5),
    )
    for texture in textures
]
# List of transformations
TRANSFORMATIONS = [
    # white dots
    (
        "RCirc_1000_5_01_w_10",
        RandomCirclesNoise(
            num_circles=1000,
            min_radius=5,
            max_radius=5,
            alpha=0.1,
            color=(255, 255, 255),
            offset=10,
            p=1.0,
        ),
    ),
    (
        "RCirc_800_7_01_w_12",
        RandomCirclesNoise(
            num_circles=800,
            min_radius=7,
            max_radius=7,
            alpha=0.1,
            color=(255, 255, 255),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_600_10_01_w_15",
        RandomCirclesNoise(
            num_circles=600,
            min_radius=10,
            max_radius=10,
            alpha=0.1,
            color=(255, 255, 255),
            offset=15,
            p=1.0,
        ),
    ),
    (
        "RCirc_1000_5_02_w_12",
        RandomCirclesNoise(
            num_circles=1000,
            min_radius=5,
            max_radius=5,
            alpha=0.2,
            color=(255, 255, 255),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_800_7_02_w_12",
        RandomCirclesNoise(
            num_circles=800,
            min_radius=7,
            max_radius=7,
            alpha=0.2,
            color=(255, 255, 255),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_600_10_02_w_15",
        RandomCirclesNoise(
            num_circles=600,
            min_radius=10,
            max_radius=10,
            alpha=0.2,
            color=(255, 255, 255),
            offset=15,
            p=1.0,
        ),
    ),
    (
        "RCirc_1000_5_03_w_10",
        RandomCirclesNoise(
            num_circles=1000,
            min_radius=5,
            max_radius=5,
            alpha=0.3,
            color=(255, 255, 255),
            offset=10,
            p=1.0,
        ),
    ),
    (
        "RCirc_800_7_03_w_12",
        RandomCirclesNoise(
            num_circles=800,
            min_radius=7,
            max_radius=7,
            alpha=0.3,
            color=(255, 255, 255),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_600_10_03_w_15",
        RandomCirclesNoise(
            num_circles=600,
            min_radius=10,
            max_radius=10,
            alpha=0.3,
            color=(255, 255, 255),
            offset=15,
            p=1.0,
        ),
    ),
    # - black dots
    (
        "RCirc_1000_5_01_b_10",
        RandomCirclesNoise(
            num_circles=1000,
            min_radius=5,
            max_radius=5,
            alpha=0.1,
            color=(0, 0, 0),
            offset=10,
            p=1.0,
        ),
    ),
    (
        "RCirc_800_7_01_b_12",
        RandomCirclesNoise(
            num_circles=800,
            min_radius=7,
            max_radius=7,
            alpha=0.1,
            color=(0, 0, 0),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_600_10_01_b_15",
        RandomCirclesNoise(
            num_circles=600,
            min_radius=10,
            max_radius=10,
            alpha=0.1,
            color=(0, 0, 0),
            offset=15,
            p=1.0,
        ),
    ),
    (
        "RCirc_1000_5_02_b_12",
        RandomCirclesNoise(
            num_circles=1000,
            min_radius=5,
            max_radius=5,
            alpha=0.2,
            color=(0, 0, 0),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_800_7_02_b_12",
        RandomCirclesNoise(
            num_circles=800,
            min_radius=7,
            max_radius=7,
            alpha=0.2,
            color=(0, 0, 0),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_600_10_02_b_15",
        RandomCirclesNoise(
            num_circles=600,
            min_radius=10,
            max_radius=10,
            alpha=0.2,
            color=(0, 0, 0),
            offset=15,
            p=1.0,
        ),
    ),
    (
        "RCirc_1000_5_03_b_10",
        RandomCirclesNoise(
            num_circles=1000,
            min_radius=5,
            max_radius=5,
            alpha=0.3,
            color=(0, 0, 0),
            offset=10,
            p=1.0,
        ),
    ),
    (
        "RCirc_800_7_03_b_12",
        RandomCirclesNoise(
            num_circles=800,
            min_radius=7,
            max_radius=7,
            alpha=0.3,
            color=(0, 0, 0),
            offset=12,
            p=1.0,
        ),
    ),
    (
        "RCirc_600_10_03_b_15",
        RandomCirclesNoise(
            num_circles=600,
            min_radius=10,
            max_radius=10,
            alpha=0.3,
            color=(0, 0, 0),
            offset=15,
            p=1.0,
        ),
    ),
    # 2 x RandomHorizontalStripes
    ("HorizontalStripes_50_5_10_01", RandomHorizontalStripes(num_stripes=50, min_thickness=5, max_thickness=10,
                                                        alpha=0.1, p=1.0)),
    ("HorizontalStripes_50_10_20_01", RandomHorizontalStripes(num_stripes=50, min_thickness=10, max_thickness=20,
                                                        alpha=0.1, p=1.0)),
    ("HorizontalStripes_50_5_10_02", RandomHorizontalStripes(num_stripes=50, min_thickness=5, max_thickness=10,
                                                        alpha=0.2, p=1.0)),
    ("HorizontalStripes_50_10_20_02", RandomHorizontalStripes(num_stripes=50, min_thickness=10, max_thickness=20,
                                                        alpha=0.2, p=1.0)),
    ("HorizontalStripes_50_5_10_03", RandomHorizontalStripes(num_stripes=50, min_thickness=5, max_thickness=10,
                                                        alpha=0.3, p=1.0)),
    ("HorizontalStripes_50_10_20_03", RandomHorizontalStripes(num_stripes=50, min_thickness=10, max_thickness=20,
                                                        alpha=0.3, p=1.0)),
    # RandomCheckerboard
    ("RandomCheckerboard_5_01", RandomCheckerboard(square_size=5, alpha=0.1)),
    ("RandomCheckerboard_10_01", RandomCheckerboard(square_size=10, alpha=0.1)),
    ("RandomCheckerboard_15_01", RandomCheckerboard(square_size=15, alpha=0.1)),
    ("RandomCheckerboard_25_01", RandomCheckerboard(square_size=25, alpha=0.1)),
    ("RandomCheckerboard_5_02", RandomCheckerboard(square_size=5, alpha=0.2)),
    ("RandomCheckerboard_10_02", RandomCheckerboard(square_size=10, alpha=0.2)),
    ("RandomCheckerboard_15_02", RandomCheckerboard(square_size=15, alpha=0.2)),
    ("RandomCheckerboard_25_02", RandomCheckerboard(square_size=25, alpha=0.2)),
    ("RandomCheckerboard_5_03", RandomCheckerboard(square_size=5, alpha=0.3)),
    ("RandomCheckerboard_10_03", RandomCheckerboard(square_size=10, alpha=0.3)),
    ("RandomCheckerboard_15_03", RandomCheckerboard(square_size=15, alpha=0.3)),
    ("RandomCheckerboard_25_03", RandomCheckerboard(square_size=25, alpha=0.3)),
    # RandomCheckerboardInCheckerboard
        (
            "RandomCheckerboardInCheckerboard_40_10_01",
            RandomCheckerboardInCheckerboard(
                square_size=40, inner_square_size=10, alpha=0.1
            ),
        ),
        (
            "RandomCheckerboardInCheckerboard_40_5_01",
            RandomCheckerboardInCheckerboard(
                square_size=40, inner_square_size=5, alpha=0.1
            ),
        ),
        (
            "RandomCheckerboardInCheckerboard_40_10_02",
            RandomCheckerboardInCheckerboard(
                square_size=40, inner_square_size=10, alpha=0.2
            ),
        ),
        (
            "RandomCheckerboardInCheckerboard_40_5_02",
            RandomCheckerboardInCheckerboard(
                square_size=40, inner_square_size=5, alpha=0.2
            ),
        ),
        (
            "RandomCheckerboardInCheckerboard_40_10_03",
            RandomCheckerboardInCheckerboard(
                square_size=40, inner_square_size=10, alpha=0.3
            ),
        ),
        (
            "RandomCheckerboardInCheckerboard_40_5_03",
            RandomCheckerboardInCheckerboard(
                square_size=40, inner_square_size=5, alpha=0.3
            ),
        ),
    #RandomDotGrid
        (
            "DotGrid_10_1_01",
            RandomDotGrid(dot_spacing=10, dot_radius=1, color=(255, 255, 255), alpha=0.1),
        ),
        (
            "DotGrid_10_2_01",
            RandomDotGrid(dot_spacing=10, dot_radius=2, color=(255, 255, 255), alpha=0.1),
        ),
        (
            "DotGrid_20_2_01",
            RandomDotGrid(dot_spacing=20, dot_radius=2, color=(255, 255, 255), alpha=0.1),
        ),
        (
            "DotGrid_20_5_01",
            RandomDotGrid(dot_spacing=20, dot_radius=5, color=(255, 255, 255), alpha=0.1),
        ),
    (
            "DotGrid_10_1_02",
            RandomDotGrid(dot_spacing=10, dot_radius=1, color=(255, 255, 255), alpha=0.2),
        ),
        (
            "DotGrid_10_2_02",
            RandomDotGrid(dot_spacing=10, dot_radius=2, color=(255, 255, 255), alpha=0.2),
        ),
        (
            "DotGrid_20_2_02",
            RandomDotGrid(dot_spacing=20, dot_radius=2, color=(255, 255, 255), alpha=0.2),
        ),
        (
            "DotGrid_20_5_02",
            RandomDotGrid(dot_spacing=20, dot_radius=5, color=(255, 255, 255), alpha=0.2),
        ),
    (
            "DotGrid_10_1_03",
            RandomDotGrid(dot_spacing=10, dot_radius=1, color=(255, 255, 255), alpha=0.3),
        ),
        (
            "DotGrid_10_2_03",
            RandomDotGrid(dot_spacing=10, dot_radius=2, color=(255, 255, 255), alpha=0.3),
        ),
        (
            "DotGrid_20_2_03",
            RandomDotGrid(dot_spacing=20, dot_radius=2, color=(255, 255, 255), alpha=0.3),
        ),
        (
            "DotGrid_20_5_03",
            RandomDotGrid(dot_spacing=20, dot_radius=5, color=(255, 255, 255), alpha=0.3),
        ),
    #RandomDottedCheckerboard
        (
            "RandomDottedCheckerboard_20_5_1_01",
            RandomDottedCheckerboard(
                square_size=20, dot_spacing=5, dot_radius=1, alpha=0.1
            ),
        ),
        (
            "RandomDottedCheckerboard_40_10_2_01",
            RandomDottedCheckerboard(
                square_size=40, dot_spacing=10, dot_radius=2, alpha=0.1
            ),
        ),
    (
            "RandomDottedCheckerboard_20_5_1_02",
            RandomDottedCheckerboard(
                square_size=20, dot_spacing=5, dot_radius=1, alpha=0.2
            ),
        ),
        (
            "RandomDottedCheckerboard_40_10_2_02",
            RandomDottedCheckerboard(
                square_size=40, dot_spacing=10, dot_radius=2, alpha=0.2
            ),
        ),
    (
            "RandomDottedCheckerboard_20_5_1_03",
            RandomDottedCheckerboard(
                square_size=20, dot_spacing=5, dot_radius=1, alpha=0.3
            ),
        ),
        (
            "RandomDottedCheckerboard_40_10_2_03",
            RandomDottedCheckerboard(
                square_size=40, dot_spacing=10, dot_radius=2, alpha=0.3
            ),
        ),
    #RandomGridOverlay
    (
        "GridOverlay_10_2_02",
        RandomGridOverlay(
            grid_spacing=10, line_thickness=2, color=(255, 255, 255), alpha=0.2
        ),
    ),
    (
        "GridOverlay_20_2_02",
        RandomGridOverlay(
            grid_spacing=20, line_thickness=2, color=(255, 255, 255), alpha=0.2
        ),
    ),
    (
        "GridOverlay_20_5_02",
        RandomGridOverlay(
            grid_spacing=20, line_thickness=5, color=(255, 255, 255), alpha=0.2
        ),
    ),
    (
        "GridOverlay_40_2_02",
        RandomGridOverlay(
            grid_spacing=40, line_thickness=2, color=(255, 255, 255), alpha=0.2
        ),
    ),
    (
        "GridOverlay_80_2_02",
        RandomGridOverlay(
            grid_spacing=80, line_thickness=2, color=(255, 255, 255), alpha=0.2
        ),
    ),
    (
        "GridOverlay_40_5_02",
        RandomGridOverlay(
            grid_spacing=40, line_thickness=5, color=(255, 255, 255), alpha=0.2
        ),
    ),
    # 4 x rains
    (
        "DrizzleRain_2_20_0",
        DrizzleRain(slant_range=(-5, 5), drop_width=2, drop_length=20,
                    p=1.0),
    ),
    (
        "DrizzleRain_2_50_0",
        DrizzleRain(slant_range=(-5, 5), drop_width=2, drop_length=50, p=1.0),
    ),
    ("HeavyRain", HeavyRain(slant_range=(-10, 10), p=1.0)),
    ("TorrentialRain", TorrentialRain(slant_range=(-15, 15), p=1.0)),
    # DirtyWindow
    ("DirtyWindow", DirtyWindow(dirt_density=0.9)),
    # VintageFilmEffect
    ("VintageFilmEffect_01", VintageFilmEffect(noise_intensity=0.1)),
    ("VintageFilmEffect_03", VintageFilmEffect(noise_intensity=0.3)),
    ("VintageFilmEffect_05", VintageFilmEffect(noise_intensity=0.5)),
    #Movie
    ("Movie_01", Movie(intensity=0.1)),
    ("Movie_03", Movie(intensity=0.3)),
    ("Movie_05", Movie(intensity=0.5)),
    #ColorReduce
    ("ColorReduce_4", ColorReduce(levels=4)),
    ("ColorReduce_8", ColorReduce(levels=8)),
    ("ColorReduce_16", ColorReduce(levels=16)),
    #Compression
    ("Compression_10", Compression(quality=10)),
    ("Compression_30", Compression(quality=30)),
    ("Compression_50", Compression(quality=50)),
    # #Reflections
    # ("Reflections", Reflections()),
    #Sharpen
    ("Sharpen_02", Sharpen(intensity=0.2)),
    ("Sharpen_05", Sharpen(intensity=0.5)),
    # Watermark
    ("WatermarkEffect_BS_line", WatermarkEffect(text="BESOS BESOS BESOS", opacity=0.5, style="line_by_line", line_spacing=0.1)),
    ("WatermarkEffect_BS_rand", WatermarkEffect(text="BESOS", opacity=0.3, style="random", density=150)),
    ("WatermarkEffect_R_line", WatermarkEffect(text="REAL REAL REAL REAL", opacity=0.5, style="line_by_line", line_spacing=0.1)),
    ("WatermarkEffect_R_rand", WatermarkEffect(text="REAL", opacity=0.3, style="random", density=150)),
    ("WatermarkEffect_F_line", WatermarkEffect(text="FAKE FAKE FAKE FAKE", opacity=0.5, style="line_by_line", line_spacing=0.1)),
    ("WatermarkEffect_F_rand", WatermarkEffect(text="FAKE", opacity=0.3, style="random", density=150)),
    ("WatermarkEffect_BBC_line", WatermarkEffect(text="BBC BBC BBC BBC BBC", opacity=0.5, style="line_by_line", line_spacing=0.1)),
    ("WatermarkEffect_BBC_rand", WatermarkEffect(text="BBC", opacity=0.3, style="random", density=150)),

    #FocusDrift
    ("FocusDrift_05", FocusDrift(intensity=0.5)), # problem z detekcją twarzy - co robimy??
    # # - done - separetly ("WindowTexture", WindowTexture(texture_path="textures/Texturelabs_Glass_125M.jpg", opacity=0.5)),
    # ("WatermarkEffect", WatermarkEffect(text="BESOS", opacity=0.5)),
    # ("GeoWarping", GeoWarping(intensity=1)),

    ## ("RandomHorizontalLines", RandomHorizontalLines(num_lines=10, min_length=50, max_length=100, alpha=0.2, p=1.0)),
    ## ("RandomHorizontalRain", RandomHorizontalRain(num_lines=50, line_length=30, p=1.0)),
    ## ("RandomDis", RandomDis(frequency=(1, 10), opacity=(0.1, 0.5))),
    ## ("Checks", Checks(size='width', x=20, opacity=0.2)),
    # #
]  + TEXTURE_TRANSFORMATIONS


def apply_and_save(image_name, image, transformation_name, transformation, output_dir="output"):
    """Apply transformation and save image ."""
    os.makedirs(output_dir, exist_ok=True)
    image_tmp = image.copy()
    transform = A.Compose([transformation])
    augmented = transform(image=image_tmp)
    augmented_image = augmented["image"]
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}_{transformation_name}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")
