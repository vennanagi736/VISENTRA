from PIL import Image, ImageDraw
import os

INPUT_DIR = r"V:\visentra\CLRNet\test_images"
OUTPUT = r"V:\visentra\CLRNet\test_images\all_lane_images.jpg"

images = [
    "lane.jpg",
    "lane4.jpg",
    "lane5.jpg",
    "lane6.jpg",
    "lane7.jpg",
    "lane8.jpg",
    "lane9.jpg",
    "lane10.jpg",
    "lane11.jpg"
]

# Grid: 3 columns × 3 rows
cols = 3
rows = 3

cell_w = 500
cell_h = 330

canvas = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white"
)

draw = ImageDraw.Draw(canvas)

for i, filename in enumerate(images):

    path = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(path):
        print("Missing:", filename)
        continue

    img = Image.open(path).convert("RGB")

    # Keep aspect ratio
    img.thumbnail((cell_w - 20, cell_h - 45))

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    # Center image
    image_x = x + (cell_w - img.width) // 2
    image_y = y + 5

    canvas.paste(
        img,
        (image_x, image_y)
    )

    # Filename
    draw.text(
        (x + 10, y + cell_h - 30),
        filename,
        fill="black"
    )

canvas.save(
    OUTPUT,
    quality=95
)

print()
print("Combined image created:")
print(OUTPUT)