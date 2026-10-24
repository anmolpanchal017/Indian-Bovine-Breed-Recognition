
from pathlib import Path
import math
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Indian_bovine_breeds" / "Indian_bovine_breeds"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "duplicate_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(REPORT_DIR / "exact_duplicates.csv")

if df.empty:
    print("No exact duplicates to review.")
    raise SystemExit

# Select groups where the same file content has different breed labels.
grouped = df.groupby("sha256", sort=False)
cross_breed_groups = []

for file_hash, group in grouped:
    if group["breed"].nunique() > 1:
        cross_breed_groups.append((file_hash, group))

print(f"Cross-breed groups to review: {len(cross_breed_groups)}")

# Create contact sheets with 4 duplicate groups per page.
GROUPS_PER_PAGE = 4
THUMB_W, THUMB_H = 300, 210
HEADER_H = 55
GAP = 18

for page_start in range(0, len(cross_breed_groups), GROUPS_PER_PAGE):
    page_groups = cross_breed_groups[
        page_start:page_start + GROUPS_PER_PAGE
    ]

    rows = len(page_groups)
    page_width = 2 * THUMB_W + 3 * GAP
    page_height = rows * (THUMB_H + HEADER_H + GAP) + GAP

    sheet = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(sheet)

    for local_idx, (file_hash, group) in enumerate(page_groups):
        y = GAP + local_idx * (THUMB_H + HEADER_H + GAP)

        breeds = sorted(group["breed"].astype(str).unique())
        title = f"Group {page_start + local_idx + 1}: " + ", ".join(breeds)
        draw.text((GAP, y), title[:120], fill="black")

        for image_idx, (_, record) in enumerate(group.iterrows()):
            x = GAP + image_idx * (THUMB_W + GAP)

            if x + THUMB_W > page_width:
                break

            image_path = DATASET_DIR / str(record["path"])

            try:
                with Image.open(image_path) as img:
                    img.seek(0)
                    thumb = img.convert("RGB")
                    thumb.thumbnail((THUMB_W, THUMB_H))

                sheet.paste(thumb, (x, y + HEADER_H))
                draw.text(
                    (x, y + HEADER_H + THUMB_H + 2),
                    f'{record["breed"]}: {record["path"]}'[:65],
                    fill="black",
                )
            except Exception as error:
                draw.text(
                    (x, y + HEADER_H),
                    f"Cannot open image: {error}"[:50],
                    fill="red",
                )

    page_number = page_start // GROUPS_PER_PAGE + 1
    output_path = OUTPUT_DIR / f"review_page_{page_number:03d}.jpg"
    sheet.save(output_path, quality=90)
    print(f"Saved: {output_path}")

print("\nReview sheets created.")
print("Open the images in:", OUTPUT_DIR)
print("No original images were modified.")
