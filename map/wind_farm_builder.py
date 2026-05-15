# ============================================================
# Ручная геопривязка картинки:
# 1) кликаем контрольные точки
# 2) вводим их реальные lat/lon
# 3) кликаем ветряки
# 4) получаем координаты ветряков
# ============================================================



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
WIND_DATA_DIR = BASE_DIR / "wind_data"
WIND_VIZ_DIR = BASE_DIR / "wind_viz"

IMAGE_PATH = WIND_VIZ_DIR / "wind_farm_map.png"          # путь к картинке
N_CONTROL_POINTS = 5                # минимум 4, лучше 5-8
N_TURBINES = 26                      # сколько ветряков будешь кликать

OUTPUT_TURBINES_CSV = WIND_DATA_DIR / "wind_farm_cords.csv"
OUTPUT_GCP_CSV = WIND_DATA_DIR / "wind_farm_anchors.csv"
OUTPUT_IMAGE = WIND_VIZ_DIR / "wind_farm_marked_map.png"

# ============================================================
# ФУНКЦИИ ПРОЕКЦИИ WEB MERCATOR
# ============================================================

def lon_to_merc_x(lon):
    return np.radians(lon)


def lat_to_merc_y(lat):
    lat = np.clip(lat, -85.05112878, 85.05112878)
    lat_rad = np.radians(lat)
    return np.log(np.tan(np.pi / 4 + lat_rad / 2))


def merc_x_to_lon(x):
    return np.degrees(x)


def merc_y_to_lat(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def latlon_to_merc(lat, lon):
    x = lon_to_merc_x(lon)
    y = lat_to_merc_y(lat)
    return x, y


def merc_to_latlon(x, y):
    lon = merc_x_to_lon(x)
    lat = merc_y_to_lat(y)
    return lat, lon


# ============================================================
# HOMOGRAPHY
# ============================================================

def normalize_points(points):
    points = np.asarray(points, dtype=float)

    mean = points.mean(axis=0)
    centered = points - mean
    mean_dist = np.sqrt((centered ** 2).sum(axis=1)).mean()

    scale = 1.0 if mean_dist == 0 else np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1],
    ])

    points_h = np.column_stack([points, np.ones(len(points))])
    norm_points = (T @ points_h.T).T[:, :2]

    return norm_points, T


def fit_homography(src_xy, dst_xy):
    """
    src_xy: координаты в Web Mercator
    dst_xy: пиксели на изображении
    """
    src_xy = np.asarray(src_xy, dtype=float)
    dst_xy = np.asarray(dst_xy, dtype=float)

    if len(src_xy) < 4:
        raise ValueError("Для homography нужно минимум 4 контрольные точки.")

    src_norm, T_src = normalize_points(src_xy)
    dst_norm, T_dst = normalize_points(dst_xy)

    A = []

    for (x, y), (u, v) in zip(src_norm, dst_norm):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H = H / H[2, 2]

    return H


def apply_homography(H, xy):
    xy = np.asarray(xy, dtype=float)

    points_h = np.column_stack([xy, np.ones(len(xy))])
    out = (H @ points_h.T).T
    out = out[:, :2] / out[:, 2:3]

    return out


# ============================================================
# ЗАГРУЗКА КАРТИНКИ
# ============================================================

img = Image.open(IMAGE_PATH).convert("RGB")
img_w, img_h = img.size

print("Размер изображения:", img_w, "x", img_h)

# ============================================================
# 1. КЛИКАЕМ КОНТРОЛЬНЫЕ ТОЧКИ
# ============================================================

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(img)
ax.set_title(
    f"Кликни {N_CONTROL_POINTS} контрольных точек.\n"
    "Лучше брать точки по краям карты: углы дорог, здания, заметные объекты."
)
ax.set_xlim(0, img_w)
ax.set_ylim(img_h, 0)
ax.grid(alpha=0.25)

control_pixels = plt.ginput(N_CONTROL_POINTS, timeout=0)
plt.close(fig)

control_pixels = np.asarray(control_pixels, dtype=float)

print("Кликнутые пиксели контрольных точек:")
for i, (x, y) in enumerate(control_pixels, 1):
    print(f"GCP{i}: x={x:.2f}, y={y:.2f}")

# ============================================================
# 2. ВВОДИМ РЕАЛЬНЫЕ КООРДИНАТЫ КОНТРОЛЬНЫХ ТОЧЕК
# ============================================================

control_rows = []

print("\nТеперь введи реальные координаты каждой контрольной точки.")
print("Формат: широта, долгота")
print("Пример: 46.867904, 38.718224\n")

for i, (x, y) in enumerate(control_pixels, 1):
    print(f"Контрольная точка GCP{i}")
    lat = float(input("lat: ").replace(",", "."))
    lon = float(input("lon: ").replace(",", "."))

    control_rows.append({
        "name": f"GCP{i}",
        "lat": lat,
        "lon": lon,
        "x": x,
        "y": y,
    })

gcp_df = pd.DataFrame(control_rows)

gcp_df.to_csv(OUTPUT_GCP_CSV, index=False, encoding="utf-8-sig")
print("Контрольные точки сохранены:", OUTPUT_GCP_CSV)

# ============================================================
# 3. СТРОИМ ПРЕОБРАЗОВАНИЕ КООРДИНАТЫ <-> ПИКСЕЛИ
# ============================================================

src_merc = []

for _, row in gcp_df.iterrows():
    mx, my = latlon_to_merc(row["lat"], row["lon"])
    src_merc.append([mx, my])

src_merc = np.asarray(src_merc, dtype=float)
dst_pixels = gcp_df[["x", "y"]].to_numpy(dtype=float)

H_geo_to_pixel = fit_homography(src_merc, dst_pixels)
H_pixel_to_geo = np.linalg.inv(H_geo_to_pixel)

# Проверка ошибки на контрольных точках
pred_pixels = apply_homography(H_geo_to_pixel, src_merc)

gcp_df["x_pred"] = pred_pixels[:, 0]
gcp_df["y_pred"] = pred_pixels[:, 1]
gcp_df["pixel_error"] = np.sqrt(
    (gcp_df["x"] - gcp_df["x_pred"]) ** 2
    + (gcp_df["y"] - gcp_df["y_pred"]) ** 2
)

print("\nОшибка привязки на контрольных точках:")

print("Средняя ошибка, пикселей:", gcp_df["pixel_error"].mean())
print("Максимальная ошибка, пикселей:", gcp_df["pixel_error"].max())

# ============================================================
# 4. КЛИКАЕМ ВЕТРЯКИ
# ============================================================

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(img)

ax.scatter(
    gcp_df["x"],
    gcp_df["y"],
    s=120,
    marker="x",
    linewidth=2.5,
    label="Контрольные точки"
)

for _, row in gcp_df.iterrows():
    ax.text(
        row["x"] + 8,
        row["y"] + 8,
        row["name"],
        fontsize=9,
        color="yellow",
        bbox=dict(facecolor="black", alpha=0.65, pad=2)
    )

ax.set_title(
    f"Кликни {N_TURBINES} ветряков.\n"
    "Кликай в центр основания/точки ветряка."
)
ax.set_xlim(0, img_w)
ax.set_ylim(img_h, 0)
ax.grid(alpha=0.25)
ax.legend()

turbine_pixels = plt.ginput(N_TURBINES, timeout=0)
plt.close(fig)

turbine_pixels = np.asarray(turbine_pixels, dtype=float)

# ============================================================
# 5. ПЕРЕВОДИМ ПИКСЕЛИ ВЕТРЯКОВ В LAT/LON
# ============================================================

turbine_merc = apply_homography(H_pixel_to_geo, turbine_pixels)

turbine_rows = []

for i, ((x, y), (mx, my)) in enumerate(zip(turbine_pixels, turbine_merc), 1):
    lat, lon = merc_to_latlon(mx, my)

    turbine_rows.append({
        "id": f"T{i}",
        "x": x,
        "y": y,
        "lat": lat,
        "lon": lon,
    })

turbines_df = pd.DataFrame(turbine_rows)

turbines_df.to_csv(OUTPUT_TURBINES_CSV, index=False, encoding="utf-8-sig")

print("\nКоординаты ветряков:")

print("Ветряки сохранены:", OUTPUT_TURBINES_CSV)

# ============================================================
# 6. ФИНАЛЬНАЯ ОТРИСОВКА
# ============================================================

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(img)

ax.scatter(
    gcp_df["x"],
    gcp_df["y"],
    s=120,
    marker="x",
    linewidth=2.5,
    label="Контрольные точки"
)

ax.scatter(
    turbines_df["x"],
    turbines_df["y"],
    s=70,
    marker="o",
    edgecolor="black",
    linewidth=1.2,
    label="Ветряки"
)

for _, row in turbines_df.iterrows():
    ax.text(
        row["x"] + 6,
        row["y"] - 6,
        row["id"],
        fontsize=8,
        weight="bold",
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, pad=1.5)
    )

for _, row in gcp_df.iterrows():
    ax.text(
        row["x"] + 8,
        row["y"] + 8,
        row["name"],
        fontsize=9,
        color="yellow",
        bbox=dict(facecolor="black", alpha=0.65, pad=2)
    )

ax.set_title("Геопривязка: контрольные точки и ветряки")
ax.set_xlim(0, img_w)
ax.set_ylim(img_h, 0)
ax.grid(alpha=0.25)
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=200, bbox_inches="tight")
plt.show()

print("Картинка сохранена:", OUTPUT_IMAGE)
