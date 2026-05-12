import cv2
import numpy as np
import os

ASSETS = "assets/output"

def save(name: str, image: np.ndarray):
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    cv2.imwrite(path, image)


def fixed_background(files, threshold: int = 30, save_at: list[int] = []):
    frames = iter_frames(files)
    background = next(frames)

    for i, frame in enumerate(frames):
        diff  = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        diff  = diff.astype(np.uint8)
        mask  = (diff > threshold).astype(np.uint8) * 255

        if i in save_at:
            save(f"fixed_frame{i:04d}_original.jpg",    frame)
            save(f"fixed_frame{i:04d}_background.jpg",  background)
            save(f"fixed_frame{i:04d}_mask.jpg",        mask)

        yield mask

def temporal_averaging(files, threshold: int = 30, save_at: list[int] = []):
    frames = iter_frames(files)

    first = next(frames)
    window = first[:, :, np.newaxis]

    for i, frame in enumerate(frames, start=1):
        background = np.mean(window, axis=2).astype(np.uint8)

        diff = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        diff = diff.astype(np.uint8)
        mask = (diff > threshold).astype(np.uint8) * 255

        if i in save_at:
            save(f"mean_frame{i:04d}_original.jpg",   frame)
            save(f"mean_frame{i:04d}_background.jpg", background)
            save(f"mean_frame{i:04d}_mask.jpg",       mask)

        new_frame = frame[:, :, np.newaxis]
        window = np.concatenate([window, new_frame], axis=2)

        yield mask


def temporal_median(files, threshold: int = 30, save_at: list[int] = []):
    frames = iter_frames(files)

    first = next(frames)
    window = first[:, :, np.newaxis]

    for i, frame in enumerate(frames, start=1):
        background = np.median(window, axis=2).astype(np.uint8)

        diff = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        diff = diff.astype(np.uint8)
        mask = (diff > threshold).astype(np.uint8) * 255

        if i in save_at:
            save(f"median_frame{i:04d}_original.jpg",   frame)
            save(f"median_frame{i:04d}_background.jpg", background)
            save(f"median_frame{i:04d}_mask.jpg",       mask)

        new_frame = frame[:, :, np.newaxis]
        window = np.concatenate([window, new_frame], axis=2)

        yield mask



def open_frames_dir(path: str):
    files = []  #FILES MUST BE ZERO PADDED!!!
    for f in os.listdir(path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            files.append(os.path.join(path, f))
    files = sorted(files)

    if not files:
        raise IOError(f"No image files found in: {path}")

    first = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    height, width = first.shape
    print(f"Opened: {path}")
    print(f"  Resolution : {width}x{height}")
    print(f"  Frame count: {len(files)}")

    return files, width, height


def iter_frames(files):
    for f in files:
        frame = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Warning: could not read {f}, skipping")
            continue
        yield frame

    
if __name__ == "__main__":

    files, w, h = open_frames_dir("assets/birds")
    save_at = [1, 20, 40, 68]  # sample frames, adjust per image

    for i, mask in enumerate(temporal_averaging(files, threshold=30, save_at=save_at)):
        cv2.imshow("Foreground Mask", mask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()