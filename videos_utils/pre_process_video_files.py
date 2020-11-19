import os
import cv2
import random

import tqdm
import face_recognition

from video_frames_manager import VideoFramesManager


class MoreThanOneFaceError(Exception):
    pass


class NoFaceFoundError(Exception):
    pass


class CorreclyFoundMoreThanOneFaceError(Exception):
    pass


ORIGINAL_VIDEOS_PATH = "data/faces/videos/original"
MANIPULATED_VIDEOS_PATH = "data/faces/videos/manipulated"

ORIGINAL_FRAMES_PATH = "data/faces/frames/original"
MANIPULATED_FRAMES_PATH = "data/faces/frames/manipulated"

ORIGINAL_FACE_CROPS_PATH = "data/faces/face_crops/original"
MANIPULATED_FACE_CROPS_PATH = "data/faces/face_crops/manipulated"

MORE_THAN_ONE_FACE_VIDS = "data/faces/more_than_one_face.txt"


def crop_face_from_image(image):
    facial_landmarks = face_recognition.face_locations(image)

    if len(facial_landmarks) > 1:
        raise MoreThanOneFaceError
    if len(facial_landmarks) == 0:
        raise NoFaceFoundError
    bounding_box = facial_landmarks[0]

    left, top, right, bottom = bounding_box
    h, w, _ = image.shape
    extended_left = max(int(left - w * 0.1), 0)
    extended_right = min(int(right + w * 0.1), w)
    extended_bottom = max(int(bottom - 0.1 * h), 0)
    extended_top = min(int(top + h * 0.1), h)
    face_crop = image[extended_left:extended_right,
                      extended_bottom:extended_top]
    return face_crop


def crop_faces_from_frames(frames_dir, face_crops_dir,
                           num_of_frames_to_crop_from=100):
    frames_list = os.listdir(frames_dir)
    frames_to_crop_faces_from = random.sample(frames_list,
                                              num_of_frames_to_crop_from)
    print(f"Cropping faces from {frames_dir}")
    for frame in tqdm.tqdm(frames_to_crop_faces_from):
        frame_path = os.path.join(frames_dir, frame)
        image = cv2.imread(frame_path)

        try:
            face_crop = crop_face_from_image(image)
        except NoFaceFoundError:
            # don't care if could not find face
            pass
        except MoreThanOneFaceError:
            with open(MORE_THAN_ONE_FACE_VIDS, 'a+') as f:
                f.writelines(frames_dir + '\n')
            break
        # write the face crop to the appropriate file
        face_crop_path = os.path.join(face_crops_dir, frame)
        cv2.imwrite(face_crop_path, face_crop)


def create_all_crops(frames_directories, face_crops_dirs_path):
    for frames_dir in frames_directories:
        video_identity = os.path.abspath(frames_dir).split('/')[-1]
        more_than_one_face_list = []
        with open(MORE_THAN_ONE_FACE_VIDS, 'r') as f:
            for line in f:
                more_than_one_face_list.append(line[:-len('\n')])
        if video_identity in more_than_one_face_list:
            print(f"Do not crop {video_identity} cause found more than one "
                  f"face.. Skipped.")
            continue
        face_crops_dir = os.path.join(face_crops_dirs_path, video_identity)
        if not os.path.exists(face_crops_dir):
            print(f"Creating: {face_crops_dir}")
            os.mkdir(face_crops_dir)
            crop_faces_from_frames(frames_dir, face_crops_dir)
        else:
            print(f"Face crops for {frames_dir} already exist.")


def create_all_frames(videos_directory, frames_directory):
    videos_list = os.listdir(videos_directory)
    videos_list.sort(key=lambda x: int(x.split(".")[0]))
    for video in tqdm.tqdm(videos_list):
        full_video_path = os.path.join(videos_directory, video)
        full_frames_dir_path = os.path.join(frames_directory,
                                            video[:-len(".mp4")])
        if not os.path.exists(full_frames_dir_path):
            print(f"Splitting {video} to frames.")
            os.mkdir(full_frames_dir_path)
            VideoFramesManager.extract_images_from_video(full_video_path,
                                                         full_frames_dir_path)
        else:
            print(f"Frames for {video} already exist in {full_frames_dir_path}")


def main():
    # create_all_frames(ORIGINAL_VIDEOS_PATH, ORIGINAL_FRAMES_PATH)
    # create_all_frames(MANIPULATED_VIDEOS_PATH, MANIPULATED_FRAMES_PATH)
    #
    # original_frames_directories = os.listdir(ORIGINAL_FRAMES_PATH)
    # original_frames_directories.sort(key=lambda x: int(x.split(".")[0]))
    # original_frames_directories = [os.path.join(ORIGINAL_FRAMES_PATH, x) for
    #                                x in original_frames_directories]
    # create_all_crops(original_frames_directories, ORIGINAL_FACE_CROPS_PATH)

    manipulated_frames_directories = os.listdir(MANIPULATED_FRAMES_PATH)
    manipulated_frames_directories.sort(key=lambda x: int(x.split(".")[0]))
    manipulated_frames_directories = [os.path.join(MANIPULATED_FRAMES_PATH,
                                                   x) for x in
                                      manipulated_frames_directories]
    create_all_crops(manipulated_frames_directories,
                     MANIPULATED_FACE_CROPS_PATH)


if __name__ == '__main__':
    main()
