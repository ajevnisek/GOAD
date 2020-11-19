"""Convert MP4 video to frames and vice versa."""
import os
import argparse
import subprocess
import cv2


class VideoFramesManager(object):
    """Manages all video and frames interactions."""

    @staticmethod
    def extract_images_from_video(path_video_in, path_frames_direcory_out):
        """Extract frames from the video in path_in to the directory in path_out."""
        count = 0
        video_capturer = cv2.VideoCapture(path_video_in)
        success, image = video_capturer.read()
        success = True
        while success:
            # video_capturer.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
            success, image = video_capturer.read()
            if count % 100 == 0 and count != 0:
                print(f'Processed {count} frames: ', success)
            if success:
                cv2.imwrite(os.path.join(path_frames_direcory_out,
                                         f"frame{count}.png"),
                            image)     # save frame as JPEG file
            # cv2.imwrite(path_frames_direcory_out + "\\frame%d_zssr_X2.00X2.00.png" % count, image)  # save frame as JPEG file
            count = count + 1

    @staticmethod
    def convert_frames_to_video(path_frames_directory_in,
                                path_to_video_out,
                                run_create_video_command=False,
                                frames_format="png"):
        """This code assumes that the frames names are frame%d.format."""
        all_frames = [frame for frame in os.listdir(path_frames_directory_in) if frame.endswith("." + frames_format)]
        all_frames.sort(key=lambda x: int(x.split("frame")[-1].split("." + frames_format)[0]))
        # all_frames.sort(key=lambda x: int(x.split("frame")[-1].split("_")[0]))
        with open('input.txt', 'w') as f:
            for frame in all_frames:
                f.write("file '" + os.path.join(path_frames_directory_in, frame) + "'\n")
        if run_create_video_command:
            all_frames = os.listdir(path_frames_directory_in)
            all_frames.sort()
            first_frame = all_frames[0]
            subprocess.run(' ' .join(['ffmpeg',
                             # f'-i {os.path.join(path_frames_directory_in, "frame1.png")}',
                             # f'-i {os.path.join(path_frames_directory_in, "frame600_zssr_X2.00X2.00.png")}',
                             # f'-i {os.path.join(path_frames_directory_in, "frame830_zssr_X2.00X2.00.png")}',
                             f'-i {os.path.join(path_frames_directory_in, first_frame)}',
                             '-f concat',
                             '-r 25',
                             '-i input.txt',
                             '-filter_complex "overlay=5:H-h-5"',
                             '-shortest',
                             f'{path_to_video_out}']), shell=True)
        else:
            print(f"""
Please run:
ffmpeg -i frame0.png -f concat -r 25 -i input.txt -filter_complex "overlay=5:H-h-5" -shortest {path_to_video_out}
                """)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--frames_to_video", help="If set, converts frames to video.", action="store_true")
    a.add_argument("--path_to_video", help="path to video")
    a.add_argument("--path_to_images_directory", help="path to images")
    a.add_argument("--run_create_video_command",
                   help="In case of frames to video conversion, if set runs the video creation command.",
                   action="store_true")
    a.add_argument("--frames_format",
                   help="In case of frames to video conversion, specify the frames file type.", default="png")
    args = a.parse_args()
    print(args)
    if args.frames_to_video:
        VideoFramesManager.convert_frames_to_video(args.path_to_images_directory,
                                                   args.path_to_video,
                                                   args.run_create_video_command,
                                                   args.frames_format)
    else:
        VideoFramesManager.extract_images_from_video(args.path_to_video, args.path_to_images_directory)
