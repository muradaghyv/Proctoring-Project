import proctor
import time

start_time = time.time()

MOUNT = "path_to_main_folder"
INPUT_PATH = MOUNT + "ipath_to_dataset"
OUTPUT_PATH = MOUNT + "output_path"

# when initializing the object, values can be adjusted,
# although they already have default value
proctor = proctor.Proctor(gaze_mid_value=0.5, face_x_mid_value=0.5, 
                          face_y_mid_value=0.5, gaze_max_variance=0.15,
                          face_x_max_variance=0.15, face_y_max_variance=0.2,
                          frame_diff_threshold=0.05, frame_skip_rate=5)

# calibration is optional, you can skip the following two lines of code 
# if you don't want to use this feature.

cal_image_path = "path_to_calibration_image/calibration_image.jpg"
proctor.calibrate(cal_image_path)

proctor.export_all_frames_from_video(video_path=INPUT_PATH + "/Yousef.avi",
                                     output_folder=OUTPUT_PATH,
                                     export_all=False, export_sus_only=False)

del proctor

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
