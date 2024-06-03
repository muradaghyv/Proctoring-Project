import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from   shutil import rmtree
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pdf")

def image_diff(image1, image2) -> float:
    """
    Calculates difference of images based on their histograms.
    Grayscale images are desirable for more accurate results.

    Args:
    ->  hist1: histogram of the first image
    ->  hist2: histogram of the second image

    Return:
    ->  diff: The difference value
    """
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # (images, channels, mask, histSize, ranges)
    hist1 = cv2.calcHist([image1_gray], [0], None, [256], [0,256]) 
    hist2 = cv2.calcHist([image2_gray], [0], None, [256], [0,256])
    
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return diff

def index_to_timestamp(frame_index, fps) -> str:
    """
    Converts frame index of a video to time format
    Datetime object is not used, for efficiency, as we only use this data in file naming

    Args:
    ->  frame_index: index of the current frame, must be an integer value
    ->  fps: frame rate of the video, must be an integer value

    Return:
    ->  hour:minute:second, in string format
    """
    num_of_seconds = frame_index // fps
    hours   = f"{num_of_seconds // 3600}".rjust(2, "0")
    minutes = f"{(num_of_seconds % 3600) // 60}".rjust(2, "0")
    seconds = f"{num_of_seconds % 60}".rjust(2, "0")

    return    f"{hours}_{minutes}_{seconds}"

def create_custom_df() -> pd.DataFrame:
    """
    Creates and returns a DataFrame for storing the values of the frames.
    """
    null_data = {
        "Index" :[],
        "Time"  :[],
        "Gaze"  :[],
        "Face_X":[],
        "Face_Y":[],
        "Sus"   :[],
        "Case"  :[]
    }

    df = pd.DataFrame(null_data)

    return df

def export_custom_df(df, path, format="excel") -> None:
    """
    Exports the dataframe to the given path in the desired format

    Args:
    ->  df: DataFrame to export
    ->  path: path to the folder that the dataframe must be exported to
    ->  format: the format of the exported file. 
        Available options: "csv", "excel"
    """
    if format   == "csv":
        df.to_csv(f"{path}/data.csv")

    elif format == "excel":
        with pd.ExcelWriter(f"{path}/data.xlsx", mode="w") as writer:
            df.to_excel(writer, sheet_name="data", index=False)
    
    else:
        print("Format not supported!")

def plot_positions(params, df, export_path) -> None:
    """
    Exports plots of gaze, head x, and head y values from DataFrame df,
    sets lines and colors according to the values from DataFrame params.
    The dots will be green if the fall into the accepted range, 
    otherwise, they will be red.
    There will be two files: 
        plot_detailed.pdf which represents the detailed plots with all timestamps.
        plot_overview.pdf which represents the simplified plots with limited timestamps.

    Args:
    ->  params: the dataframe that contains statistical values.
    ->  df: the dataframe that contains the gaze, head x, and head y values.
    ->  export_path: the path to the folder where the plot will be saved to.
    """
    lookup_table = [
        ["Gaze",   "GAZE_MID_VALUE",   "GAZE_MAX_VARIANCE",   "Gaze position values"],
        ["Face_X", "FACE_X_MID_VALUE", "FACE_X_MAX_VARIANCE", "Head position values, x-axis"],
        ["Face_Y", "FACE_Y_MID_VALUE", "FACE_Y_MAX_VARIANCE", "Head position values, y-axis"]
    ]

    f, axs      = plt.subplots(3, 1, figsize=(len(df.index) // 5 , 25))
    f_s, axs_s  = plt.subplots(3, 1, figsize=(len(df.index) // 10, 25)) # The simpler plots

    for l, ax, ax_s in zip(lookup_table, axs, axs_s):
        max_val = float(params[l[1]].iloc[0]) + float(params[l[2]].iloc[0])
        min_val = float(params[l[1]].iloc[0]) - float(params[l[2]].iloc[0])

        ax.scatter(df['Time'], df[l[0]], s=20, 
                color=["red" if not min_val < val < max_val else "green" for val in df[l[0]]])
        
        ax_s.scatter(df['Time'], df[l[0]], s=20, 
                color=["red" if not min_val < val < max_val else "green" for val in df[l[0]]])

        ax.set_xticks(range(df["Time"].nunique()))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # converting the timestamp format to (hour:minute) and 
        # storing the unique ones 
        time_new = df["Time"].apply(lambda x: x[:5]).unique()
        
        ax_s.set_xticks(np.linspace(0, df["Time"].nunique(), len(time_new)))
        ax_s.set_xticklabels(time_new)
        ax_s.tick_params(axis='x', which='major', labelsize=15)

        ax.grid()
        ax_s.grid()

        ax.set_ylabel(  l[3], fontsize=30)
        ax_s.set_ylabel(l[3], fontsize=30)
        ax_s.set_xlabel("Time (hour:minute)", fontsize=20)

        ax.axhline  (y=max_val, xmin=0, xmax=1, c='b', linewidth=1)
        ax.axhline  (y=min_val, xmin=0, xmax=1, c='b', linewidth=1)
        ax_s.axhline(y=max_val, xmin=0, xmax=1, c='b', linewidth=1)
        ax_s.axhline(y=min_val, xmin=0, xmax=1, c='b', linewidth=1)
    
    f.suptitle(  "Output of The Proctor", fontsize=50)
    f_s.suptitle("Output of The Proctor", fontsize=50)

    f.savefig(  f"{export_path}/plot_detailed.pdf", bbox_inches="tight")
    f_s.savefig(f"{export_path}/plot_overview.pdf", bbox_inches="tight")

    plt.close()
    plt.close() # closing the second plot, preventing the plot from being displayed automatically

class Proctor:
    def __init__(self, gaze_mid_value=0.5, face_x_mid_value=0.5, 
                 face_y_mid_value=0.5, gaze_max_variance=0.1,
                 face_x_max_variance=0.1, face_y_max_variance=0.1,
                 frame_diff_threshold=0.05, frame_skip_rate=5) -> None:

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=2,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5)
        
        self.gaze_mid_value       = gaze_mid_value
        self.face_x_mid_value     = face_x_mid_value
        self.face_y_mid_value     = face_y_mid_value
        self.gaze_max_variance    = gaze_max_variance
        self.face_x_max_variance  = face_x_max_variance
        self.face_y_max_variance  = face_y_max_variance
        self.frame_diff_threshold = frame_diff_threshold
        self.frame_skip_rate      = frame_skip_rate

    def eye_rotation(self, frame, eye_landmarks) -> float:
        """
        Calculates how much the eye is rotated to either direction.
        The logic is based on the color difference between white and black sections of the eye.
        
        Args:
        ->  frame: the image that contains face information
        ->  eye_landmarks: part of the multi_face_landmarks for the eyes

        Return:
        ->  ratio: the ratio of the left part of the eye to the right part. 
            if the person is looking left, it returns a value less than 0.5, 
            if looking right, more than 0.5, if looking straight, returns 0.5.
        """
        eye_region = np.array([
            (int(eye_landmarks[i].x * frame.shape[1]),
             int(eye_landmarks[i].y * frame.shape[0])) \
                for i in range(4)])

        min_x = np.min(eye_region[:, 0])
        max_x = np.max(eye_region[:, 0])
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        eye = frame[min_y:max_y, min_x:max_x]

        eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

        eye_left_part  = eye_gray[:eye.shape[0]//2, :  eye.shape[1]//2]
        eye_right_part = eye_gray[ eye.shape[0]//2  : (eye.shape[0]//2)*2, 
                                   eye.shape[1]//2  : (eye.shape[1]//2)*2]
        
        # a neglegible value is added to the denominator to prevent zero division error.
        # In this case, the nominator will already be zero, so we will get 0 instead of a runtime error.
        ratio = np.sum(eye_right_part) / (np.sum(eye_right_part) + np.sum(eye_left_part) + 1e-3)
        
        return ratio
    
    def face_gaze_detection(self, frame) -> tuple:
        """
        Calculates how much the eyes are rotated to either direction, and
            how much the face is rotated to either direction in each axis.
            face_mesh instance that belongs to this object must already be ready.

        Args:
        ->  frame: the image that contains face information.
        
        Return:
        ->  ratio_gaze: a value between 0 and 1. If value:
            less than ~0.5, the person is looking at the left.
            greater than ~0.5, the person is looking at the right.
            equals to ~0.5, the person is looking at the center.
        ->  ratio_face_x: a value between 0 and 1. If value:
            less than ~0.5, the face is positioned towards the right.
            greater than ~0.5, the face is positioned towards the left.
            equals to ~0.5, the face is positioned towards the center.
        ->  ratio_face_y: a value between 0 and 1. If value:
            less than ~0.5, the face is positioned towards the top.
            greater than ~0.5, the face is positioned towards the bottom.
            equals to ~0.5, the face is positioned towards the center.

        In case of not  detecion of face, all output values will be 0. 
        In case of more than one detecion of face, all output values will be 1. 

        The logic of gaze detection is based on the color difference between white and black sections of the eye.
        The logic of face position detection in x axis is based on the ratio of distances from 
            the leftmost point and the rightmost point of the face.
        The logic of face position detection in y axis is based on the ratio of distances from 
            the upmost point and the lowest point of the face.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) == 1:

                eye_1_ratio = self.eye_rotation(
                    frame, 
                    [results.multi_face_landmarks[0].landmark[x] for x in [154, 157, 161, 163]])
                eye_2_ratio = self.eye_rotation(
                    frame, 
                    [results.multi_face_landmarks[0].landmark[x] for x in [381, 384, 388, 390]])
                
                ratio_gaze = (eye_1_ratio + eye_2_ratio) / 2

                # nose: 4,  left eye leftmost point: 33,    right eye rightmost point: 263
                face_left_depth = results.multi_face_landmarks[0].landmark[33].z - \
                    results.multi_face_landmarks[0].landmark[4].z
                
                face_right_depth = results.multi_face_landmarks[0].landmark[263].z - \
                    results.multi_face_landmarks[0].landmark[4].z
                
                ratio_face_x = face_left_depth/(face_left_depth+face_right_depth + 1e-6)

                face_top_depth = results.multi_face_landmarks[0].landmark[8].z - \
                    results.multi_face_landmarks[0].landmark[4].z
                
                face_bottom_depth = results.multi_face_landmarks[0].landmark[18].z - \
                    results.multi_face_landmarks[0].landmark[4].z
                
                ratio_face_y = face_top_depth/(face_top_depth+face_bottom_depth + 1e-6)

                return ratio_gaze, ratio_face_x, ratio_face_y
            
            elif len(results.multi_face_landmarks) > 1:
                return 1, 1, 1
        
        return -1, -1, -1

    def calibrate(self, calibration_image_path) -> None:
        """
        Gets the middle values from an image that shows the person looking at the screen perfectly.
            As some people may have different face shape, the accuracy of the proctoring system
            with the default values may fall short in some use cases. Therefore, the parameters can 
            be updated via this method. This is optional.
            If the calibration fails due to the detection system problem, a failure message will be printed.

        Args:
        ->  calibration_image_path: path to the image that the face and gaze is towards the camera
        """
        try:
            calibration_image = cv2.imread(calibration_image_path)

        except:
            print("Calibration image not found!")

        try:
            ratio_gaze, ratio_face_x, ratio_face_y = self.face_gaze_detection(calibration_image)
            if 0 < ratio_gaze < 1 and 0 < ratio_face_x < 1 and 0 < ratio_face_y < 1:

                self.gaze_mid_value   = ratio_gaze
                self.face_x_mid_value = ratio_face_x
                self.face_y_mid_value = ratio_face_y

                print("Calibrated successfully!")

            else:
                print("Face not detected for calibration!")

        except:
            print("Calibration failed!")

    def export_frame(self, frame, output_path, timestamp) -> tuple:
        """
        Exports a CV2 frame to the required folder according to the video name, 
            index, and cheating/not cheating.
        
        Args:
        ->  frame: an image from the video containing a human face
        ->  output_path: the folder to export the images
        ->  timestamp: a string that represents from which time of the video the frame is taken

        Return:
        ->  the output of the face_gaze_detection function. Might be useful for further analysis
        """
        case_lookup_table = [
            "not_cheating",
            "cheating",
            "face_not_detected", 
            "another_person_detected",
            "fault"]
        
        suspicious = 0

        try:
            ratio_gaze, ratio_face_x, ratio_face_y = self.face_gaze_detection(frame)
            
            # Face not detected
            if ratio_gaze == -1 or ratio_face_x == -1 or ratio_face_y == -1:
                suspicious = 1
                case_index = 2

            # Another person detected
            elif ratio_gaze == 1 or ratio_face_x == 1 or ratio_face_y == 1:
                suspicious = 1
                case_index = 3

            else:
                cheating = False
                if not self.gaze_mid_value   - self.gaze_max_variance   < ratio_gaze   < self.gaze_mid_value   + self.gaze_max_variance:
                    cheating = True
                if not self.face_x_mid_value - self.face_x_max_variance < ratio_face_x < self.face_x_mid_value + self.face_x_max_variance:
                    cheating = True
                if not self.face_y_mid_value - self.face_y_max_variance < ratio_face_y < self.face_y_mid_value + self.face_y_max_variance:
                    cheating = True

                if cheating:
                    # Cheating
                    suspicious = 1
                    case_index = 1
                else:
                    # Not cheating
                    case_index = 0

            if suspicious == 1:
                case_type = "suspicious_case"
            else:
                case_type = "trustful_case"

            path = f"{output_path}/{case_type}/{case_lookup_table[case_index]}"

            if not os.path.exists(path):
                os.makedirs(path)

            filename = f"{path}/{timestamp}_gaze_{ratio_gaze:.2f}_face_x_{ratio_face_x:.2f}_face_y_{ratio_face_y:.2f}.jpg"
            cv2.imwrite(filename, frame)

            return ratio_gaze, ratio_face_x, ratio_face_y, suspicious, case_lookup_table[case_index]

        except:
            suspicious = 1
            case_type = "suspicious_case"
            # Fault
            case_index = 4

            path = f"{output_path}/{case_type}/{case_lookup_table[case_index]}"
            if not os.path.exists(path):
                os.makedirs(path)

            filename = f"{path}/{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            return -1, -1, -1, suspicious, case_lookup_table[case_index]
    
    def export_params(self, path, format="excel") -> pd.DataFrame:
        """
        Exports the current parameters of the object.
        Can be used for accurate data analysis.

        Args:
        ->  path: the path to the folder where the data must be exported to.
        ->  format: a string to choose any of the supported options:
            "excel" or "csv".
        """
        params = {
            "GAZE_MID_VALUE"       : [self.gaze_mid_value       ],
            "FACE_X_MID_VALUE"     : [self.face_x_mid_value     ],
            "FACE_Y_MID_VALUE"     : [self.face_y_mid_value     ],
            "GAZE_MAX_VARIANCE"    : [self.gaze_max_variance    ],
            "FACE_X_MAX_VARIANCE"  : [self.face_x_max_variance  ],
            "FACE_Y_MAX_VARIANCE"  : [self.face_y_max_variance  ],
            "FRAME_DIFF_THRESHOLD" : [self.frame_diff_threshold ],
            "FRAME_SKIP_RATE"      : [self.frame_skip_rate      ]
        }

        df = pd.DataFrame(params)

        if format   == "csv":
            df.to_csv(f"{path}/parameters.csv")

        elif format == "excel":
            with pd.ExcelWriter(f"{path}/parameters.xlsx", mode="w") as writer:
                df.to_excel(writer, sheet_name="parameters", index=False)
        
        else:
            print("Format not supported!")

        return df

    def export_all_frames_from_video(self, video_path, output_folder, overwrite=True,
                                     export_all=False, plot=True,
                                     export_sus_only=True) -> None:
        """
        Exports frames from a video. The user can choose whether to export all frames, 
            or only the different-looking ones. Function, basically, applies export_frame
            function to multiple frames. 
            Also saves the values to the data.txt file for further analysis

        Args:
        ->  video_path: path to the video
        ->  output_folder: folder that the frames will be exported to
        ->  overwrite: whether remove the old folder and write new data or not, 
            default is True
        ->  export_all: whether to export all frames or only the different ones.
            might be handy for time and storage saving. The default is false, hence,
            the default will export only the different frames.
        """
        video = cv2.VideoCapture(video_path)

        video_name = video_path.split("/")[-1].split(".")[0]

        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Create a text file to save all frame data
        output_path = f"{output_folder}/{video_name}"

        if overwrite:
            if os.path.exists(output_path):
                rmtree(output_path)

            os.makedirs(output_path)
        
        elif not os.path.exists(output_path):
            os.makedirs(output_path)

        # Store parameters of the object in an excel file
        params = self.export_params(output_path, format="excel")

        # Initialize a dataframe to store the parameters of every frame of the video
        df = create_custom_df()

        # Number of all frames of the video
        total_number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a buffer frame for holding the last selected different image. 
        # The new different-looking image will be considered as the buffer frame.
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, buffer_frame = video.read()
        if success:

            if export_all:
                skip_rate = 1
            else:
                skip_rate = self.frame_skip_rate

            for i in range(1, total_number_of_frames, skip_rate):
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()

                if ret:

                    if image_diff(buffer_frame, frame) >= self.frame_diff_threshold or export_all:
                        frame_timestamp = index_to_timestamp(i, fps)

                        position = self.export_frame(frame, output_path, frame_timestamp)

                        # Append the values to the df dataframe
                        df.loc[len(df.index)] = [
                            i,                  # Index
                            frame_timestamp,    # Time
                            position[0],        # Gaze
                            position[1],        # Face_X
                            position[2],        # Face_Y
                            position[3],        # Sus
                            position[4]         # Case
                        ]

                        buffer_frame = frame

            df["Time"] = df["Time"].apply(lambda x: x.replace("_", ":"))
            
            if plot:
                plot_positions(params, df, output_path)

            if export_sus_only:
                df = df[df["Sus"] == 1]

            export_custom_df(df, output_path, "excel")

            print(f"Filtered and exported frames from the video {video_name}")
            
        else:
            print(f"{video_name} cannot be processed")

        # Close the video file
        video.release()

    def process_all_videos(self, input_path, output_path) -> None:
        """
        Applies the export_all_frames_from_video function to all videos in a folder.
            Uses default values of the export_all argument.
            Might be useful if there are videos of multiple exam taker in a folder.

        Args:
        ->  input_path:  path to the foolder that contains the videos.
        ->  output_path: path to a folder to export frames of the videos.
        """
        videos = os.listdir(input_path)
        for video in videos:
            video_path = f"{input_path}/{video}"
            self.export_all_frames_from_video(video_path=video_path, 
                                              output_folder=output_path,)

    def __del__(self) -> None:
        # Destructor method to release resources
        del self.face_mesh
        del self.mp_face_mesh
