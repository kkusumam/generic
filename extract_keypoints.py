import mmcv
from mmpose.apis import inference_pose_model, init_pose_model
import pickle
import os
from tqdm import tqdm

# Set the input and output paths
input_path = "path/to/input/videos"
output_path = "path/to/output/pickle"

# Create an MMPOSE model with the required parameters
config_file = "path/to/config/file"
checkpoint_file = "path/to/checkpoint/file"
device = "cuda:0" # set to "cpu" if you don't have a GPU
model = init_pose_model(config_file, checkpoint_file, device=device)

# Define a function to extract key points from a video using the MMPOSE model and give a frame id to each frame
def extract_keypoints(video_path):
    cap = mmcv.VideoReader(video_path)
    frame_count = len(cap)
    keypoints_list = []

    for i in tqdm(range(frame_count)):
        frame = cap[i]
        result = inference_pose_model(model, frame)
        keypoints_list.append({
            "frame_id": i,
            "keypoints": result['pred_joints']
        })

    cap.close()
    return keypoints_list

# Iterate over all the videos in the input path and extract key points for each video using the above function
keypoints_dict = {}
for video_file in os.listdir(input_path):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_path, video_file)
        keypoints_list = extract_keypoints(video_path)
        keypoints_dict[video_file] = keypoints_list

# Save the extracted key points in a pickle file
with open(output_path, 'wb') as f:
    pickle.dump(keypoints_dict, f)

