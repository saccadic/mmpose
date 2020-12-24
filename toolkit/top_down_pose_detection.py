import os
from argparse import ArgumentParser

from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    vis_pose_result
)

import cv2

# -----------------------------------------------------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')

    parser.add_argument('--input-image-path', type=str, default='', help='Image path')
    parser.add_argument('--input-movie-path', type=str, default='', help='Movie path')
    parser.add_argument('--output-image', type=str, default='', help='Output img path.')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    pose_model = init_pose_model(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device
    )

    if len(args.input_image_path) > 0:
        img = cv2.imread(args.input_image_path)

        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            img,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.show:




if __name__ == '__main__':
    main()




