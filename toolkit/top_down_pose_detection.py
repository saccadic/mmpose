
# python ./toolkit/top_down_pose_detection.py ./configs/top_down/resnet/coco/res50_coco_256x192.py --input-movie-path ./movies/sample2.mp4 --output-movie-path ./movies/result3.mp4

import os
import pathlib
from tqdm import tqdm
from argparse import ArgumentParser
import cv2
from pycocotools.coco import COCO
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    inference_top_down_pose_model, 
    init_pose_model,
    vis_pose_result
)

def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results
    return det_results[cat_id]


# optional
return_heatmap = False

# e.g. use ('backbone', ) to return backbone feature
output_layer_names = None

modelList = {
   "res50_coco_256x192.py" : "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth",
}

# -----------------------------------------------------------------------

def main():
    print("Current directory : ",os.getcwd())

    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')

    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--show', action='store_true', default=False, help='whether to show img')

    parser.add_argument('--input-image-path', type=str, default='', help='Image path')
    parser.add_argument('--input-movie-path', type=str, default='', help='Movie path')
    parser.add_argument('--output-image-path', type=str, default='', help='Output img path.')
    parser.add_argument('--output-movie-path', type=str, default='', help='Output movie path.')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()


    if (args.input_image_path == "") and (args.input_movie_path == ""):
        print("Please input a image or movie file.")
        exit()
    
    if args.show:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    print(args.pose_config, modelList[str(pathlib.Path(args.pose_config).name)])

    det_model = init_detector(
        "./demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py", 
        "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth", 
        device=args.device
    )
    

    pose_model = init_pose_model(
        args.pose_config,
        modelList[str(pathlib.Path(args.pose_config).name)],
        device=args.device
    )
    dataset = pose_model.cfg.data['test']['type']

    print(os.getcwd())

    if len(args.input_image_path) > 0:
        print("load image : ",args.input_image_path)
        img = cv2.imread(args.input_image_path)

        if img is None:
            print("Image load faile.")
            exit()

        mmdet_results = inference_detector(det_model, img)

        person_bboxes = process_mmdet_results(mmdet_results)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_bboxes,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            return_heatmap=return_heatmap,
            outputs=output_layer_names
        )

        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            # show the results
            cv2.imshow("result",vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if len(args.output_image_path) > 0:
            cv2.imwrite(args.output_image_path, vis_img)
            print("Save image : ", args.output_image_path)
    elif len(args.input_movie_path) > 0:

        video = cv2.VideoCapture(args.input_movie_path)

        if video.isOpened() == False:
            print("Movie load faile.")
            exit()

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))        # 動画の画面横幅
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 動画の画面縦幅
        size = (width, height)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))           # フレームレート

        if len(args.output_movie_path) > 0:
            fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
            savePath = args.output_movie_path
            writer = cv2.VideoWriter(savePath, fmt, frame_rate, size)  # ライター作成

        for i in tqdm(range(frame_count)):
            ret, frame = video.read()
            # img = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            if ret:

                mmdet_results = inference_detector(det_model, frame)

                person_bboxes = process_mmdet_results(mmdet_results)

                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    frame,
                    person_bboxes,
                    bbox_thr=args.bbox_thr,
                    format='xyxy',
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names
                )

                vis_img = vis_pose_result(
                    pose_model,
                    frame,
                    pose_results,
                    dataset=dataset,
                    kpt_score_thr=args.kpt_thr,
                    show=False
                )

                if args.show:
                    # show the results
                    cv2.imshow("result", vis_img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()


                if len(args.output_movie_path) > 0:
                    writer.write(vis_img)

                if args.show:
                    # show the results
                    cv2.imshow("result", vis_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break;

        cv2.destroyAllWindows()

        if len(args.output_movie_path) > 0:
            writer.release()


if __name__ == '__main__':
    main()




