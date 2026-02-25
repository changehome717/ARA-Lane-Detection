import os
import argparse
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon

import json
from collections import defaultdict, Counter


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = np.array(lane, dtype=np.int32)
    cv2.polylines(img, [lane],
                  isClosed=False,
                  color=(255, 255, 255),
                  thickness=width)

    return img


def resize_lane(points, size=(1640, 590)):
    old_h, old_w = size
    new_h, new_w = 590, 1640

    fx = new_w / old_w
    fy = new_h / old_h

    new_points = []
    for x, y in points:
        new_x = x * fx
        new_y = y * fy
        new_points.append((new_x, new_y))

    return new_points


def make_img(points, shape=(590, 1640, 3)):
    img = np.zeros(shape, dtype=np.uint8)
    for lane in points:
        img = draw_lane(lane, img=img, img_shape=shape)

    return img


def culane_lane_to_linestring(lane, img_shape=(590, 1640, 3)):
    lane = resize_lane(lane, size=(img_shape[0], img_shape[1]))
    lane = np.array(lane, dtype=np.int32)
    line = LineString(lane)
    return line


def culane_lane_to_polygon(lane, img_shape=(590, 1640, 3), width=30):
    lane = resize_lane(lane, size=(img_shape[0], img_shape[1]))
    lane = np.array(lane, dtype=np.int32)
    line = LineString(lane)
    poly = line.buffer(width / 2., cap_style=2, join_style=2)
    return poly


def fourier_iou(xs, ys, img_shape=(590, 1640, 3), width=30):
    image = Polygon([
        (0, 0),
        (0, img_shape[0] - 1),
        (img_shape[1] - 1, img_shape[0] - 1),
        (img_shape[1] - 1, 0)
    ])

    xs = [
        culane_lane_to_polygon(lane, img_shape=img_shape, width=width)
        for lane in xs
    ]
    ys = [
        culane_lane_to_polygon(lane, img_shape=img_shape, width=width)
        for lane in ys
    ]

    xs = [lane.intersection(image) for lane in xs]
    ys = [
        lane.buffer(1.5, cap_style=2, join_style=2).intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(590, 1640, 3)):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                           dtype=object)  # (10, 50, 2)

    if official:
        ious = fourier_iou(interp_pred,
                           interp_anno,
                           img_shape=img_shape,
                           width=width)
    else:
        pred_img = make_img(interp_pred, img_shape)
        anno_img = make_img(interp_anno, img_shape)

        pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        anno_gray = cv2.cvtColor(anno_img, cv2.COLOR_BGR2GRAY)

        iou = cal_iou(pred_gray, anno_gray)
        ious = np.zeros((len(pred), len(anno)))
        for i in range(len(pred)):
            for j in range(len(anno)):
                ious[i, j] = iou

    if len(pred) != 0 and len(anno) != 0:
        row_ind, col_ind = linear_sum_assignment(-ious)

        for thr in iou_thresholds:
            for i, j in zip(row_ind, col_ind):
                if ious[i, j] >= thr:
                    _metric[thr][0] += 1
                else:
                    _metric[thr][1] += 1
                    _metric[thr][2] += 1

            _metric[thr][1] += len(pred) - len(row_ind)
            _metric[thr][2] += len(anno) - len(col_ind)

    return _metric


def cal_iou(pred, anno):
    overlap = np.logical_and(pred, anno)
    union = np.logical_or(pred, anno)
    if union.sum() == 0:
        return 0
    else:
        return overlap.sum() / union.sum()


def load_culane_img_data(data_dir):
    with open(data_dir, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_culane_data(data_dir, file_list_path):
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            os.path.join(
                data_dir, line[1 if line[0] == '/' else 0:].rstrip().replace(
                    '.jpg', '.lines.txt')) for line in file_list.readlines()
        ]

    data = []
    for path in filepaths:
        img_data = load_culane_img_data(path)
        data.append(img_data)

    return data


def load_image_bins(json_path):
    """Load image -> curvature-bin mapping from curve_lane_bins.json.

    The JSON has lane-level keys "img_rel#lane_idx" -> "low"/"medium"/"high".
    Here we aggregate to image-level via majority vote; ties are broken in favor
    of higher curvature (high > medium > low).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    lane_bins = data['lane_bins']  # "img_rel#idx" -> bin_name

    image2bins = defaultdict(list)
    for key, bin_name in lane_bins.items():
        img_rel, lane_idx_str = key.rsplit('#', 1)
        image2bins[img_rel].append(bin_name)

    rank = {'low': 0, 'medium': 1, 'high': 2}
    image_bin = {}
    for img_rel, bins in image2bins.items():
        cnt = Counter(bins)
        best_bin, _ = max(cnt.items(), key=lambda kv: (kv[1], rank[kv[0]]))
        image_bin[img_rel] = best_bin

    k1 = data.get('k1', None)
    k2 = data.get('k2', None)
    return image_bin, k1, k2


def eval_predictions(pred_dir,
                     anno_dir,
                     list_path,
                     iou_thresholds=[0.5],
                     width=30,
                     official=True,
                     sequential=False,
                     curve_bins_json=None):
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Calculating metric for List: {}'.format(list_path))

    # Read relative image paths (consistent with load_culane_data)
    img_rel_list = []
    with open(list_path, 'r') as file_list:
        for line in file_list:
            line = line.strip()
            if not line:
                continue
            img_rel = line[1 if line[0] == '/' else 0:]
            img_rel_list.append(img_rel)

    # Load predictions and annotations
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    img_shape = (590, 1640, 3)

    # Optional: load image -> curvature-bin mapping
    image_bin_map = None
    if curve_bins_json is not None:
        image_bin_map, k1, k2 = load_image_bins(curve_bins_json)
        logger.info(
            f'Loaded curvature bins from {curve_bins_json}, k1={k1}, k2={k2}'
        )

    # Run CULane metric per image (sequentially or in parallel)
    if sequential:
        results = map(
            partial(culane_metric,
                    width=width,
                    official=official,
                    iou_thresholds=iou_thresholds,
                    img_shape=img_shape), predictions, annotations)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(
                culane_metric,
                zip(predictions, annotations,
                    repeat(width),
                    repeat(iou_thresholds),
                    repeat(official),
                    repeat(img_shape)))

    results = list(results)

    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
                    'precision: {}, recall: {}, f1: {}'.format(
                        thr, tp, fp, fn, precision, recall, f1))
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

        # If curvature bins are provided, accumulate TP/FP/FN per bin
        if image_bin_map is not None:
            bin_stats = {
                'low':   {'TP': 0, 'FP': 0, 'FN': 0},
                'medium':{'TP': 0, 'FP': 0, 'FN': 0},
                'high':  {'TP': 0, 'FP': 0, 'FN': 0},
            }
            for img_rel, m in zip(img_rel_list, results):
                img_bin = image_bin_map.get(img_rel, None)
                if img_bin is None:
                    continue
                tp_i, fp_i, fn_i = m[thr]
                bin_stats[img_bin]['TP'] += tp_i
                bin_stats[img_bin]['FP'] += fp_i
                bin_stats[img_bin]['FN'] += fn_i

            for b in ['low', 'medium', 'high']:
                TPb = bin_stats[b]['TP']
                FPb = bin_stats[b]['FP']
                FNb = bin_stats[b]['FN']
                if TPb == 0:
                    prec_b = rec_b = f1_b = 0.0
                else:
                    prec_b = TPb / (TPb + FPb) if (TPb + FPb) > 0 else 0.0
                    rec_b = TPb / (TPb + FNb) if (TPb + FNb) > 0 else 0.0
                    f1_b = (2 * prec_b * rec_b /
                            (prec_b + rec_b)) if (prec_b + rec_b) > 0 else 0.0
                logger.info(
                    f'  [curvature bin={b}] thr={thr:.2f}, '
                    f'TP={TPb}, FP={FPb}, FN={FNb}, '
                    f'Precision={prec_b:.4f}, Recall={rec_b:.4f}, F1={f1_b:.4f}'
                )
            # Optionally store per-bin stats as well
            ret[f'bins@{thr}'] = bin_stats

    if len(iou_thresholds) > 2:
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret


def main():
    args = parse_args()
    for list_path in args.list:
        results = eval_predictions(args.pred_dir,
                                   args.anno_dir,
                                   list_path,
                                   width=args.width,
                                   official=args.official,
                                   sequential=args.sequential,
                                   curve_bins_json=args.curve_bins_json)

        header = '=' * 20 + ' Results ({})'.format(
            os.path.basename(list_path)) + '=' * 20
        print(header)
        for metric, value in results.items():
            if isinstance(value, float):
                print('{}: {:.4f}'.format(metric, value))
            else:
                print('{}: {}'.format(metric, value))
        print('=' * len(header))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument(
        "--pred_dir",
        help="Path to directory containing the predicted lanes",
        required=True)
    parser.add_argument(
        "--anno_dir",
        help="Path to directory containing the annotated lanes",
        required=True)
    parser.add_argument("--width",
                        type=int,
                        default=30,
                        help="Width of the lane")
    parser.add_argument("--list",
                        nargs='+',
                        help="Path to txt file containing the list of files",
                        required=True)
    parser.add_argument("--sequential",
                        action='store_true',
                        help="Run sequentially instead of in parallel")
    parser.add_argument("--official",
                        action='store_true',
                        help="Use official way to calculate the metric")
    parser.add_argument("--curve_bins_json",
                        type=str,
                        default=None,
                        help="Path to curve_lane_bins.json for curvature-wise evaluation")

    return parser.parse_args()


if __name__ == '__main__':
    main()
