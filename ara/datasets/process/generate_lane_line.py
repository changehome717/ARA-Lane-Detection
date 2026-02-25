import math
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline

from .transforms import AraTransforms
from ..registry import PROCESS


@PROCESS.register_module
class GenerateLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True, logger=None):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.training = training

        # lightweight logger (optional)
        self.logger = logger

        if transforms is None:
            transforms = AraTransforms(self.img_h, self.img_w)

        img_transforms = []
        if transforms is not None:
            for aug in transforms:
                p = aug["p"]
                if aug["name"] != "OneOf":
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=getattr(iaa, aug["name"])(**aug["parameters"]),
                        )
                    )
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf(
                                [
                                    getattr(iaa, aug_["name"])(**aug_["parameters"])
                                    for aug_ in aug["transforms"]
                                ]
                            ),
                        )
                    )

        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        return [LineString(lane) for lane in lanes]

    def sample_lane(self, points, sample_ys):
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception("Annotation points must be sorted")
        x, y = points[:, 0], points[:, 1]

        # --- handle descending sample_ys safely ---
        sample_ys_asc = np.sort(sample_ys)
        domain_min_y, domain_max_y = y.min(), y.max()

        insert_pos_min = np.searchsorted(sample_ys_asc, domain_min_y, side="right")
        insert_pos_max = np.searchsorted(sample_ys_asc, domain_max_y, side="left")

        start_idx = len(sample_ys_asc) - insert_pos_min
        end_idx = len(sample_ys_asc) - insert_pos_max - 1

        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(sample_ys) - 1)

        sample_ys_inside_domain = sample_ys[end_idx : start_idx + 1]

        interp = InterpolatedUnivariateSpline(
            y[::-1], x[::-1], k=min(3, len(points) - 1)
        )
        interp_xs = interp(sample_ys_inside_domain)

        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > sample_ys[end_idx]]
        extrap_xs = np.polyval(extrap, extrap_ys)

        all_xs = np.hstack((extrap_xs, interp_xs))
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        return all_xs[~inside_mask], all_xs[inside_mask]

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])
        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h
        old_lanes = anno["lanes"]

        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        old_lanes = [
            [[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
            for lane in old_lanes
        ]

        lanes = np.ones(
            (self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5
        lanes_endpoints = np.ones((self.max_lanes, 2))

        lanes[:, 0] = 1
        lanes[:, 1] = 0

        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            if len(xs_inside_image) <= 1:
                continue

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)
                ) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas)
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs

            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        return {"label": lanes, "old_anno": anno, "lane_endpoints": lanes_endpoints}

    def linestrings_to_lanes(self, lines):
        return [line.coords for line in lines]

    def __call__(self, sample):
        img_org = sample["img"]
        line_strings_org = self.lane_to_linestrings(sample["lanes"])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)

        for i in range(30):
            if self.training:
                mask_org = SegmentationMapsOnImage(sample["mask"], shape=img_org.shape)
                img, line_strings, seg = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org,
                )
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                )
                seg = None

            line_strings.clip_out_of_image_()
            new_anno = {"lanes": self.linestrings_to_lanes(line_strings)}

            try:
                annos = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))
                label = annos["label"]
                lane_endpoints = annos["lane_endpoints"]
                break
            except Exception as e:
                if (i + 1) == 30:
                    if self.logger is not None:
                        self.logger.critical(f"Transform annotation failed 30 times: {e}")
                    raise

        sample["img"] = img.astype(np.float32) / 255.0
        sample["lane_line"] = label
        sample["lanes_endpoints"] = lane_endpoints
        sample["gt_points"] = new_anno["lanes"]
        sample["seg"] = seg.get_arr() if (self.training and seg is not None) else np.zeros(img_org.shape)
        return sample