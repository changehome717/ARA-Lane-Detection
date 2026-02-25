# ara/models/losses/talat_loss.py
import torch


def angle_center_calculate(pred, target, delta_l):
    # valid mask: stop once hit invalid (-1e5)
    valid_mask = (target != -1e5).cumprod(dim=-1).bool()
    valid_dx_mask = valid_mask[..., 1:] & valid_mask[..., :-1]

    pred_left, pred_right = pred[..., :-1], pred[..., 1:]
    target_left, target_right = target[..., :-1], target[..., 1:]

    pred_dx = (pred_right - pred_left) * valid_dx_mask
    target_dx = (target_right - target_left) * valid_dx_mask

    delta_l_tensor = torch.full_like(pred_dx, delta_l)

    pred_theta = torch.atan2(pred_dx, delta_l_tensor) * valid_dx_mask
    target_theta = torch.atan2(target_dx, delta_l_tensor) * valid_dx_mask

    pred_midx = (pred_right + pred_left) * 0.5 * valid_dx_mask
    target_midx = (target_right + target_left) * 0.5 * valid_dx_mask

    cos_target_theta = torch.cos(target_theta)
    pred_center_x = (pred_midx - target_midx) * cos_target_theta * valid_dx_mask
    target_center_x = torch.zeros_like(target_midx)

    delta_l_sq = delta_l ** 2
    pred_length = torch.sqrt(pred_dx.square() + delta_l_sq) * valid_dx_mask
    target_length = torch.sqrt(target_dx.square() + delta_l_sq) * valid_dx_mask

    pred_center_y = (0.5 * pred_length + (pred_midx - target_midx) * torch.sin(target_theta)) * valid_dx_mask
    target_center_y = (0.5 * target_length) * valid_dx_mask

    return (
        pred_theta, pred_center_x, pred_center_y, pred_length,
        target_theta, target_center_x, target_center_y, target_length
    )


def line_iou(pred, target, img_w, width=30, aligned=True):
    """
    Angle-aware lane overlap measure (historical name kept for compatibility).

    Args:
        pred: (N, S) if aligned else (Np, S)
        target: (N, S) if aligned else (Nt, S)
        img_w: image width
        width: lane width (px)
        aligned: True for matched pairs, False for pairwise matching
    Returns:
        iou_assign, iou_loss (aligned only), angle_diff_avg
    """
    eps = 1e-6
    img_h = 320
    delta_l = img_h / 63

    pred_midx = (pred[..., 1:] + pred[..., :-1]) * 0.5
    target_midx = (target[..., 1:] + target[..., :-1]) * 0.5

    px1_line = pred_midx - width / 2
    px2_line = pred_midx + width / 2
    tx1_line = target_midx - width / 2
    tx2_line = target_midx + width / 2

    if aligned:
        invalid_mask_bottom = target[..., :-1]
        invalid_mask_up = target[..., 1:]
        ovr = torch.min(px2_line, tx2_line) - torch.max(px1_line, tx1_line)
        union = torch.max(px2_line, tx2_line) - torch.min(px1_line, tx1_line)

        (pred_theta, pred_center_x, pred_center_y, pred_length,
         target_theta, target_center_x, target_center_y, target_length) = angle_center_calculate(
            pred, target, delta_l
        )
    else:
        num_pred = pred.shape[0]
        num_target = target.shape[0]
        invalid_mask_bottom = target[..., :-1].repeat(num_pred, 1, 1)
        invalid_mask_up = target[..., 1:].repeat(num_pred, 1, 1)

        ovr = (torch.min(px2_line[:, None, :], tx2_line[None, ...]) -
               torch.max(px1_line[:, None, :], tx1_line[None, ...]))
        union = (torch.max(px2_line[:, None, :], tx2_line[None, ...]) -
                 torch.min(px1_line[:, None, :], tx1_line[None, ...]))

        (pred_theta, pred_center_x, pred_center_y, pred_length,
         target_theta, target_center_x, target_center_y, target_length) = angle_center_calculate(
            pred.unsqueeze(1).expand(-1, num_target, -1),
            target.unsqueeze(0).expand(num_pred, -1, -1),
            delta_l
        )

    invalid_masks = (
        (invalid_mask_bottom < 0) | (invalid_mask_bottom >= img_w) |
        (invalid_mask_up < 0) | (invalid_mask_up >= img_w)
    )
    num_valid = (~invalid_masks).sum(dim=-1).clamp(min=1)

    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.

    # x IoU on centers
    px1 = pred_center_x - width / 2
    px2 = pred_center_x + width / 2
    tx1 = target_center_x - width / 2
    tx2 = target_center_x + width / 2

    over_x = (torch.min(px2, tx2) - torch.max(px1, tx1))
    union_x = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=0)
    over_x[invalid_masks] = 0.

    # y IoU on segment lengths
    py1 = pred_center_y - pred_length / 2
    py2 = pred_center_y + pred_length / 2
    ty1 = target_center_y - target_length / 2
    ty2 = target_center_y + target_length / 2

    over_y = (torch.min(py2, ty2) - torch.max(py1, ty1))
    union_y = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=0)
    over_y[invalid_masks] = 0.
    union_y[invalid_masks] = 0.

    # area IoU
    over_area = (over_x * over_y).clamp(min=0)
    union_area = (pred_length * width + target_length * width - over_area + eps).clamp(min=0)
    over_area[invalid_masks] = 0.
    union_area[invalid_masks] = 0.

    anchor_angle_diff = torch.abs(pred_theta - target_theta)
    anchor_angle_diff[invalid_masks] = 0.

    # 2-stage condition
    angle_area_thresh = (anchor_angle_diff <= 0.01) & (over_area < union_area) & (over_area > 0)
    angle_line_thresh = ~angle_area_thresh
    angle_area_thresh[invalid_masks] = 0.
    angle_line_thresh[invalid_masks] = 0.

    over_assign = (ovr * angle_line_thresh).sum(dim=-1) + (over_area * angle_area_thresh).sum(dim=-1)
    union_assign = (union * angle_line_thresh).sum(dim=-1) + (union_area * angle_area_thresh).sum(dim=-1)
    iou_assign = over_assign / (union_assign + eps)

    iou_loss = (1 - iou_assign).mean() if aligned else None
    angle_diff_avg = (anchor_angle_diff.sum(dim=-1) / num_valid).mean()

    return iou_assign, iou_loss, angle_diff_avg


def liou_loss(pred, target, img_w, width=30):
    return line_iou(pred, target, img_w, width)


def talat_loss(pred, target, img_w, width=30, aligned=True):
    return line_iou(pred, target, img_w, width=width, aligned=aligned)