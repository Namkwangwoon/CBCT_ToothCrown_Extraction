import numpy as np
from torch.nn import functional as F
import torch
from torch import nn


from torch.autograd import Variable
import math


# from model.utils.bbox_tools import generate_anchor_base
# from model.utils.creator_tool import ProposalCreator


# class RegionProposalNetwork(nn.Module):
#     """Region Proposal Network introduced in Faster R-CNN.

#     This is Region Proposal Network introduced in Faster R-CNN [#]_.
#     This takes features extracted from images and propose
#     class agnostic bounding boxes around "objects".

#     .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
#     Faster R-CNN: Towards Real-Time Object Detection with \
#     Region Proposal Networks. NIPS 2015.

#     Args:
#         in_channels (int): The channel size of input.
#         mid_channels (int): The channel size of the intermediate tensor.
#         ratios (list of floats): This is ratios of width to height of
#             the anchors.
#         anchor_scales (list of numbers): This is areas of anchors.
#             Those areas will be the product of the square of an element in
#             :obj:`anchor_scales` and the original area of the reference
#             window.
#         feat_stride (int): Stride size after extracting features from an
#             image.
#         initialW (callable): Initial weight value. If :obj:`None` then this
#             function uses Gaussian distribution scaled by 0.1 to
#             initialize weight.
#             May also be a callable that takes an array and edits its values.
#         proposal_creator_params (dict): Key valued paramters for
#             :class:`model.utils.creator_tools.ProposalCreator`.

#     .. seealso::
#         :class:`~model.utils.creator_tools.ProposalCreator`

#     """

#     def __init__(
#             self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
#             anchor_scales=[8, 16, 32], feat_stride=16,
#             proposal_creator_params=dict(),
#     ):
#         super(RegionProposalNetwork, self).__init__()
#         self.anchor_base = generate_anchor_base(
#             anchor_scales=anchor_scales, ratios=ratios)
#         self.feat_stride = feat_stride
#         self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
#         n_anchor = self.anchor_base.shape[0]
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
#         self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
#         self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
#         normal_init(self.conv1, 0, 0.01)
#         normal_init(self.score, 0, 0.01)
#         normal_init(self.loc, 0, 0.01)

#     def forward(self, x, img_size, scale=1.):
#         """Forward Region Proposal Network.

#         Here are notations.

#         * :math:`N` is batch size.
#         * :math:`C` channel size of the input.
#         * :math:`H` and :math:`W` are height and witdh of the input feature.
#         * :math:`A` is number of anchors assigned to each pixel.

#         Args:
#             x (~torch.autograd.Variable): The Features extracted from images.
#                 Its shape is :math:`(N, C, H, W)`.
#             img_size (tuple of ints): A tuple :obj:`height, width`,
#                 which contains image size after scaling.
#             scale (float): The amount of scaling done to the input images after
#                 reading them from files.

#         Returns:
#             (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

#             This is a tuple of five following values.

#             * **rpn_locs**: Predicted bounding box offsets and scales for \
#                 anchors. Its shape is :math:`(N, H W A, 4)`.
#             * **rpn_scores**:  Predicted foreground scores for \
#                 anchors. Its shape is :math:`(N, H W A, 2)`.
#             * **rois**: A bounding box array containing coordinates of \
#                 proposal boxes.  This is a concatenation of bounding box \
#                 arrays from multiple images in the batch. \
#                 Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
#                 bounding boxes from the :math:`i` th image, \
#                 :math:`R' = \\sum _{i=1} ^ N R_i`.
#             * **roi_indices**: An array containing indices of images to \
#                 which RoIs correspond to. Its shape is :math:`(R',)`.
#             * **anchor**: Coordinates of enumerated shifted anchors. \
#                 Its shape is :math:`(H W A, 4)`.

#         """
#         n, _, hh, ww = x.shape
#         anchor = _enumerate_shifted_anchor(
#             np.array(self.anchor_base),
#             self.feat_stride, hh, ww)

#         n_anchor = anchor.shape[0] // (hh * ww)
#         h = F.relu(self.conv1(x))

#         rpn_locs = self.loc(h)
#         # UNNOTE: check whether need contiguous
#         # A: Yes
#         rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
#         rpn_scores = self.score(h)
#         rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
#         rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
#         rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
#         rpn_fg_scores = rpn_fg_scores.view(n, -1)
#         rpn_scores = rpn_scores.view(n, -1, 2)

#         rois = list()
#         roi_indices = list()
#         for i in range(n):
#             roi = self.proposal_layer(
#                 rpn_locs[i].cpu().data.numpy(),
#                 rpn_fg_scores[i].cpu().data.numpy(),
#                 anchor, img_size,
#                 scale=scale)
#             batch_index = i * np.ones((len(roi),), dtype=np.int32)
#             rois.append(roi)
#             roi_indices.append(batch_index)

#         rois = np.concatenate(rois, axis=0)
#         roi_indices = np.concatenate(roi_indices, axis=0)
#         return rpn_locs, rpn_scores, rois, roi_indices, anchor


# def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
#     # Enumerate all shifted anchors:
#     #
#     # add A anchors (1, A, 4) to
#     # cell K shifts (K, 1, 4) to get
#     # shift anchors (K, A, 4)
#     # reshape to (K*A, 4) shifted anchors
#     # return (K*A, 4)

#     # !TODO: add support for torch.CudaTensor
#     # xp = cuda.get_array_module(anchor_base)
#     # it seems that it can't be boosed using GPU
#     import numpy as xp
#     shift_y = xp.arange(0, height * feat_stride, feat_stride)
#     shift_x = xp.arange(0, width * feat_stride, feat_stride)
#     shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
#     shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
#                       shift_y.ravel(), shift_x.ravel()), axis=1)

#     A = anchor_base.shape[0]
#     K = shift.shape[0]
#     anchor = anchor_base.reshape((1, A, 4)) + \
#              shift.reshape((1, K, 4)).transpose((1, 0, 2))
#     anchor = anchor.reshape((K * A, 4)).astype(np.float32)
#     return anchor


# def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
#     # Enumerate all shifted anchors:
#     #
#     # add A anchors (1, A, 4) to
#     # cell K shifts (K, 1, 4) to get
#     # shift anchors (K, A, 4)
#     # reshape to (K*A, 4) shifted anchors
#     # return (K*A, 4)

#     # !TODO: add support for torch.CudaTensor
#     # xp = cuda.get_array_module(anchor_base)
#     import torch as t
#     shift_y = t.arange(0, height * feat_stride, feat_stride)
#     shift_x = t.arange(0, width * feat_stride, feat_stride)
#     shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
#     shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
#                       shift_y.ravel(), shift_x.ravel()), axis=1)

#     A = anchor_base.shape[0]
#     K = shift.shape[0]
#     anchor = anchor_base.reshape((1, A, 4)) + \
#              shift.reshape((1, K, 4)).transpose((1, 0, 2))
#     anchor = anchor.reshape((K * A, 4)).astype(np.float32)
#     return anchor


# def normal_init(m, mean, stddev, truncated=False):
#     """
#     weight initalizer: truncated normal and random normal.
#     """
#     # x is a parameter
#     if truncated:
#         m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
#     else:
#         m.weight.data.normal_(mean, stddev)
#         m.bias.data.zero_()



BACKBONE_STRIDES = [4, 8, 16, 32, 64]
RPN_ANCHOR_STRIDE = 1
RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
GPU_COUNT = 1
IMAGE_SHAPE = (128, 128, 128, 1)



class RPN(nn.Module):
    """Builds the model of Region Proposal Network.
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_logits: [batch, D, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, D, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, D, H, W, (dz, dy, dx, log(dd), log(dh), log(dw))] Deltas to be applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, channel, conv_channel):
        super(RPN, self).__init__()
        self.conv_shared = nn.Conv3d(channel, conv_channel, kernel_size=3, stride=anchor_stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv3d(conv_channel, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv3d(conv_channel, 6 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(x))

        # Anchor Score. [batch, anchors per location * 2, depth, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 4, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, anchors per location * 6, D, H, W]
        # where 6 == delta [z, y, x, log(d), log(h), log(w)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, anchors, 6]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 4, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 6)
        
        
        anchors = Variable(torch.from_numpy(generate_pyramid_anchors((32, 64, 128, 256, 512),
                                                                     [1],
                                                                     compute_backbone_shapes(
                                                                         None, IMAGE_SHAPE),
                                                                     BACKBONE_STRIDES,
                                                                     RPN_ANCHOR_STRIDE)).float(),
                   requires_grad=False)
        
        
        rpn_rois = []
        roi_indices = []
        
        for i, (probs, bbox) in enumerate(zip(rpn_probs, rpn_bbox)):
            roi = proposal_layer([probs, bbox],
                            proposal_count=2000,
                            nms_threshold=0.7,
                            anchors=anchors,
                            config=None)
            rpn_rois.append(roi)
            roi_indices.append(torch.Tensor([i]*roi.shape[1]).cuda())
        
        
        return torch.cat(rpn_rois, dim=1)[0], torch.cat(roi_indices)
        
        
        




def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 6] where each row is z1, y1, x1, z2, y2, x2
    deltas: [N, 6] where each row is [dz, dy, dx, log(dd), log(dh), log(dw)]
    """
    # Convert to z, y, x, d, h, w
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= torch.exp(deltas[:, 3])
    height *= torch.exp(deltas[:, 4])
    width *= torch.exp(deltas[:, 5])
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([z1, y1, x1, z2, y2, x2], dim=1)
    return result



def compute_iou(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [z1, y1, x1, z2, y2, x2]
    boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
    box_volume: float. the volume of 'box'
    boxes_volume: array of depth boxes_count.

    Note: the volumes are passed in rather than calculated here for
          efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection volumes
    z1 = np.maximum(box[0], boxes[:, 0])
    z2 = np.minimum(box[3], boxes[:, 3])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / (union + 1e-6)
    return iou




def non_max_suppression(boxes, scores, threshold, max_num):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)]. Notice that (z2, y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    max_num: Int. The max number of boxes to keep.
    Return the index of boxes to keep.
    """
    # Compute box volumes
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    volume = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # Get indices of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        if len(pick) >= max_num:
            break
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], volume[i], volume[ixs[1:]])
        # Identify boxes with IoU over the threshold. This returns indices into ixs[1:],
        # so add 1 to get indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)



def clip_boxes(boxes, window):
    """boxes: [N, 6] each col is z1, y1, x1, z2, y2, x2
    window: [6] in the form z1, y1, x1, z2, y2, x2
    """
    boxes = torch.stack(
        [boxes[:, 0].clamp(float(window[0]), float(window[3])),
         boxes[:, 1].clamp(float(window[1]), float(window[4])),
         boxes[:, 2].clamp(float(window[2]), float(window[5])),
         boxes[:, 3].clamp(float(window[0]), float(window[3])),
         boxes[:, 4].clamp(float(window[1]), float(window[4])),
         boxes[:, 5].clamp(float(window[2]), float(window[5]))], 1)
    return boxes



def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    Returns:
        Proposals in normalized coordinates [batch, rois, (z1, y1, x1, z2, y2, x2)]
    """

    # Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 6]
    deltas = inputs[1]
    std_dev = torch.from_numpy(np.reshape(RPN_BBOX_STD_DEV, [1, 6])).float()
    if GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = anchors.size()[0]
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.detach(), :]
    anchors = anchors[order.detach().cpu(), :].cuda()

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (z1, y1, x1, z2, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (z1, y1, x1, z2, y2, x2)]
    height, width, depth = IMAGE_SHAPE[:3]
    window = np.array([0, 0, 0, depth, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Non-max suppression
    keep = non_max_suppression(boxes.cpu().detach().numpy(),
                                     scores.cpu().detach().numpy(), nms_threshold, proposal_count)
    keep = torch.from_numpy(keep).long()
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes



def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [1]
    shape: [depth, height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    # TODO: conditions when we have different ratios?
    # Here I apply a trick.
    depths = scales
    heights = scales
    widths = scales

    # Enumerate shifts in feature space
    shifts_z = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_y = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[2], anchor_stride) * feature_stride
    shifts_z, shifts_y, shifts_x = np.meshgrid(shifts_z, shifts_y, shifts_x)

    # Enumerate combinations of shifts, widths, and heights
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (z, y, x) and a list of (d, h, w)
    box_centers = np.stack(
        [box_centers_z, box_centers_y, box_centers_x], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_depths, box_heights, box_widths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (z1, y1, x1, z2, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes



def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (z1, y1, x1, z2, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (z1, y1, x1, z2, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)



def compute_backbone_shapes(config, image_shape):
    """Computes the depth, width and height of each stage of the backbone network.
    Returns:
        [N, (depth, height, width)]. Where N is the number of stages
    """
    H, W, D = image_shape[:3]
    return np.array(
        [[int(math.ceil(D / stride)),
          int(math.ceil(H / stride)),
          int(math.ceil(W / stride))]
         for stride in BACKBONE_STRIDES])