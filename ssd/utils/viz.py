import numpy as np
from six.moves import range
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

try:
    FONT = ImageFont.truetype('arial.ttf', 24)
except IOError:
    FONT = ImageFont.load_default()


def _draw_single_box(image, xmin, ymin, xmax, ymax, color='black', display_str=None, font=None, thickness=2):
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = xmin, xmax, ymin, ymax
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    if display_str is not None:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)

    return image


def draw_bounding_boxes(image, boxes, labels=None, probs=None, class_name_map=None):
    """Draw bboxes(labels, probs) on image
    Args:
        image: numpy array image, shape should be (height, width, channel)
        boxes: bboxes, shape should be (N, 4), and each row is (xmin, ymin, xmax, ymax)
        labels: labels, shape: (N, )
        probs: label scores, shape: (N, ), can be False/True or None
        class_name_map: list or dict, map class id to class name for visualization.
             can be False/True or None
    Returns:
        An image with information drawn on it.
    """
    num_boxes = boxes.shape[0]
    gt_boxes_new = boxes.copy()
    draw_image = Image.fromarray(np.uint8(image))
    for i in range(num_boxes):
        display_str = None
        this_class = 0
        if labels is not None:
            this_class = labels[i]
            class_name = class_name_map[this_class] if class_name_map is not None else str(this_class)
            class_name = class_name.decode('utf-8') if isinstance(class_name, bytes) else class_name
            if probs is not None:
                prob = probs[i]
                display_str = '{}:{:.2f}'.format(class_name, prob)
            else:
                display_str = class_name
        draw_image = _draw_single_box(image=draw_image,
                                      xmin=gt_boxes_new[i, 0],
                                      ymin=gt_boxes_new[i, 1],
                                      xmax=gt_boxes_new[i, 2],
                                      ymax=gt_boxes_new[i, 3],
                                      color=STANDARD_COLORS[this_class % NUM_COLORS],
                                      display_str=display_str,
                                      font=FONT)

    image = np.array(draw_image, dtype=np.float32)
    return image
