#!/usr/bin/env python3
"""
Framework for image compression comparison.

A new codec can be easily added to the framework.
Add the definition to TUPLE_CODECS and implement corresponding encoding, decoding,
and metric calculation steps in method f(). Please use existing codecs as examples.

Our study showing that VMAF has very high correlation with human scores can be found in
IEEE Transactions on Image Processing with article title
"Quality Measurement of Images on Mobile Streaming Interfaces Deployed at Scale".

Our methodology here is to find encoder parameters for a given source image
in order to achieve a given target VMAF quality. Having such encodes
using different codecs that achieve the same VMAF score simplifies comparison
of compression efficiency.
"""
import errno
import os
import sys
import subprocess
import json
import glob
from collections import Counter
import logging
from collections import namedtuple
import datetime
import uuid
import multiprocessing
import ntpath
import threading
import sqlite3

__author__ = "Aditya Mavlankar"
__copyright__ = "Copyright 2019-2020, Netflix, Inc."
__credits__ = ["Kyle Swanson", "Jan de Cock", "Marjan Parsa"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__maintainer__ = "Aditya Mavlankar"
__email__ = "amavlankar@netflix.com"
__status__ = "Development"


TOTAL_BYTES = Counter()
TOTAL_METRIC = Counter()
TOTAL_ERRORS = Counter()

CodecType = namedtuple('CodecType', ['name', 'inverse', 'param_start', 'param_end', 'ab_tol', 'subsampling'])
TUPLE_CODECS = (
    CodecType('jpeg', False, 5, 100, 1, '420'),
    CodecType('jpeg', False, 5, 100, 1, '444'),
    CodecType('jpeg', False, 5, 100, 1, '444u'),

    CodecType('jpeg-mse', False, 5, 100, 1, '420'),
    CodecType('jpeg-mse', False, 5, 100, 1, '444'),
    CodecType('jpeg-mse', False, 5, 100, 1, '444u'),

    CodecType('jpeg-ms-ssim', False, 5, 100, 1, '420'),
    CodecType('jpeg-ms-ssim', False, 5, 100, 1, '444'),
    CodecType('jpeg-ms-ssim', False, 5, 100, 1, '444u'),

    CodecType('jpeg-im', False, 5, 100, 1, '420'),
    CodecType('jpeg-im', False, 5, 100, 1, '444'),
    CodecType('jpeg-im', False, 5, 100, 1, '444u'),

    CodecType('jpeg-hvs-psnr', False, 5, 100, 1, '420'),
    CodecType('jpeg-hvs-psnr', False, 5, 100, 1, '444'),
    CodecType('jpeg-hvs-psnr', False, 5, 100, 1, '444u'),

    # webp can only encode 420
    CodecType('webp', False, 5, 100, 1, '420'),
    CodecType('webp', False, 5, 100, 1, '444u'),

    CodecType('kakadu-mse', False, 0.01, 3.0, 0.03, '420'),
    CodecType('kakadu-mse', False, 0.01, 3.0, 0.03, '444'),
    CodecType('kakadu-mse', False, 0.01, 3.0, 0.03, '444u'),

    CodecType('kakadu-visual', False, 0.01, 3.0, 0.03, '420'),
    CodecType('kakadu-visual', False, 0.01, 3.0, 0.03, '444'),
    CodecType('kakadu-visual', False, 0.01, 3.0, 0.03, '444u'),

    CodecType('openjpeg', False, 30.0, 60.0, 0.05, '420'),
    CodecType('openjpeg', False, 30.0, 60.0, 0.05, '444'),
    CodecType('openjpeg', False, 30.0, 60.0, 0.05, '444u'),

    CodecType('hevc', True, 10, 51, 1, '420'),
    CodecType('hevc', True, 10, 51, 1, '444'),
    CodecType('hevc', True, 10, 51, 1, '444u'),

    # The codecs 'avif-mse' and 'avif-ssim' write out a "single-frame video" file in IVF format,
    # so technically not a proper AVIF. But they accomplish "full-range" YUV encoding similar to a "proper" AVIF.
    CodecType('avif-mse', True, 8, 63, 1, '420'),
    CodecType('avif-mse', True, 8, 63, 1, '444'),
    CodecType('avif-mse', True, 8, 63, 1, '444u'),

    CodecType('avif-ssim', True, 8, 63, 1, '420'),
    CodecType('avif-ssim', True, 8, 63, 1, '444'),
    CodecType('avif-ssim', True, 8, 63, 1, '444u'),

    # Codecs starting with 'avifenc' prefix write out a proper AVIF; avifenc is a libavif-based tool.
    # https://github.com/AOMediaCodec/libavif
    CodecType('avifenc-sp-0', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-0', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-0', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-2', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-2', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-2', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-4', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-4', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-4', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-6', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-6', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-6', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-8', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-8', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-8', True, 8, 62, 1, '444u'),

    # CodecType('avifenc_ssim-sp-0', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-0', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-0', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-2', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-2', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-2', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-4', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-4', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-4', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-6', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-6', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-6', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-8', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-8', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-8', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-0-crf', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-0-crf', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-0-crf', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-2-crf', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-2-crf', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-2-crf', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-4-crf', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-4-crf', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-4-crf', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-6-crf', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-6-crf', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-6-crf', True, 8, 62, 1, '444u'),

    CodecType('avifenc-sp-8-crf', True, 8, 62, 1, '420'),
    CodecType('avifenc-sp-8-crf', True, 8, 62, 1, '444'),
    CodecType('avifenc-sp-8-crf', True, 8, 62, 1, '444u'),

    # CodecType('avifenc_ssim-sp-0-crf', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-0-crf', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-0-crf', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-2-crf', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-2-crf', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-2-crf', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-4-crf', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-4-crf', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-4-crf', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-6-crf', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-6-crf', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-6-crf', True, 8, 62, 1, '444u'),
    #
    # CodecType('avifenc_ssim-sp-8-crf', True, 8, 62, 1, '420'),
    # CodecType('avifenc_ssim-sp-8-crf', True, 8, 62, 1, '444'),
    # CodecType('avifenc_ssim-sp-8-crf', True, 8, 62, 1, '444u'),

)

LOGGER = logging.getLogger('image.compression')
CONNECTION = None


def setup_logging(worker, worker_id):
    """
    set up logging for the process calling this function.
    :param worker: True means it is a worker process from the pool. False means it is the main process.
    :param worker_id: unique ID identifying the process
    :return:
    """
    for h in list(LOGGER.handlers):
        LOGGER.removeHandler(h)
    now = datetime.datetime.now()
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    name = "compression_results_"
    if worker:
        name += "worker_"
        global CONNECTION
        CONNECTION = None
    name += str(worker_id) + '_' + now.isoformat() + '.txt'
    file_log_handler = logging.FileHandler(name)
    file_log_handler.setFormatter(formatter)
    LOGGER.addHandler(file_log_handler)

    console_log_handler = logging.StreamHandler()
    console_log_handler.setFormatter(formatter)
    LOGGER.addHandler(console_log_handler)

    LOGGER.setLevel('DEBUG')


def mkdir_p(path):
    """ mkdir -p
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def listdir_full_path(directory):
    """ like os.listdir(), but returns full absolute paths
    """
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            yield os.path.abspath(os.path.join(directory, f))


def decode(value):
    """ Convert bytes to string, if needed
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def run_program_simple(*args, **kwargs):
    """ simple way to run a command ignoring stderr and stdout
    """
    output = os.system(" ".join(*args) + " >/dev/null 2>&1", **kwargs)
    if output != 0:
        raise RuntimeError("failed to execute " + " ".join(*args))


def run_program(*args, **kwargs):
    """ run command using subprocess.check_output, collect stdout and stderr together and return
    """
    kwargs.setdefault("stderr", subprocess.STDOUT)
    kwargs.setdefault("shell", True)

    try:
        output = subprocess.check_output(" ".join(*args), **kwargs)
        return decode(output)

    except subprocess.CalledProcessError as e:
        LOGGER.error("***** ATTENTION : subprocess call crashed: %s\n%s", args, e.output)
        raise


def get_dimensions(image):
    """ given a source image, return dimensions and bit-depth
    """
    dimension_cmd = ["identify", '-format', '%w,%h,%z', image]
    res = run_program(dimension_cmd)
    res_split = res.split('\n')
    width, height, depth = res_split[len(res_split) - 1].split(',')  # assuming last line has the goodies; warnings, if any, appear before
    return width, height, depth
    

def get_pixel_format_for_metric_computation(subsampling):
    """ helper to set pixel format from subsampling, assuming full range,
        for metric computation
    """
    pixel_format = None
    if subsampling == '420':
        pixel_format = 'yuvj420p'
    elif subsampling == '444':
        pixel_format = 'yuvj444p'
    elif subsampling == '444u':
        pixel_format = 'yuvj444p'
    else:
        raise RuntimeError('Unsupported subsampling ' + subsampling)
    return pixel_format


def get_pixel_format_for_encoding(subsampling):
    """ helper to set pixel format from subsampling, assuming full range,
        for converting source to yuv prior to encoding
    """
    pixel_format = None
    if subsampling == '420':
        pixel_format = 'yuvj420p'
    elif subsampling == '444':
        pixel_format = 'yuvj444p'
    elif subsampling == '444u':
        pixel_format = 'yuvj420p'
    else:
        raise RuntimeError('Unsupported subsampling ' + subsampling)
    return pixel_format


def compute_vmaf(ref_image, dist_image, width, height, temp_folder, subsampling):
    """ given a pair of reference and distorted images:
        use the ffmpeg libvmaf filter to compute vmaf, vif, ssim, and ms_ssim.
    """

    pixel_format = get_pixel_format_for_metric_computation(subsampling)

    log_path = get_filename_with_temp_folder(temp_folder, 'stats.json')
    cmd = ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', pixel_format, '-s:v', '%s,%s' % (width, height), '-i', dist_image,
           '-f', 'rawvideo', '-pix_fmt', pixel_format, '-s:v', '%s,%s' % (width, height), '-i', ref_image,
           '-lavfi', 'libvmaf=psnr=true:ssim=true:ms_ssim=true:log_fmt=json:log_path=' + log_path,
           '-f', 'null', '-'
           ]

    run_program(cmd)

    vmaf_log = json.load(open(log_path))

    vmaf_dict = dict()
    vmaf_dict["vmaf"] = vmaf_log["frames"][0]["metrics"]["vmaf"]
    vmaf_dict["vif"] = vmaf_log["frames"][0]["metrics"]["vif_scale0"]
    vmaf_dict["ssim"] = vmaf_log["frames"][0]["metrics"]["ssim"]
    vmaf_dict["ms_ssim"] = vmaf_log["frames"][0]["metrics"]["ms_ssim"]
    vmaf_dict["adm2"] = vmaf_log["frames"][0]["metrics"]["adm2"]
    return vmaf_dict


def compute_psnr(ref_image, dist_image, width, height, temp_folder, subsampling):
    """ given a pair of reference and distorted images:
        use the ffmpeg psnr filter to compute psnr and mse for each channel.
    """

    pixel_format = get_pixel_format_for_metric_computation(subsampling)

    log_path = get_filename_with_temp_folder(temp_folder, 'stats.log')
    cmd = ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', pixel_format, '-s:v', '%s,%s' % (width, height), '-i', dist_image,
           '-f', 'rawvideo', '-pix_fmt', pixel_format, '-s:v', '%s,%s' % (width, height), '-i', ref_image,
           '-lavfi', 'psnr=stats_file=' + log_path,
           '-f', 'null', '-'
           ]

    run_program(cmd)

    psnr_dict = dict()
    psnr_log = open(log_path).read()
    for stat in psnr_log.rstrip().split(" "):
        key, value = stat.split(":")
        if key is not "n":
            psnr_dict[key] = float(value)
    return psnr_dict


def compute_metrics(ref_image, dist_image, width, height, temp_folder, subsampling):
    """ given a pair of reference and distorted images:
        call vmaf and psnr functions, return results in a dict.
    """

    vmaf = compute_vmaf(ref_image, dist_image, width, height, temp_folder, subsampling)
    psnr = compute_psnr(ref_image, dist_image, width, height, temp_folder, subsampling)
    stats = vmaf.copy()
    stats.update(psnr)
    return stats


def my_exec(cmd):
    """ helper to choose method for running commands
    """
    return run_program(cmd)


def get_filename_with_temp_folder(temp_folder, filename):
    """ helper to get filename with temp folder
    """
    return os.path.join(temp_folder, filename)


def convert_420_to_444_source_and_decoded(source_yuv_420, decoded_yuv_420, image, width, height, subsampling):
    suffix = '.420_to_444.yuv'
    source_yuv = source_yuv_420 + suffix
    decoded_yuv = decoded_yuv_420 + suffix

    # 444 source yuv is directly made from input source image
    cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_metric_computation(subsampling), source_yuv]
    my_exec(cmd)

    # 444 decoded yuv is made from previously decoded 420 yuv
    convert_420_to_444(decoded_yuv_420, decoded_yuv, width, height, subsampling)

    return source_yuv, decoded_yuv


def convert_420_to_444(input_420, output_444, width, height, subsampling):
    assert subsampling == '444u'
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', get_pixel_format_for_encoding(subsampling),
           '-s:v', '%s,%s' % (width, height), '-i', input_420,
           '-f', 'rawvideo', '-pix_fmt', get_pixel_format_for_metric_computation(subsampling), output_444]
    my_exec(cmd)


def kakadu_encode_helper(image, subsampling, temp_folder, source_yuv, param, codec):
    cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
    my_exec(cmd)

    encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.mj2')
    # kakadu derives width, height, sub-sampling from file name so avoid confusion with folder name
    cmd = ['/tools/kakadu/KDU805_Demo_Apps_for_Linux-x86-64_200602/kdu_v_compress', '-quiet', '-i',
           ntpath.basename(source_yuv),
           '-o', ntpath.basename(encoded_file), '-precise', '-rate', param, '-tolerance', '0']
    if codec == 'kakadu-mse':
        cmd[-2:-2] = ['-no_weights']
    run_program(cmd, cwd=temp_folder)

    decoded_yuv = get_filename_with_temp_folder(temp_folder, 'special_kakadu_decoded.yuv')

    for filePath in glob.glob(get_filename_with_temp_folder(temp_folder, 'special_kakadu_decoded_*.yuv')):
        try:
            os.remove(filePath)
        except:
            LOGGER.error("Error while deleting file : " + filePath)

    cmd = ['/tools/kakadu/KDU805_Demo_Apps_for_Linux-x86-64_200602/kdu_v_expand', '-quiet', '-i', encoded_file,
           '-o', decoded_yuv]
    my_exec(cmd)

    decoded_yuv_files = glob.glob(get_filename_with_temp_folder(temp_folder, 'special_kakadu_decoded_*.yuv'))
    assert len(decoded_yuv_files) == 1

    decoded_yuv = decoded_yuv_files[0]

    return encoded_file, decoded_yuv, source_yuv


def jpeg_encode_helper(codec, cmd, encoded_file, temp_folder):
    if codec == 'jpeg-mse':
        cmd[1:1] = ['-qt', '1']
    elif codec == 'jpeg-ms-ssim':
        cmd[1:1] = ['-qt', '2']
    elif codec == 'jpeg-im':
        cmd[1:1] = ['-qt', '3']
    elif codec == 'jpeg-hvs-psnr':
        cmd[1:1] = ['-qt', '4']
    my_exec(cmd)

    cmd = ['/tools/jpeg/jpeg', encoded_file, get_filename_with_temp_folder(temp_folder, 'decoded.ppm')]
    my_exec(cmd)


def f(param, codec, image, width, height, temp_folder, subsampling):
    """
    Method to run an encode with given codec, subsampling, etc. and requested codec parameter like QP or bpp.
    All the commands for encoding, decoding, computing metric, etc. for any given codec should be visible
    in one contiguous block of code. Don't care about repeated commands across codecs.
    :param param: codec param like QP, bpp, etc.
    :param codec: codec
    :param image: filename for source image
    :param width: width of source image and target encode
    :param height: height of source image and target encode
    :param temp_folder: directory for intermediate files, encodes, stats files, etc.
    :param subsampling: color subsampling
    :return:
    """
    param = str(param)
    LOGGER.debug("Encoding image " + image + " with codec " + codec + " for param " + param)

    encoded_file = get_filename_with_temp_folder(temp_folder, 'encoded_file_whoami')
    source_yuv = get_filename_with_temp_folder(temp_folder, 'source.yuv')
    decoded_yuv = get_filename_with_temp_folder(temp_folder, 'decoded.yuv')

    if codec in ['jpeg', 'jpeg-mse', 'jpeg-ms-ssim', 'jpeg-im', 'jpeg-hvs-psnr'] and subsampling in ['420', '444u']:
        cmd = ['convert', image, get_filename_with_temp_folder(temp_folder, 'source.ppm')]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.jpg')
        cmd = ['/tools/jpeg/jpeg', '-q', str(int(param)), '-s', '1x1,2x2,2x2', get_filename_with_temp_folder(temp_folder, 'source.ppm'),
               encoded_file]
        jpeg_encode_helper(codec, cmd, encoded_file, temp_folder)

        cmd = ['convert', get_filename_with_temp_folder(temp_folder, 'source.ppm'), '-interlace', 'plane', '-sampling-factor',
               '4:2:0' if subsampling == '420' else '4:4:4', source_yuv]
        my_exec(cmd)

        cmd = ['convert', get_filename_with_temp_folder(temp_folder, 'decoded.ppm'), '-interlace', 'plane', '-sampling-factor',
               '4:2:0' if subsampling == '420' else '4:4:4', decoded_yuv]
        my_exec(cmd)

    elif codec in ['jpeg', 'jpeg-mse', 'jpeg-ms-ssim', 'jpeg-im', 'jpeg-hvs-psnr'] and subsampling == '444':
        cmd = ['convert', image, get_filename_with_temp_folder(temp_folder, 'source.ppm')]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.jpg')
        cmd = ['/tools/jpeg/jpeg', '-q', str(int(param)), '-s', '1x1,1x1,1x1', get_filename_with_temp_folder(temp_folder, 'source.ppm'),
               encoded_file]
        jpeg_encode_helper(codec, cmd, encoded_file, temp_folder)

        cmd = ['convert', get_filename_with_temp_folder(temp_folder, 'source.ppm'), '-interlace', 'plane', '-sampling-factor',
               '4:4:4', source_yuv]
        my_exec(cmd)

        cmd = ['convert', get_filename_with_temp_folder(temp_folder, 'decoded.ppm'), '-interlace', 'plane', '-sampling-factor',
               '4:4:4', decoded_yuv]
        my_exec(cmd)

    elif codec == "webp" and subsampling in ['420', '444u']:
        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.webp')
        cmd = ['/tools/libwebp-1.0.2-linux-x86-64/bin/cwebp', '-m', '6', '-q', param, '-s', str(width), str(height),
               '-quiet', source_yuv, '-o', encoded_file]
        my_exec(cmd)

        cmd = ['/tools/libwebp-1.0.2-linux-x86-64/bin/dwebp', encoded_file, '-yuv', '-quiet', '-o', decoded_yuv]
        my_exec(cmd)

        if subsampling == '444u':
            source_yuv, decoded_yuv = convert_420_to_444_source_and_decoded(source_yuv, decoded_yuv, image, width, height, subsampling)

    elif codec in ['kakadu-mse', 'kakadu-visual'] and subsampling in ['420', '444u']:
        source_yuv = get_filename_with_temp_folder(temp_folder, 'kakadu_{}x{}_{}.yuv'.format(width, height, '420'))

        encoded_file, decoded_yuv, source_yuv = kakadu_encode_helper(image, subsampling, temp_folder, source_yuv, param,
                                                                     codec)
        if subsampling == '444u':
            source_yuv, decoded_yuv = convert_420_to_444_source_and_decoded(source_yuv, decoded_yuv, image, width, height, subsampling)

    elif codec in ['kakadu-mse', 'kakadu-visual'] and subsampling == '444':

        use_ppm_source = False

        if use_ppm_source:
            source_ppm = get_filename_with_temp_folder(temp_folder, 'source.ppm')
            decoded_ppm = get_filename_with_temp_folder(temp_folder, 'decoded.ppm')

            cmd = ['convert', image, source_ppm]
            my_exec(cmd)

            encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.mj2')
            cmd = ['/tools/kakadu/KDU7A2_Demo_Apps_for_Ubuntu-x86-64_170827/kdu_compress', '-quiet', '-i', source_ppm,
                   '-o', encoded_file, '-rate', param, '"Ssampling={1,1}"', '-precise', '-tolerance', '0']
            if codec == 'kakadu-mse':
                cmd[-4:-4] = ['-no_weights']
            my_exec(cmd)  # not sure whether yuv444 is being coded in the codestream

            cmd = ['/tools/kakadu/KDU7A2_Demo_Apps_for_Ubuntu-x86-64_170827/kdu_expand', '-quiet', '-i', encoded_file,
                   '-o', decoded_ppm]
            my_exec(cmd)

            cmd = ['convert', source_ppm, '-interlace', 'plane', '-sampling-factor', '4:4:4', source_yuv]
            my_exec(cmd)

            cmd = ['convert', decoded_ppm, '-interlace', 'plane', '-sampling-factor', '4:4:4', decoded_yuv]
            my_exec(cmd)
        else:
            source_yuv = get_filename_with_temp_folder(temp_folder, 'kakadu_{}x{}_{}.yuv'.format(width, height, '444'))

            encoded_file, decoded_yuv, source_yuv = kakadu_encode_helper(image, subsampling, temp_folder, source_yuv,
                                                                         param, codec)

    elif codec == 'openjpeg' and subsampling in ['420', '444u']:
        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.j2k')
        decoded_file = get_filename_with_temp_folder(temp_folder, 'decoded.ppm')
        source_raw = get_filename_with_temp_folder(temp_folder, 'source.raw')

        # cmd = ['convert', image, '-interlace', 'plane', '-sampling-factor', '4:2:0', source_yuv]
        # my_exec(cmd)

        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        cmd = ['cp', source_yuv, source_raw]
        my_exec(cmd)

        # param is PSNR value [dB]
        cmd = ['opj_compress', '-i', source_raw, '-F', '%s,%s,%s,%s,u@1x1:2x2:2x2' % (width,height,3,8), '-q', param, '-o', encoded_file]
        my_exec(cmd)

        cmd = ['opj_decompress', '-i', encoded_file, '-o', decoded_file]
        my_exec(cmd)

        cmd = ['convert', decoded_file, '-interlace', 'plane', '-sampling-factor',
               '4:2:0' if subsampling == '420' else '4:4:4', decoded_yuv]
        my_exec(cmd)

        if subsampling == '444u':
            cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_metric_computation(subsampling), source_yuv]
            my_exec(cmd)

    elif codec == 'openjpeg' and subsampling == '444':
        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.j2k')
        decoded_file = get_filename_with_temp_folder(temp_folder, 'decoded.raw')
        source_raw = get_filename_with_temp_folder(temp_folder, 'source.raw')

        # cmd = ['convert', image, '-interlace', 'plane', '-sampling-factor', '4:4:4', source_yuv]
        # my_exec(cmd)

        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        cmd = ['cp', source_yuv, source_raw]
        my_exec(cmd)

        # param is PSNR value [dB]
        cmd = ['opj_compress', '-i', source_raw, '-F', '%s,%s,%s,%s,u@1x1:1x1:1x1' % (width,height,3,8), '-q', param, '-o', encoded_file]
        my_exec(cmd)

        cmd = ['opj_decompress', '-i', encoded_file, '-o', decoded_file]
        my_exec(cmd)

        cmd = ['cp', decoded_file, decoded_yuv]
        my_exec(cmd)

    elif codec == 'hevc' and subsampling in ['420', '444u']:
        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.hevc')
        cmd = ['/tools/HM-16.20+SCM-8.8/bin/TAppEncoderStatic',
               '-c', '/tools/HM-16.20+SCM-8.8/cfg/encoder_intra_main_rext.cfg', '-f', '1', '-fr', '1', '-q', param,
               '-wdt', width, '-hgt', height, '--InputChromaFormat=420', '--ConformanceWindowMode=1',
               '-i', source_yuv, '-b', encoded_file, '-o', '/dev/null']
        my_exec(cmd)

        cmd = ['/tools/HM-16.20+SCM-8.8/bin/TAppDecoderStatic', '-b', encoded_file, '-d', '8', '-o', decoded_yuv]
        my_exec(cmd)

        if subsampling == '444u':
            source_yuv, decoded_yuv = convert_420_to_444_source_and_decoded(source_yuv, decoded_yuv, image, width, height, subsampling)

    elif codec == 'hevc' and subsampling == '444':
        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.hevc')
        cmd = ['/tools/HM-16.20+SCM-8.8/bin/TAppEncoderStatic',
               '-c', '/tools/HM-16.20+SCM-8.8/cfg/encoder_intra_main_rext.cfg', '-f', '1', '-fr', '1', '-q', param,
               '-wdt', width, '-hgt', height, '--InputChromaFormat=444', '--ConformanceWindowMode=1',
               '-i', source_yuv, '-b', encoded_file, '-o', '/dev/null']
        my_exec(cmd)

        cmd = ['/tools/HM-16.20+SCM-8.8/bin/TAppDecoderStatic', '-b', encoded_file, '-d', '8', '-o', decoded_yuv]
        my_exec(cmd)

    elif codec in ['avif-mse', 'avif-ssim'] and subsampling in ['420', '444u']:
        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.avif')
        cmd = ['aomenc', '--i420', '--width={}'.format(width), '--height={}'.format(height), '--ivf', '--cpu-used=1',
               '--end-usage=q', '--cq-level={}'.format(param), '--min-q={}'.format(param), '--max-q={}'.format(param),
               '--passes=2', '--input-bit-depth={}'.format(8), '--bit-depth={}'.format(8), '--lag-in-frames=0',
               '--frame-boost=0', '--disable-warning-prompt',
               '--output={}'.format(encoded_file), source_yuv]
        if codec == 'avif-ssim':
            cmd[-2:-2] = ['--tune=ssim']
        my_exec(cmd)

        cmd = ['aomdec', '--i420', '-o', decoded_yuv, encoded_file]
        my_exec(cmd)

        if subsampling == '444u':
            source_yuv, decoded_yuv = convert_420_to_444_source_and_decoded(source_yuv, decoded_yuv, image, width, height, subsampling)

    elif codec in ['avif-mse', 'avif-ssim'] and subsampling == '444':
        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.avif')
        cmd = ['aomenc', '--i444', '--width={}'.format(width), '--height={}'.format(height), '--ivf', '--cpu-used=1',
               '--end-usage=q', '--cq-level={}'.format(param), '--min-q={}'.format(param), '--max-q={}'.format(param),
               '--passes=2', '--input-bit-depth={}'.format(8), '--bit-depth={}'.format(8), '--lag-in-frames=0',
               '--frame-boost=0', '--disable-warning-prompt',
               '--output={}'.format(encoded_file), source_yuv]
        if codec == 'avif-ssim':
            cmd[-2:-2] = ['--tune=ssim']
        my_exec(cmd)

        cmd = ['aomdec', '--rawvideo', '-o', decoded_yuv, encoded_file]
        my_exec(cmd)

    elif codec in ['avifenc-sp-0', 'avifenc-sp-2', 'avifenc-sp-4', 'avifenc-sp-6', 'avifenc-sp-8',
                   'avifenc_ssim-sp-0', 'avifenc_ssim-sp-2', 'avifenc_ssim-sp-4', 'avifenc_ssim-sp-6',
                   'avifenc_ssim-sp-8',
                   'avifenc-sp-0-crf', 'avifenc-sp-2-crf', 'avifenc-sp-4-crf', 'avifenc-sp-6-crf',
                   'avifenc-sp-8-crf',
                   'avifenc_ssim-sp-0-crf', 'avifenc_ssim-sp-2-crf', 'avifenc_ssim-sp-4-crf', 'avifenc_ssim-sp-6-crf',
                   'avifenc_ssim-sp-8-crf'
                   ] \
            and subsampling in ['420', '444u', '444']:
        min_QP = int(param) - 1
        max_QP = int(param) + 1
        if codec.endswith('crf'):
            min_QP = 0
            max_QP = 63
            crf_val = int(param)
        info = codec.split('-')
        speed = info[2]

        cmd = ['ffmpeg', '-y', '-i', image, '-pix_fmt', get_pixel_format_for_encoding(subsampling), source_yuv]
        my_exec(cmd)

        source_y4m = get_filename_with_temp_folder(temp_folder, 'source.y4m')
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', get_pixel_format_for_encoding(subsampling),
               '-s:v', '%s,%s' % (width, height), '-i', source_yuv, source_y4m]
        my_exec(cmd)

        encoded_file = get_filename_with_temp_folder(temp_folder, 'temp.avif')
        # bit-depth and yuv subsampling are maintained for y4m input
        # can't hurt to specify -d 8
        if codec.endswith('crf'):
            # default tuning is PSNR; ssim can be added with "-a tune=ssim"
            cmd = ['/tools/libavif/build/avifenc', source_y4m, encoded_file, '-d', '8',
                   '--nclx', '1/13/6', '--min', str(min_QP), '--max', str(max_QP),
                   '-a', 'end-usage=q', '-a', 'cq-level={}'.format(crf_val),
                   '--speed', speed, '--codec', 'aom', '--jobs', '4']
        else:
            cmd = ['/tools/libavif/build/avifenc', source_y4m, encoded_file, '-d', '8',
                   '--nclx', '1/13/6', '--min', str(min_QP), '--max', str(max_QP),
                   '--speed', speed, '--codec', 'aom', '--jobs', '4']
        if codec.find('ssim') > 0:
            cmd.append('-a')
            cmd.append('tune=ssim')
        LOGGER.debug(cmd)
        my_exec(cmd)

        decoded_y4m = get_filename_with_temp_folder(temp_folder, 'decoded.y4m')
        cmd = ['/tools/libavif/build/avifdec', '--codec', 'aom', encoded_file, decoded_y4m]
        my_exec(cmd)

        # explicitly convert to 420 or 444 depending on the case from the y4m
        cmd = ['ffmpeg', '-y', '-i', decoded_y4m, '-pix_fmt', get_pixel_format_for_encoding(subsampling),
               decoded_yuv]
        my_exec(cmd)

        if subsampling == '444u':
            source_yuv, decoded_yuv = convert_420_to_444_source_and_decoded(source_yuv, decoded_yuv, image, width, height, subsampling)

    else:
        raise RuntimeError('Unsupported codec and subsampling ' + codec + ' / ' + subsampling)

    stats = compute_metrics(source_yuv, decoded_yuv, width, height, temp_folder, subsampling)
    stats['file_size_bytes'] = os.path.getsize(encoded_file)

    return stats, encoded_file


def make_my_tuple(image, width, height, codec, metric, target, subsampling, uuid=None):
    """ make unique tuple for unique directory, primary key in DB, etc.
    """
    my_tuple = '{image}_{width}x{height}_{codec}_{metric}_{target}_{subsampling}' \
        .format(image=ntpath.basename(image), width=width, height=height, codec=codec,
                metric=metric, target=target, subsampling=subsampling)
    if uuid is not None:
        my_tuple = uuid + my_tuple
    if len(my_tuple) > 255:  # limits due to max dir name or file name length on UNIX
        LOGGER.error("ERROR : Tuple too long : " + my_tuple)
    assert len(my_tuple) < 256
    return my_tuple


def bisection(inverse, a, b, ab_tol, metric, target, target_tol, codec, image, width, height, subsampling):
    """ Perform encode with given codec, subsampling, etc. with the goal of hitting given target quality as
    closely as possible. Employs binary search.
    :param inverse: boolean True means QP, else quality factor
    :param a: a should be less than b
    :param b: far-end of codec parameter range
    :param ab_tol: how close can a and b get before we exit, used as a terminating condition, just in case.
    :param metric: string, vmaf or PSNR
    :param target: target value of vmaf or PSNR
    :param target_tol: say 2 for VMAF, 0.2 for PSNR
    :param codec: string identifying codec
    :param image: source image
    :param width: width of source image and target encode
    :param height: height of source image and target encode
    :param subsampling: color subsampling
    :return:
    """
    temp_uuid = str(uuid.uuid4())
    temp_folder = make_my_tuple(image, width, height, codec, metric, target, subsampling, temp_uuid)
    tuple_minus_uuid = make_my_tuple(image, width, height, codec, metric, target, subsampling)
    mkdir_p(temp_folder)

    LOGGER.debug(repr((multiprocessing.current_process(), temp_folder,
                       inverse, a, b, ab_tol, metric, target, target_tol, codec, image, width, height, subsampling)))

    c = (a + b) / 2
    if isinstance(ab_tol, int):
        c = int(c)
    last_c = c
    while (b - a) > ab_tol:

        quality, encoded_file = f(c, codec, image, width, height, temp_folder, subsampling)
        last_c = c

        if abs(quality[metric] - target) < target_tol:
            return (last_c, codec, metric, target, quality, encoded_file, os.path.getsize(encoded_file), subsampling,
                    tuple_minus_uuid, ntpath.basename(image), width, height, temp_folder)
        else:
            if inverse:
                if quality[metric] < target:
                    b = c
                else:
                    a = c
            else:
                if quality[metric] < target:
                    a = c
                else:
                    b = c
            c = (a + b) / 2
            if isinstance(ab_tol, int):
                c = int(c)

    return (last_c, codec, metric, target, quality, encoded_file, os.path.getsize(encoded_file), subsampling,
            tuple_minus_uuid, ntpath.basename(image), width, height, temp_folder)


def get_insert_command():
    """ helper to get DB insert command
    """
    return '''INSERT INTO ENCODES (ID,SOURCE,WIDTH,HEIGHT,CODEC,CODEC_PARAM,
    TEMP_FOLDER,TARGET_METRIC,TARGET_VALUE,
    VMAF,SSIM,MS_SSIM,VIF,MSE_Y,MSE_U,MSE_V,MSE_AVG,PSNR_Y,PSNR_U,PSNR_V,PSNR_AVG,ADM2,
    SUB_SAMPLING,FILE_SIZE_BYTES,ENCODED_FILE) 
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''


def get_create_table_command():
    """ helper to get create table command
    """
    return '''CREATE TABLE ENCODES
         (ID TEXT PRIMARY KEY     NOT NULL,
         SOURCE           TEXT    NOT NULL,
         WIDTH            INT     NOT NULL,
         HEIGHT           INT     NOT NULL,
         CODEC            TEXT    NOT NULL,
         CODEC_PARAM      REAL    NOT NULL,
         TEMP_FOLDER      TEXT    NOT NULL,
         TARGET_METRIC    TEXT    NOT NULL,
         TARGET_VALUE     REAL    NOT NULL,
         VMAF             REAL    NOT NULL,
         SSIM             REAL    NOT NULL,
         MS_SSIM          REAL    NOT NULL,
         VIF              REAL    NOT NULL,
         MSE_Y            REAL    NOT NULL,
         MSE_U            REAL    NOT NULL,
         MSE_V            REAL    NOT NULL,
         MSE_AVG          REAL    NOT NULL,
         ADM2             REAL    NOT NULL,
         PSNR_Y           REAL    NOT NULL,
         PSNR_U           REAL    NOT NULL,
         PSNR_V           REAL    NOT NULL,
         PSNR_AVG         REAL    NOT NULL,
         SUB_SAMPLING     TEXT    NOT NULL,
         FILE_SIZE_BYTES  REAL    NOT NULL,
         ENCODED_FILE     TEXT    NOT NULL);'''


def update_stats(results):
    """ callback function called when a worker process finishes an encoding job with target quality value
    """
    c, codec, metric, target, quality, encoded_file, file_size_bytes, subsampling, \
    tuple_minus_uuid, source_image, width, height, temp_folder = results
    LOGGER.info('<<' + codec.upper() + '>>' + " Param " + str(c) + ", quality "
                + repr(quality) + ", encoded_file: " + encoded_file
                + " size: " + str(file_size_bytes) + " bytes")
    TOTAL_BYTES[codec + subsampling + metric + str(target)] += os.path.getsize(encoded_file)
    TOTAL_METRIC[codec + subsampling + metric + str(target)] += quality[metric]
    CONNECTION.execute(get_insert_command(), (tuple_minus_uuid, source_image, width, height,
                                              codec, c, temp_folder, metric, target,
                                              quality['vmaf'], quality['ssim'], quality['ms_ssim'], quality['vif'],
                                              quality['mse_y'], quality['mse_u'], quality['mse_v'], quality['mse_avg'],
                                              quality['psnr_y'], quality['psnr_u'], quality['psnr_v'],
                                              quality['psnr_avg'], quality['adm2'],
                                              subsampling, file_size_bytes, encoded_file))
    CONNECTION.commit()
    remove_files_keeping_encode(temp_folder, encoded_file)  # comment out to keep all files


def remove_files_keeping_encode(temp_folder, encoded_file):
    """ remove files but keep encode
    """
    for f in listdir_full_path(temp_folder):
        if ntpath.basename(f) != ntpath.basename(encoded_file):
            os.remove(f)


def error_function(error):
    """ error callback called when an encoding job in a worker process encounters an exception
    """
    LOGGER.error('***** ATTENTION %s', type(error))
    LOGGER.error('***** ATTENTION %s', repr(error))


def initialize_worker():
    """ method called before a worker process picks up jobs
    """
    setup_logging(worker=True, worker_id=multiprocessing.current_process().ident)
    LOGGER.info('initialize_worker() called for %s %s', multiprocessing.current_process(),
                multiprocessing.current_process().ident)


def create_table_if_needed(connection, only_perform_missing_encodes):
    if only_perform_missing_encodes:
        if connection.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ENCODES' ''').fetchall()[0][0] == 1:
            LOGGER.info('Will add missing entries to table')
        else:
            LOGGER.error('only_perform_missing_encodes is True but table does not exist')
            sys.exit(1)
    else:
        if connection.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ENCODES' ''').fetchall()[0][0] == 1:
            LOGGER.error('Table already exists. Exiting . . . ')
            sys.exit(1)
        connection.execute(get_create_table_command())


def does_entry_exist(connection, primary_key):
    res = connection.execute("SELECT * FROM ENCODES WHERE ID='{}'".format(primary_key)).fetchall()
    if len(res) > 1:
        LOGGER.error('Found more than one entry with given primary key')
        sys.exit(1)
    elif len(res) == 1:
        return True
    else:
        return False


def main(metric, target_arr, target_tol, db_file_name, only_perform_missing_encodes=False):
    """ create a pool of worker processes and submit encoding jobs, collect results and exit
    """
    setup_logging(worker=False, worker_id=multiprocessing.current_process().ident)
    LOGGER.info(
        'started main, current thread ID %s %s %s', multiprocessing.current_process(),
        multiprocessing.current_process().ident,
        threading.current_thread().ident)

    if only_perform_missing_encodes:
        if os.path.isfile(db_file_name):
            LOGGER.info("Will add missing entries to file " + db_file_name)
        else:
            LOGGER.error("only_perform_missing_encodes is True but db file " + db_file_name + " does not exist.")
            sys.exit(1)

    global CONNECTION
    CONNECTION = sqlite3.connect(db_file_name, check_same_thread=False)
    create_table_if_needed(CONNECTION, only_perform_missing_encodes)

    pool = multiprocessing.Pool(processes=4, initializer=initialize_worker)
    images = set(listdir_full_path('images'))
    if len(images) <= 0:
        LOGGER.error(" no source files in ./images.")
        sys.exit(1)

    results = list()

    for target in target_arr:
        for num, image in enumerate(images):
            width, height, depth = get_dimensions(image)
            LOGGER.info("[" + str(num) + "] Source image: " + image + " {" + width + "x" + height + "} bit-depth: " + depth)

            for codec in TUPLE_CODECS:
                LOGGER.debug(" ")
                skip_encode = False
                if only_perform_missing_encodes:
                    unique_id = make_my_tuple(image, width, height, codec.name, metric, target, codec.subsampling)
                    skip_encode = does_entry_exist(CONNECTION, unique_id)

                if not skip_encode:
                    results.append(
                        (pool.apply_async(bisection,
                                          args=(codec.inverse, codec.param_start, codec.param_end, codec.ab_tol,
                                                metric, target, target_tol, codec.name, image, width, height,
                                                codec.subsampling),
                                          callback=update_stats, error_callback=error_function), codec.name,
                         codec.subsampling))
            LOGGER.info('-----------------------------------------------------------------------------------------')

    pool.close()
    pool.join()

    # time.sleep(4)

    LOGGER.debug(" ")
    LOGGER.debug("Will get results from AsyncResult objects and list them now. This ensures callbacks complete.")
    for result in results:
        try:
            task_result = result[0].get()
            LOGGER.info(repr(task_result))
        except Exception as e:  ## all the exceptions encountered during parallel execution can be collected here at the end nicely
            LOGGER.error(repr(e))
            TOTAL_ERRORS[result[1] + result[2]] += 1

    LOGGER.info("\n\n")
    if not only_perform_missing_encodes:
        LOGGER.info("Total payload in kilo Bytes:")
        for codec in TUPLE_CODECS:
            for target in target_arr:
                if codec.name + codec.subsampling in TOTAL_ERRORS:
                    LOGGER.info('  {}: {} (Total errors {})'.format(codec.name + codec.subsampling + metric + str(target),
                                                                    TOTAL_BYTES[codec.name + codec.subsampling + metric + str(target)] / 1000.0,
                                                                    TOTAL_ERRORS[codec.name + codec.subsampling + metric + str(target)]))
                else:
                    LOGGER.info('  {}: {}'.format(codec.name + codec.subsampling + metric + str(target),
                                                  TOTAL_BYTES[codec.name + codec.subsampling + metric + str(target)] / 1000.0))

        LOGGER.info("Average metric value:")
        for codec in TUPLE_CODECS:
            for target in target_arr:
                if codec.name + codec.subsampling in TOTAL_ERRORS:
                    LOGGER.info('  {}: {} (Total errors {})'.format(codec.name + codec.subsampling + metric + str(target),
                                                                    TOTAL_METRIC[codec.name + codec.subsampling + metric + str(target)]
                                                                    / float(len(images)),
                                                                    TOTAL_ERRORS[codec.name + codec.subsampling + metric + str(target)]))
                else:
                    LOGGER.info('  {}: {}'.format(codec.name + codec.subsampling + metric + str(target),
                                                  TOTAL_METRIC[codec.name + codec.subsampling + metric + str(target)] / float(len(images))))

    CONNECTION.close()
    # sys.exit(0)


if __name__ == "__main__":
    # if some encodes don't materialize, you can break out with Ctrl+C
    # then comment this out and run below for missing encodes
    main(metric='ssim', target_arr=[0.92, 0.95, 0.97, 0.99], target_tol=0.005, db_file_name='encoding_results_ssim.db')
    main(metric='vmaf', target_arr=[75, 80, 85, 90, 95], target_tol=0.5, db_file_name='encoding_results_vmaf.db')

    # to perform encodes targeting a certain file size
    # main(metric='file_size_bytes', target_arr=[4000, 20000, 40000, 80000], target_tol=500, db_file_name='encoding_results_file_size.db')

    # to run missing encodes
    # main(metric='ssim', target_arr=[0.92, 0.95, 0.97, 0.99], target_tol=0.005, db_file_name='encoding_results_ssim.db', only_perform_missing_encodes=True)
    # main(metric='vmaf', target_arr=[75, 80, 85, 90, 95], target_tol=0.5, db_file_name='encoding_results_vmaf.db', only_perform_missing_encodes=True)

    # Results:
    # Can be easily viewed in Python, say the Python Interactive Console, like so:
    #
    # import sqlite3
    #
    # conn = sqlite3.connect('encoding_results.db')
    # all_results = conn.execute('select * from encodes').fetchall()
    # OR
    # some_results = conn.execute('select codec,sub_sampling,vmaf,file_size_bytes from encodes').fetchall()

