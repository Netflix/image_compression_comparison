import sqlite3
import statistics
import sys
import logging
import ntpath
import matplotlib.pyplot as plt
from collections import defaultdict

__author__ = "Aditya Mavlankar"
__copyright__ = "Copyright 2019-2020, Netflix, Inc."
__credits__ = ["Kyle Swanson", "Jan de Cock", "Marjan Parsa"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__maintainer__ = "Aditya Mavlankar"
__email__ = "amavlankar@netflix.com"
__status__ = "Development"


def query_for_codec(codec, sub_sampling, target_metric, target_value):
    return "SELECT {},FILE_SIZE_BYTES,VMAF FROM ENCODES WHERE CODEC='{}' AND SUB_SAMPLING='{}' AND TARGET_METRIC='{}' AND TARGET_VALUE={}" \
        .format(target_metric.upper(), codec, sub_sampling, target_metric, target_value)


def get_metric_value_file_size_bytes(results):
    metric_values = [elem[0] for elem in results]
    file_size_values = [elem[1] for elem in results]
    vmaf_values = [elem[2] for elem in results]
    return metric_values, file_size_values, vmaf_values


def get_mean_metric_value_file_size_bytes(results):
    metric_values, file_size_values, vmaf_values = get_metric_value_file_size_bytes(results)
    return statistics.mean(metric_values), statistics.mean(file_size_values), len(metric_values), statistics.mean(vmaf_values)


def get_mean_metric_print(metric_name, metric_value, vmaf_value):
    if metric_name.upper() == 'SSIM':
        return '{:.5f} (mean VMAF {:.2f})'.format(metric_value, vmaf_value)
    else:
        return '{:.2f}'.format(metric_value)


def get_print_string(codec, sub_sampling, count, metric_value, file_size, metric_name, vmaf_value):
    line = '{} {} ({} images): mean {} {}, mean file size in bytes {}'.format(codec,
                                                                              sub_sampling,
                                                                              count,
                                                                              metric_name.upper(),
                                                                              get_mean_metric_print(metric_name,
                                                                                                    metric_value,
                                                                                                    vmaf_value),
                                                                              file_size)
    return line


def apply_size_check(connection):
    width_height_pairs = connection.execute('SELECT DISTINCT WIDTH,HEIGHT FROM ENCODES').fetchall()
    total_pixels = width_height_pairs[0][0] * width_height_pairs[0][1]
    for pair in width_height_pairs:
        if pair[0] * pair[1] != total_pixels:
            print('Images with different number of pixels detected in the database.')
            print('Cannot aggregate results for images with different number of pixels.')
            sys.exit(1)
    return total_pixels


def apply_checks_before_analyzing(connection, metric_name):
    target_metrics_in_db = connection.execute('SELECT DISTINCT TARGET_METRIC FROM ENCODES').fetchall()
    target_metrics_in_db = [elem[0] for elem in target_metrics_in_db]
    if metric_name not in target_metrics_in_db:
        print('Target metric {} not found in database. Target metrics in db {}.'.format(metric_name, repr(target_metrics_in_db)))
        sys.exit(1)
    total_pixels = apply_size_check(connection)
    all_metric_values = connection.execute('SELECT DISTINCT TARGET_VALUE FROM ENCODES').fetchall()
    all_metric_values = [elem[0] for elem in all_metric_values]
    unique_sorted_metric_values = sorted(list(set(all_metric_values)))
    return unique_sorted_metric_values, total_pixels


def main(argv):
    metric_name = 'vmaf'
    # metric_name = 'ssim'
    db_file_name = 'encoding_results_{}.db'.format(metric_name)
    if len(argv) > 0:
        if len(argv) != 2:
            print('Need 2 arguments when explicitly supplying arguments')
            print('  python3 analyze_encoding_results.py [metric_name] [db_file_name]')
            sys.exit(1)
        metric_name = argv[0]
        db_file_name = argv[1]
    connection = sqlite3.connect(db_file_name)
    unique_sorted_metric_values, total_pixels = apply_checks_before_analyzing(connection, metric_name)

    logger = logging.getLogger('report.bitrate_savings')
    logger.addHandler(logging.FileHandler('bitrate_savings_' + ntpath.basename(db_file_name) + '.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('DEBUG')

    baseline_codec = 'jpeg'
    sub_sampling_arr = ['420', '444']
    codecs = ['jpeg-mse', 'jpeg-ms-ssim', 'jpeg-im', 'jpeg-hvs-psnr', 'webp', 'kakadu-mse', 'kakadu-visual', 'openjpeg',
              'hevc', 'avif-mse', 'avif-ssim']
    plot_list = ['webp', 'kakadu-mse', 'kakadu-visual', 'hevc', 'avif-mse', 'avif-ssim']
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    marker_list = ['o', 'v', '>', '<', 's', 'p', 'd', '4', 'P', 'X']
    assert len(color_list) == len(marker_list)  # 10 curves on one plot is the limit, beyond that is sensory overload
    assert len(plot_list) <= len(marker_list)
    for sub_sampling in sub_sampling_arr:
        results_dict = dict()
        results_bpp = defaultdict(list)
        results_quality = defaultdict(list)
        for target in unique_sorted_metric_values:
            baseline_results = connection.execute(
                query_for_codec(baseline_codec, sub_sampling, metric_name, target)).fetchall()
            baseline_metric_value, baseline_file_size, baseline_count, baseline_vmaf_value = get_mean_metric_value_file_size_bytes(baseline_results)
            print('Baseline is ' + get_print_string(baseline_codec, sub_sampling, baseline_count, baseline_metric_value,
                                                    baseline_file_size, metric_name, baseline_vmaf_value))
            results_bpp[baseline_codec].append(baseline_file_size * 8.0 / total_pixels)
            results_quality[baseline_codec].append(baseline_metric_value)
            results_list = list()
            results_list_terse = list()
            for codec in codecs:
                if codec == 'webp' and sub_sampling == '444':
                    continue
                results = connection.execute(query_for_codec(codec, sub_sampling, metric_name, target)).fetchall()
                metric_value, file_size, count, vmaf_value = get_mean_metric_value_file_size_bytes(results)
                print(' Compared codec is ' + get_print_string(codec, sub_sampling, count, metric_value, file_size, metric_name, vmaf_value))
                # negative is better. Positive means increase in file_size
                print('  Average reduction is {:.2f}%'.format((file_size - baseline_file_size) / baseline_file_size * 100.0))
                results_list.append('{} {:.2f}%'.format(codec, (file_size - baseline_file_size) / baseline_file_size * 100.0))
                results_list_terse.append(
                    '{:.2f}%'.format((file_size - baseline_file_size) / baseline_file_size * 100.0).rjust(16))
                results_bpp[codec].append(file_size * 8.0 / total_pixels)
                results_quality[codec].append(metric_value)
            results_dict[target] = (results_list, results_list_terse)
            print("")
        print('\n')
        logger.info('=' * (8 + 16 * len(codecs)))
        sub_sampling_report = '{} subsampling'.format(sub_sampling)
        logger.info(sub_sampling_report)
        logger.info('-' * len(sub_sampling_report))
        codecs_string = ' ' * 8
        for codec in codecs:
            if codec == 'webp' and sub_sampling == '444':
                continue
            codecs_string += codec.rjust(16)
        logger.info(codecs_string)
        for target in unique_sorted_metric_values:
            all_codec_results, all_codec_results_terse = results_dict[target]
            consolidated_results = ""
            for a in all_codec_results_terse:
                consolidated_results += a.ljust(16)
            logger.info('{} : {}'.format(str(target).ljust(5), consolidated_results))
        logger.info('=' * (8 + 16 * len(codecs)))
        logger.info("\n\n")

        fig = plt.figure(figsize=(12.8, 7.2))
        marker_num = 0
        plt.plot(results_bpp[baseline_codec], results_quality[baseline_codec], linewidth=2.0,
                 color=color_list[marker_num], marker=marker_list[marker_num], label=baseline_codec)
        for codec in codecs:
            if codec in plot_list:
                if codec == 'webp' and sub_sampling == '444':
                    marker_num += 1
                    continue
                marker_num += 1
                plt.plot(results_bpp[codec], results_quality[codec], linewidth=2.0, color=color_list[marker_num],
                         marker=marker_list[marker_num], label=codec)
        plt.legend(loc='lower right')
        plt.grid()
        plt.xlabel('bit per pixel [bpp]')
        plt.ylabel(metric_name)
        plt.title('{} subsampling, using metric {}'.format(sub_sampling, metric_name.upper()))
        plt.tight_layout()
        fig.savefig('{}_{}_{}.png'.format(sub_sampling, metric_name, ntpath.basename(db_file_name)))


if __name__ == '__main__':
    main(sys.argv[1:])

