import sqlite3
import sys
import logging
import ntpath
from statistics import mean
from collections import namedtuple
from collections import defaultdict
from bd_rate_calculator import BDrateCalculator
from analyze_encoding_results import apply_size_check

__author__ = "Aditya Mavlankar"
__copyright__ = "Copyright 2019-2020, Netflix, Inc."
__credits__ = ["Kyle Swanson", "Jan de Cock", "Marjan Parsa"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__maintainer__ = "Aditya Mavlankar"
__email__ = "amavlankar@netflix.com"
__status__ = "Development"

RateQualityPoint = namedtuple('RateQualityPoint', ['bpp', 'quality', 'target_metric', 'target_value'])
BD_RATE_EXCEPTION_STRING = 'BD_RATE_EXCEPTION'


def get_unique_sources_sorted(connection):
    unique_sources = connection.execute('SELECT DISTINCT SOURCE FROM ENCODES').fetchall()
    unique_sources = [elem[0] for elem in unique_sources]
    return sorted(list(set(unique_sources)))


def get_quality_dict(elem, list_of_metrics):
    quality = dict()
    for index, metric in enumerate(list_of_metrics):
        quality[metric] = elem[index]
    return quality


def get_rate_quality_points(connection, sub_sampling, codec, source, total_pixels, list_of_metrics):
    # print('{} {} {}'.format(codec, sub_sampling, source))
    csv_metrics_upper = ','.join([elem.upper() for elem in list_of_metrics])
    points = connection.execute("SELECT {},FILE_SIZE_BYTES,TARGET_METRIC,TARGET_VALUE FROM ENCODES WHERE CODEC='{}' AND SUB_SAMPLING='{}' AND SOURCE='{}'"
                                .format(csv_metrics_upper, codec, sub_sampling, source)).fetchall()
    rate_quality_points = [
        RateQualityPoint(elem[len(list_of_metrics)] * 8 / total_pixels, get_quality_dict(elem, list_of_metrics),
                         elem[len(list_of_metrics) + 1], elem[len(list_of_metrics) + 2]) for elem in points]
    # print(repr(rate_quality_points))
    return rate_quality_points


def get_rates(rate_quality_points):
    return [rate_quality_point.bpp for rate_quality_point in rate_quality_points]


def get_quality(rate_quality_points, metric):
    return [rate_quality_point.quality[metric] for rate_quality_point in rate_quality_points]


def get_formatted_bdrate(val):
    if isinstance(val, str):
        return val
    else:
        return '{:.2f}'.format(val).rjust(6)


def get_formatted_mean_bdrate(val):
    return '{:.2f}'.format(val).rjust(22)


def my_shorten(name, width):
    return (name[:width - 3] + '...') if len(name) > width else name


def print_bd_rates(bdrates_various_metrics, codec, unique_sources, black_list_source_various_metrics,
                   list_of_metrics):
    bdrates_this_codec_various_metrics = dict()
    for metric in list_of_metrics:
        bdrates_this_codec_various_metrics[metric] = list()
    max_len_source_name = len(max(unique_sources, key=len))
    max_len_to_use_for_printing = min(80, max_len_source_name)
    for source in unique_sources:
        print_string = '  {} {}'.format(my_shorten(source,
                                                    max_len_to_use_for_printing)
                                         .ljust(max_len_to_use_for_printing),
                                         codec)
        for metric in list_of_metrics:
            print_string += ' BDRate-{} {}'.format(metric.upper(), get_formatted_bdrate(bdrates_various_metrics[metric][codec][source]))
            if source not in black_list_source_various_metrics[metric]:
                bdrates_this_codec_various_metrics[metric].append(bdrates_various_metrics[metric][codec][source])
        print(print_string)
    result = codec.ljust(16)
    result_local = result
    for metric in list_of_metrics:
        result += '{}'.format(get_formatted_mean_bdrate(mean(bdrates_this_codec_various_metrics[metric])))
        result_local += '   Mean BDRate-{} {:.2f}'.format(metric.upper(), mean(bdrates_this_codec_various_metrics[metric]))
    print(result_local + '\n')
    return result


def main(argv):
    db_file_name = 'encoding_results_vmaf.db'
    if len(argv) > 0:
        db_file_name = argv[0]
    connection = sqlite3.connect(db_file_name)

    logger = logging.getLogger('report.bdrates')
    logger.addHandler(logging.FileHandler('bdrates_' + ntpath.basename(db_file_name) + '.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('DEBUG')

    unique_sources = get_unique_sources_sorted(connection)
    total_pixels = apply_size_check(connection)

    baseline_codec = 'jpeg'
    sub_sampling_arr = ['420', '444']
    codecs = ['jpeg-mse', 'jpeg-ms-ssim', 'jpeg-im', 'jpeg-hvs-psnr', 'webp', 'kakadu-mse', 'kakadu-visual', 'openjpeg',
              'hevc', 'avif-mse', 'avif-ssim']
    metrics_for_BDRate = ['vmaf', 'ssim', 'ms_ssim', 'vif', 'psnr_y', 'psnr_avg']
    for sub_sampling in sub_sampling_arr:
        bdrates_various_metrics = dict()
        black_list_source_various_metrics = dict()
        for metric in metrics_for_BDRate:
            bdrates_various_metrics[metric] = defaultdict(dict)
            black_list_source_various_metrics[metric] = list()
        print('\n\nComputing BD rates for subsampling {}'.format(sub_sampling))
        for source in unique_sources:
            baseline_rate_quality_points = get_rate_quality_points(connection, sub_sampling, baseline_codec, source, total_pixels, metrics_for_BDRate)
            for codec in codecs:
                if codec == 'webp' and sub_sampling == '444':
                    continue
                rate_quality_points = get_rate_quality_points(connection, sub_sampling, codec, source, total_pixels, metrics_for_BDRate)

                for metric in metrics_for_BDRate:
                    # print(metric.upper())
                    try:
                        bd_rate_val = 100.0 * BDrateCalculator.CalcBDRate(
                            list(zip(get_rates(baseline_rate_quality_points), get_quality(baseline_rate_quality_points, metric))),
                            list(zip(get_rates(rate_quality_points), get_quality(rate_quality_points, metric))))
                        bdrates_various_metrics[metric][codec][source] = bd_rate_val
                    except AssertionError as e:
                        print('{} {} {} {}: '.format(metric, source, codec, sub_sampling) + str(e))
                        bdrates_various_metrics[metric][codec][source] = BD_RATE_EXCEPTION_STRING
                        # BD rate computation failed for one of the codecs,
                        # so to be fair, ignore this source for final results
                        if source not in black_list_source_various_metrics[metric]:
                            black_list_source_various_metrics[metric].append(source)

        for metric in metrics_for_BDRate:
            print('{} black list {} BD RATE\n '.format(sub_sampling, metric.upper()) + repr(black_list_source_various_metrics[metric]))
        results = dict()
        for codec in codecs:
            if codec == 'webp' and sub_sampling == '444':
                continue
            print('Codec {} subsampling {}, BD rates:'.format(codec, sub_sampling))
            result = print_bd_rates(bdrates_various_metrics, codec, unique_sources,
                                    black_list_source_various_metrics,
                                    metrics_for_BDRate)
            results[codec] = result
        logger.info('\n\n===================' + '=' * 22 * len(metrics_for_BDRate))
        results_header = 'Results for subsampling {}'.format(sub_sampling)
        logger.info(results_header)
        logger.info('-' * len(results_header))
        table_header = 'Codec'.ljust(16)
        for metric in metrics_for_BDRate:
            table_header += ' Mean BDRate-{}'.format(metric.upper()).rjust(22)
        logger.info(table_header)
        for codec in codecs:
            if codec == 'webp' and sub_sampling == '444':
                continue
            logger.info(results[codec])
        logger.info('===================' + '=' * 22 * len(metrics_for_BDRate))


if __name__ == '__main__':
    main(sys.argv[1:])

