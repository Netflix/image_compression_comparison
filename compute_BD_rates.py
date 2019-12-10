import sqlite3
import sys
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

RateQualityPoint = namedtuple('RateQualityPoint', ['bpp', 'vmaf', 'ssim', 'target_metric', 'target_value'])
BD_RATE_EXCEPTION_STRING = 'BD_RATE_EXCEPTION'


def get_unique_sources_sorted(connection):
    unique_sources = connection.execute('SELECT DISTINCT SOURCE FROM ENCODES').fetchall()
    unique_sources = [elem[0] for elem in unique_sources]
    return sorted(list(set(unique_sources)))


def get_rate_quality_points(connection, sub_sampling, codec, source, total_pixels):
    # print('{} {} {}'.format(codec, sub_sampling, source))
    points = connection.execute("SELECT VMAF,SSIM,FILE_SIZE_BYTES,TARGET_METRIC,TARGET_VALUE FROM ENCODES WHERE CODEC='{}' AND SUB_SAMPLING='{}' AND SOURCE='{}'"
                                .format(codec, sub_sampling, source)).fetchall()
    rate_quality_points = [RateQualityPoint(elem[2] / total_pixels, elem[0], elem[1], elem[3], elem[4]) for elem in points]
    # print(repr(rate_quality_points))
    return rate_quality_points


def get_rates(rate_quality_points):
    return [rate_quality_point.bpp for rate_quality_point in rate_quality_points]


def get_vmaf(rate_quality_points):
    return [rate_quality_point.vmaf for rate_quality_point in rate_quality_points]


def get_ssim(rate_quality_points):
    return [rate_quality_point.ssim for rate_quality_point in rate_quality_points]


def get_formatted_bdrate(val):
    if isinstance(val, str):
        return val
    else:
        return '{:.2f}'.format(val).rjust(6)


def get_formatted_mean_bdrate(val):
    return '{:.2f}'.format(val).rjust(6)


def my_shorten(name, width):
    return (name[:width - 3] + '...') if len(name) > width else name


def print_bd_rates(bdrates_vmaf, bdrates_ssim, codec, unique_sources, black_list_source_VMAF_BDRATE,
                   black_list_source_SSIM_BDRATE):
    bdrates_vmaf_this_codec = list()
    bdrates_ssim_this_codec = list()
    max_len_source_name = len(max(unique_sources, key=len))
    max_len_to_use_for_printing = min(80, max_len_source_name)
    for source in unique_sources:
        print('  {} {} BDRate-VMAF {} BDRate-SSIM {}'.format(my_shorten(source,
                                                                        max_len_to_use_for_printing)
                                                             .ljust(max_len_to_use_for_printing),
                                                             codec,
                                                             get_formatted_bdrate(bdrates_vmaf[codec][source]),
                                                             get_formatted_bdrate(bdrates_ssim[codec][source])))
        if source not in black_list_source_VMAF_BDRATE:
            bdrates_vmaf_this_codec.append(bdrates_vmaf[codec][source])
        if source not in black_list_source_SSIM_BDRATE:
            bdrates_ssim_this_codec.append(bdrates_ssim[codec][source])
    result = '{} Mean BDRate-VMAF {}   Mean BDRate-SSIM {}'.format(codec.ljust(16),
                                                                   get_formatted_mean_bdrate(
                                                                       mean(bdrates_vmaf_this_codec)),
                                                                   get_formatted_mean_bdrate(
                                                                       mean(bdrates_ssim_this_codec)))
    print(result + '\n')
    return mean(bdrates_vmaf_this_codec), mean(bdrates_ssim_this_codec), result


def main(argv):
    db_file_name = 'encoding_results_vmaf.db'
    if len(argv) > 0:
        db_file_name = argv[0]
    connection = sqlite3.connect(db_file_name)
    unique_sources = get_unique_sources_sorted(connection)
    total_pixels = apply_size_check(connection)

    baseline_codec = 'jpeg'
    sub_sampling_arr = ['420', '444']
    codecs = ['jpeg-mse', 'jpeg-ms-ssim', 'jpeg-im', 'jpeg-hvs-psnr', 'webp', 'kakadu-mse', 'kakadu-visual', 'openjpeg',
              'hevc', 'avif-mse', 'avif-ssim']

    for sub_sampling in sub_sampling_arr:
        bdrates_vmaf = defaultdict(dict)
        bdrates_ssim = defaultdict(dict)
        print('\n\nComputing BD rates for subsampling {}'.format(sub_sampling))
        black_list_source_VMAF_BDRATE = list()
        black_list_source_SSIM_BDRATE = list()
        for source in unique_sources:
            baseline_rate_quality_points = get_rate_quality_points(connection, sub_sampling, baseline_codec, source, total_pixels)
            for codec in codecs:
                if codec == 'webp' and sub_sampling == '444':
                    continue
                rate_quality_points = get_rate_quality_points(connection, sub_sampling, codec, source, total_pixels)
                # print('VMAF')
                try:
                    bd_rate_vmaf = 100.0 * BDrateCalculator.CalcBDRate(
                        list(zip(get_rates(baseline_rate_quality_points), get_vmaf(baseline_rate_quality_points))),
                        list(zip(get_rates(rate_quality_points), get_vmaf(rate_quality_points))))
                    bdrates_vmaf[codec][source] = bd_rate_vmaf
                except AssertionError as e:
                    print('vmaf {} {} {}: '.format(source, codec, sub_sampling) + str(e))
                    bdrates_vmaf[codec][source] = BD_RATE_EXCEPTION_STRING
                    # BD rate computation failed for one of the codecs,
                    # so to be fair, ignore this source for final results
                    if source not in black_list_source_VMAF_BDRATE:
                        black_list_source_VMAF_BDRATE.append(source)

                # print('SSIM')
                try:
                    bd_rate_ssim = 100.0 * BDrateCalculator.CalcBDRate(
                        list(zip(get_rates(baseline_rate_quality_points), get_ssim(baseline_rate_quality_points))),
                        list(zip(get_rates(rate_quality_points), get_ssim(rate_quality_points))))
                    bdrates_ssim[codec][source] = bd_rate_ssim
                except AssertionError as e:
                    print('ssim {} {} {}: '.format(source, codec, sub_sampling) + str(e))
                    bdrates_ssim[codec][source] = BD_RATE_EXCEPTION_STRING
                    # BD rate computation failed for one of the codecs,
                    # so to be fair, ignore this source for final results
                    if source not in black_list_source_SSIM_BDRATE:
                        black_list_source_SSIM_BDRATE.append(source)

        print('{} black list VMAF BD RATE\n '.format(sub_sampling) + repr(black_list_source_VMAF_BDRATE))
        print('{} black list SSIM BD RATE\n '.format(sub_sampling) + repr(black_list_source_SSIM_BDRATE))
        results = dict()
        for codec in codecs:
            if codec == 'webp' and sub_sampling == '444':
                continue
            print('Codec {} subsampling {}, BD rates:'.format(codec, sub_sampling))
            mean_bd_vmaf, mean_bd_ssim, result = print_bd_rates(bdrates_vmaf, bdrates_ssim, codec, unique_sources,
                                                                black_list_source_VMAF_BDRATE,
                                                                black_list_source_SSIM_BDRATE)
            results[codec] = result
        print('\n\n------------------------------------------------')
        print('Results for subsampling {}'.format(sub_sampling))
        for codec in codecs:
            if codec == 'webp' and sub_sampling == '444':
                continue
            print(results[codec])
        print('------------------------------------------------')


if __name__ == '__main__':
    main(sys.argv[1:])

