from nowcasting.config import cfg
import os
from urllib import request
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

download_jobs =\
[[cfg.HKO_VALID_DATETIME_PATH,
  'https://www.dropbox.com/s/c372wb6ciygu73i/valid_datetime.pkl?dl=1'],
 [cfg.HKO_PD.ALL,
  'https://www.dropbox.com/s/38a2wf9pmkzef8q/hko7_all.pkl?dl=1'],
 [cfg.HKO_PD.ALL_09_14,
  'https://www.dropbox.com/s/n7vir3vbkadyrbp/hko7_all_09_14.pkl?dl=1'],
 [cfg.HKO_PD.ALL_15,
  'https://www.dropbox.com/s/q7cdp3g00b53fat/hko7_all_15.pkl?dl=1'],
 [cfg.HKO_PD.RAINY_TRAIN,
  'https://www.dropbox.com/s/wutmla45tn606cl/hko7_rainy_train.pkl?dl=1'],
 [cfg.HKO_PD.RAINY_VALID,
  'https://www.dropbox.com/s/uwumfiw9dbdg5x0/hko7_rainy_valid.pkl?dl=1'],
 [cfg.HKO_PD.RAINY_TEST,
  'https://www.dropbox.com/s/2kzt2me55hybfqw/hko7_rainy_test.pkl?dl=1']]

parser = argparse.ArgumentParser(description='Downloading the necessary data')
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help='Whether to overwrite the stored data files')
args = parser.parse_args()

for target_path, src_path in download_jobs:
    if not os.path.exists(target_path) or args.overwrite:
        print('Downloading from %s to %s...' % (src_path, target_path))
        data_file = request.urlopen(src_path)
        with open(target_path, 'wb') as output:
            output.write(data_file.read())
        print('Done!')
    else:
        print('Found %s' % target_path)
