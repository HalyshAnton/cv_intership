import argparse
from data_preprocessor import preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image preprocessing')

    parser.add_argument('--path', type=str,
                        default='Scan_202465_4io-R77sgBI (1).pdf',
                        help='path to pdf file')

    parser.add_argument('--show', type=bool, default=False,
                        help='whether to show results')

    args = parser.parse_args()
    preprocessing(args.path, show=args.show)