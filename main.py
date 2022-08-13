import argparse
from inFolder import enhance

def main():
    parser = argparse.ArgumentParser(description='Enhance images from a directory.')

    parser.add_argument(
        '-o', '--origin', required=True, type=str, help='PATH to origin')
    parser.add_argument(
        '-s', '--save', required=True, type=str, help='PATH to save')
    parser.add_argument(
        '-n', '--methodname', required=True, type=str, help='method of enhancement')
    args = parser.parse_args()
    

    pathOrigin = args.origin

    path2Save = args.save

    method = args.methodname

    enhance(pathOrigin, path2Save, method)


if __name__ == "__main__":
    main()