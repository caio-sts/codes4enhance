import argparse
from time import time
from inFolder import enhance, geometricImageAugmentation, calculateImageMetrics

def main():
    parser = argparse.ArgumentParser(description='Enhance images from a directory.')

    parser.add_argument(
        '-o', '--origin', required=True, type=str, help='PATH to origin')
    parser.add_argument(
        '-s', '--save', required=True, type=str, help='PATH to save')
    parser.add_argument(
        '-n', '--methodname', nargs="?", type=str, help='method of enhancement')
    parser.add_argument(
        '-g', '--geometrics', nargs="?", type=bool, help='geometric transformations for data augmentation', default= False)
    parser.add_argument(
        '-x', '--numberGenerat', nargs="?", type=int, help='number of transformations for data augmentation')
    parser.add_argument(
        '-sh', '--shear', nargs="?", type=int, help='shear angle for data augmentation')
    parser.add_argument(
        '-r', '--rotation', nargs="?", type=int, help='rotation angle for data augmentation')
    parser.add_argument(
        '-tx', '--translatx', nargs="?", type=int, help='scale factor for horizontal translation for data augmentation')
    parser.add_argument(
        '-ty', '--translaty', nargs="?", type=int, help='scale factor for vertical translation for data augmentation')
    parser.add_argument(
        '-cm', '--calculateMetrics', nargs="?", type=bool, help='calculate metrics comparing images in an origin folder and generated images in destiny folder')

    args = parser.parse_args()


    if args.methodname != None:
        enhance(args.origin, args.save, args.methodname)
    
    elif args.geometrics == True:
        geometricImageAugmentation(args.origin, args.save, args.numberGenerat, 
                                    args.shear, args.rotation, args.translatx, args.translaty)

    elif args.calculateMetrics == True:
        calculateImageMetrics(args.origin, args.save)


if __name__ == "__main__":
    startTime = time()
    main()
    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")
