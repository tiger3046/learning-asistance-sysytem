import os
import shutil


def allremove():
    os.makedirs('ディレクトリ名', exist_ok=True)

    target_dir = 'photofile'

    shutil.rmtree(target_dir)

    os.mkdir(target_dir)

    print(os.listdir('photofile/'))
    # []
