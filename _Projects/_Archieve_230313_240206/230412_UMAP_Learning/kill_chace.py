

#%%

import os


def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():


    root_folder = r'D:\ZR\Anaconda'
    i=0
    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                    i = i+1
                except Exception as e:
                    print("failed on %s", root)

    print(f'Total {i} cache folder killed.')

#%%
kill_numba_cache()