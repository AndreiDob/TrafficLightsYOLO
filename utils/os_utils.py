import os


def create_dir(directory):
    '''
    creates a directory if it doesn't exist
    '''
    if os.path.exists(directory) is False or os.path.isdir(directory) is False:
        print('Creating Folder', directory)
        os.makedirs(directory)
