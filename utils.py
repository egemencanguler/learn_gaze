def get_files(dir):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(dir) if isfile(join(dir, f)) and not f.startswith(".")]