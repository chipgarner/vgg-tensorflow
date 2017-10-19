import os


# Returns the absolute directory that this file is in.
def this_directory():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return this_dir