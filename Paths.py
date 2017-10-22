import os


# Returns the absolute directory that this file is in.
# Useful for adding to relative paths when you put it in the project directory.
def this_directory():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return this_dir


def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
