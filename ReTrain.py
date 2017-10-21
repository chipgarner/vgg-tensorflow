import Pbfile
import Paths


class ReTrain:
    def __init__(self):
        directory = Paths.this_directory()
        self.graph = Pbfile.load_graph(directory + '/out/output_graph.pb')
