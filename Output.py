class Output():
    def __init__(self, verbose):
        self.verbose=verbose
    def printc(self, string):
        if self.verbose:
            print(string)

