import sys
from time import sleep

class ProgressBar:
    def __init__(self, width, loopLength, label=""):
        self.barWidth = width
        self.loopLength = loopLength
        self.label = label

        sys.stdout.write("{0} [{1}] 0%".format(self.label, self.barWidth * " "))
        sys.stdout.flush()


    def update(self, iteration):
        if not (self.shouldUpdate(iteration)):
            return

        finished = int(iteration / (float(self.loopLength) / self.barWidth)) * "-"
        remaining = int(self.barWidth - (iteration / (float(self.loopLength) / self.barWidth))) * " "
        percent = int(float(iteration) / self.loopLength * 100)

        sys.stdout.write("\r")
        sys.stdout.flush()

        sys.stdout.write("{0} [{1}{2}] {3}%".format(self.label, finished, remaining, percent))
        sys.stdout.flush()

    def shouldUpdate(self, iteration):
        if self.barWidth > self.loopLength:
            return True
        return iteration % (self.loopLength / self.barWidth) == 0

    def changeLabel(self, newLabel):
        self.label = newLabel

    def clear(self):
        sys.stdout.write("\r")
        sys.stdout.write(" " * (len(self.label) + 7 + self.barWidth))
        sys.stdout.write("\r")
        sys.stdout.flush()
