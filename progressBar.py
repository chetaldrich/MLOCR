import sys

class ProgressBar:
    """
    A Class to create progress bars to show in the terminal during loops
    """
    def __init__(self, width, loopLength, label=""):
        """
        Give progres bar a width, length of loop (to determine progress) and optional label
        """
        self.barWidth = width
        self.loopLength = loopLength
        self.label = label

        sys.stdout.write("{0} [{1}] 0%".format(self.label, self.barWidth * " "))
        sys.stdout.flush()


    def update(self, iteration):
        """
        updates the progress bar based on current index in the loop
        """
        # check if we are at a new point that can be handled by a bar
        # of current size
        if not (self.shouldUpdate(iteration)):
            return

        # create strings to show progress
        finished = int(iteration / (float(self.loopLength) / self.barWidth)) * "-"
        remaining = int(self.barWidth - (iteration / (float(self.loopLength) / self.barWidth))) * " "
        # get percent completed
        percent = int(float(iteration) / self.loopLength * 100)

        # re-write line
        sys.stdout.write("\r")
        sys.stdout.flush()

        # give progress
        sys.stdout.write("{0} [{1}{2}] {3}%".format(self.label, finished, remaining, percent))
        sys.stdout.flush()

    def shouldUpdate(self, iteration):
        """
        determines whether or not we need to update a progress bar based on the iteration
        """
        # if the bar is longer than the loop, always true!
        if self.barWidth > self.loopLength:
            return True

        # otherwise, check to see if we have reached a point where we are
        # approximately n/self.loopLength of the way through the loop
        return iteration % (self.loopLength / self.barWidth) == 0

    def changeLabel(self, newLabel):
        """
        updates the label on a progress bar
        """
        self.label = newLabel

    def clear(self):
        """
        provides a 'clean slate' after loop is finished
        """
        # overwrite all of the content on previously written line with spaces
        sys.stdout.write("\r")
        sys.stdout.write(" " * (len(self.label) + 7 + self.barWidth))
        # go to start of line
        sys.stdout.write("\r")
        sys.stdout.flush()
