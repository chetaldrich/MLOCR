# progressBar.py
# -----------------
# Main file for the progress bar used for showing progress throughout the program.
#
# Chet Aldrich, Laura Biester

import sys

class ProgressBar:
    """
    A Class to create progress bars to show in the terminal during loops.
    """
    def __init__(self, width, loopLength, label=""):
        """
        Initialization of the Progress Bar.

        Keyword Arguments:
        width -- designates the width of the progress bar in characters
        loopLength -- designates the number of iterations in the loop
        label -- designates a label for the progress bar
        """
        self.barWidth = width
        self.loopLength = loopLength
        self.label = label

        sys.stdout.write("{0} [{1}] 0%".format(self.label, self.barWidth * " "))
        sys.stdout.flush()


    def update(self, iteration):
        """
        update() Updates the progress bar based on current index in the loop.

        Keyword Arguments:
        iteration -- give the iteration value, which will update the progress bar
                     when higher than the current value.
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
        shouldUpdate() determines when the bar should update.

        Keyword Arguments:
        iteration -- the current iteration of the loop
        """
        # if the bar is longer than the loop, always true!
        if self.barWidth > self.loopLength:
            return True

        # otherwise, check to see if we have reached a point where we are
        # approximately n/self.loopLength of the way through the loop
        return iteration % (self.loopLength / self.barWidth) == 0

    def changeLabel(self, newLabel):
        """
        changeLabel() updates the label on a progress bar.

        Keyword Arguments:
        newLabel -- the new label
        """
        self.label = newLabel

    def clear(self):
        """
        clear() eliminates the progess bar upon completion of a loop.
        """
        # overwrite all of the content on previously written line with spaces
        sys.stdout.write("\r")
        sys.stdout.write(" " * (len(self.label) + 7 + self.barWidth))
        # go to start of line
        sys.stdout.write("\r")
        sys.stdout.flush()
