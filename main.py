# This is a sample Python script.
from time import sleep

import matplotlib.pyplot as plt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):

    # You probably won't need this if you're embedding things in a tkinter plot...
    plt.ion()

    fig = plt.figure()
    sub_plt = fig.add_subplot(111)
    line1,  = sub_plt.plot([1,2,3])  # Returns a tuple of line objects, thus the comma
    fig.canvas.draw()
    fig.canvas.flush_events()
    sleep(1)
    line1,  = sub_plt.plot([3,2,1])  # Returns a tuple of line objects, thus the comma
    fig.canvas.draw()
    fig.canvas.flush_events()
    sleep(10)

    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
