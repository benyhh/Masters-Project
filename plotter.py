import matplotlib.pyplot as plt
from time import sleep 
import os

class Plotter():
    def __init__(self, name, path):
        
        self.FULL_PATH_LOSSCURVE = os.path.join(path, f'losscurve_{name}.png')
        self.name = name
        self.bestvalmeasure=None
        self.max_val = 0
        with plt.ion():
            self.fig, self.ax = plt.subplots()
    
        self.fig.show()

        self.fig.suptitle("Performance curves")
        sleep(0.1)
        self.ax.set_ylabel('Loss')
        self.ax.set_xlabel('epoch [#]')

        # train_line = self.ax.plot([],[],color='blue', label='Train', marker='.', linestyle="")
        # val_line   = self.ax.plot([], [], color='red', label='Validation', marker='.', linestyle="")

        train_line = self.ax.plot([],[],color='blue', label='Train', marker='.', linestyle="")
        val_line   = self.ax.plot([], [], color='red', label='Validation', marker='.', linestyle="")

        self.ax.legend()
        self.ax.set_axisbelow(True)

        sleep(0.1)
        plt.tight_layout()
        return


    def update(self, current_epoch, loss, mode):
        if loss > self.max_val:
            self.max_val = loss
        self.ax.scatter(current_epoch, loss, c='b')
        self.fig.canvas.draw()
        sleep(0.1)
        self.ax.set_ylim([0, self.max_val])

        self.save()

        return

    def update_withval(self, current_epoch, loss, valmeasure, mode):
        if loss > self.max_val:
            self.max_val = loss
        if valmeasure > self.max_val:
            self.max_val = valmeasure
        if (5 * loss < self.max_val) & (loss != 0):
            self.max_val = 5 * loss

        if self.bestvalmeasure is None:
          self.bestvalmeasure = valmeasure
        elif self.bestvalmeasure > valmeasure :
          self.bestvalmeasure = valmeasure
        # print('\n\n\ncurrent best val measure and current valmeasure ',self.bestvalmeasure, valmeasure)
    
        self.ax.scatter(current_epoch, loss      , c='b')
        self.ax.scatter(current_epoch, valmeasure, c='r')
        self.fig.suptitle(f"Performance curves | Best val: {self.bestvalmeasure:.10}")
        self.fig.canvas.draw()
        self.ax.set_ylim([0, self.max_val])

        sleep(0.1)

        self.save()

        return

    def save(self):

        self.fig.savefig(self.FULL_PATH_LOSSCURVE)
        return