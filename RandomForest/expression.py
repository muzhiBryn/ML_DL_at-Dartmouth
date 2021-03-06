import csv, matplotlib.pyplot as plt, numpy as np

class ExpressionProfile:
    """A sample id, label (for supervised), and profile (list of expression values)."""
    def __init__(self, id, label, values):
        self.id = id
        self.label = label
        self.values = values

    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, feat):
        return self.values[feat]
        
    def __str__(self):
        return self.id + (':'+self.label if self.label is not None else '')

    def __repr__(self):
        return str(self)

    @staticmethod
    def load(filename, delimiter=','):
        """Read instances from the file, in delimited (default: comma-separated) format. 
        If the first column has a ':' in it, the name is the part before and the label is the part after; 
        else the name is the whole thing and the label is None.
        Args:
          filename: path to file
          delimeter: separates columns in file
        Returns:
          [ ExpressionProfile ], expression profiles constructed from the file
        """
        reader = csv.reader(open(filename,'r'), delimiter=delimiter)
        profiles = []
        for row in reader:
            if row[0].find(':')<0: (id,label) = (row[0],None)
            else: (id,label) = row[0].split(':')
            values = np.array([float(v) for v in row[1:]])
            profiles.append(ExpressionProfile(id, label, values))
        return profiles

def plot_profiles(eps, width=12, height=10, cbar_frac=0.02):
    """Plot the profiles.
    Uses a symmetric blue-red colormap (blue for negative, red for positive)
    Args:
      eps: [ ExpressionProfile ] to plot
    """
    cmap = plt.cm.RdBu_r
    cmap.set_bad('k')
    matrix = np.vstack([ep.values for ep in eps])
    mmax = max(np.nanmax(matrix), abs(np.nanmin(matrix)))
    plt.figure(figsize=(width,height))
    plt.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-mmax, vmax=mmax)
    plt.xlim(0, len(eps[0]))
    plt.ylim(0, len(eps))
    plt.xlabel('genes')
    plt.ylabel('samples')
    plt.xticks([])
    plt.yticks(0.5+np.arange(len(eps)), [str(ep) for ep in eps])
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.colorbar(fraction=cbar_frac)
