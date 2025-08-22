import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from sklearn import metrics as met
from matplotlib.colors import Normalize



colors = (['Greens', 'Reds', 'Blues', 'Oranges',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])

class FocalH:
    """
    Class to hold read-out geometry of FocalH so I can plot nice heatmaps.
    Construct cells as container of Polygon and leverage Path for fast lookup.
    """

    def __init__(self):

        # Geometry
        self.detector_width = 19.5
        self.detector_height = 19.5
        self.module_columns = 3
        self.module_rows = 3
        self.center_module_rows = 7
        self.center_module_cols = 7
        self.outer_module_rows = 5
        self.outer_module_cols = 5
        self.module_width = self.detector_width/self.module_columns
        self.module_height = self.detector_height/self.module_rows

        self.n_poly = self.center_module_rows**2 + (self.outer_module_rows**2)*8
    
        self.polygons = []
        self.patches = []
        self.values = np.zeros(self.n_poly)
        


        
        self.__make_module(self.outer_module_rows, self.module_width, (-self.module_width,self.module_width)) # 00
        self.__make_module(self.outer_module_rows, self.module_width, (0,self.module_width)) # 01
        self.__make_module(self.outer_module_rows, self.module_width, (self.module_width,self.module_width)) # 02
        self.__make_module(self.outer_module_rows, self.module_width, (-self.module_width,0)) # 10
        self.__make_module(self.center_module_rows, self.module_width) # 11
        self.__make_module(self.outer_module_rows, self.module_width, (self.module_width,0)) # 12
        self.__make_module(self.outer_module_rows, self.module_width, (-self.module_width,-self.module_width)) # 20
        self.__make_module(self.outer_module_rows, self.module_width, (0,-self.module_width)) # 21
        self.__make_module(self.outer_module_rows, self.module_width, (self.module_width,-self.module_width)) # 22
        

        

        
        self.paths = [polygon.get_path() for polygon in self.polygons]
    
    def __make_module(self, size, width, center=(0,0)):
        d = width / size
        offset = int(size/2)
        for r in reversed(range(size)):
            for c in range(size):
                x = (c-offset)*d + center[0]
                y = (r-offset)*d + center[1]
                self.polygons.append(Polygon([(x-d/2,y-d/2), (x-d/2,y+d/2), (x+d/2,y+d/2), (x+d/2,y-d/2)]))
    
    def squawk(self):
        print("FocalH read-out geometry.")

    def read_tree_entry(self, tree, entry):
    
        entries = tree.GetEntries()
    
        if entries < entry:
            print("Out of bounds")
            return
        
        tree.GetEntry(entry)

        npx = np.array(tree.x, dtype=float)
        npy = np.array(tree.y, dtype=float)
        npvals = np.array(tree.value, dtype=float)
        
        npfracs = np.array(tree.fractions, dtype=float)
        nplabels = np.array(tree.labels, dtype=int)
        nplabelidx = np.array(tree.label_indices, dtype=int)
            
        return (npx, npy, npvals, npfracs, nplabels, nplabelidx)


    def read_tree_entry_clustered(self, tree, entry):
    
        entries = tree.GetEntries()
    
        if entries < entry:
            print("Out of bounds")
            return
        
        tree.GetEntry(entry)

        npx = np.array(tree.x, dtype=float)
        npy = np.array(tree.y, dtype=float)
        npvals = np.array(tree.value, dtype=float)
        
        npfracs = np.array(tree.fractions, dtype=float)
        nplabels = np.array(tree.labels, dtype=int)
        nplabelidx = np.array(tree.label_indices, dtype=int)

        npclusters = np.array(tree.clusters, dtype=int)
        npclusteridx = np.array(tree.cluster_indices, dtype=int)
            
        return (npx, npy, npvals, npfracs, nplabels, nplabelidx, npclusters, npclusteridx)


    def heatmap(self, values, labels, ax=None):
        """
        Generic heatmap created from values and labels.
        Assumes the caller knows the order of cells.
        """

        if ax is None:
            fig, ax = plt.subplots()

        c = np.zeros(len(values) * 4).reshape(len(values), 4)
        unique = [int(l) for l in set(labels) if l != 0]
        print(unique)
        #norm = Normalize(vmin=0, vmax=4096)
        norm = Normalize(vmin=0, vmax=249)

        for l in set(labels):
            mask = labels == l
            if l == 0:
                c[mask] = [1,1,1,1]
            c[mask] = plt.colormaps[colors[l-1]](norm(values[mask]))

        for i in range(len(labels)):
            c[i] = norm(i)

        c[0] = norm(249)

        patches = PatchCollection(self.polygons, alpha=1)
        #patches.set_clim(0, 4096)
        patches.set_facecolor(c)

        ax.add_collection(patches)
        ax.set_xlim(-self.detector_width/2, self.detector_width/2)
        ax.set_ylim(-self.detector_height/2, self.detector_height/2)
        ax.set_aspect("equal")

        return ax

    
    def heatmap_clustered(self, ttree, entry, ax=None, colors=""):
        if ax is None:
            fig, ax = plt.subplots()

        x, y, values, fractions, labels, label_indices, clusters, cluster_indices = self.read_tree_entry_clustered(ttree, entry)

        # Cluster labels, but 0 untagged is included
        NCollections = len(set(clusters))
#        grouped = np.zeros(NCollections * len(values)).reshape(NCollections, -1)
        grouped_c = np.zeros(len(values) * 4).reshape(len(values), 4)
        norm = Normalize(vmin=0, vmax=4096)
        if colors == "":
            colors = (['Greens', 'Reds', 'Blues', 'Oranges',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])


        for icell, cl in enumerate(clusters):
            if cl == 0 or values[icell] == 0:
                grouped_c[self.search(x[icell], y[icell])] = [1,1,1,1]
            else:
                grouped_c[self.search(x[icell], y[icell])] = plt.colormaps[colors[cl-1]](norm(values[icell]))

        patches = PatchCollection(self.polygons, alpha=1)
        patches.set_clim(0, 4096)
        patches.set_facecolor(grouped_c)
                
        ax.add_collection(patches)
        ax.set_xlim(-self.detector_width/2, self.detector_width/2)
        ax.set_ylim(-self.detector_height/2, self.detector_height/2)
        ax.set_aspect("equal")

        for spine in ax.spines.values():
            spine.set_linestyle('--')  # Set the linestyle to dashed
            spine.set_linewidth(2)     # Optionally, set the linewidth
            spine.set_color("grey")     # Optionally, set the linewidth



    def heatmap_labels(self, ttree, entry, ax=None, colors=""):
        if ax is None:
            fig, ax = plt.subplots()

        x, y, values, fractions, labels, label_indices = self.read_tree_entry(ttree, entry)
        
        NCollections = len(set(labels))
        
        grouped = np.zeros(NCollections * len(values)).reshape(NCollections, -1) # needed?
        grouped_c = np.zeros(len(values) * 4).reshape(len(values), 4) # 4 for rgb

        norm = Normalize(vmin=0, vmax=4096)

        if colors == "":
            colors = (['Greens', 'Reds', 'Blues', 'Oranges',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])        
        
        for icell, e in enumerate(label_indices):
            try:
                e_next = label_indices[icell+1]
            except IndexError:
                e_next = len(labels)

            max_e = e
            maj = labels[e]
            for j in range(e, e_next):
                if fractions[j] > fractions[max_e]:
                    max_e = j
                    maj = labels[j]
                #grouped[e-j][self.search(x[icell], y[icell])] = fractions[j] * values[icell]
            if values[icell] == 0:
                temp_color = [1,1,1,1]
            else:
                temp_color = plt.colormaps[colors[maj-1]](norm(fractions[max_e] * values[icell])) # some modulo of max colors to be safe            
            grouped_c[self.search(x[icell], y[icell])] = temp_color

        patches = PatchCollection(self.polygons, alpha=1)
        patches.set_clim(0, 4096)
        patches.set_facecolor(grouped_c)
        ax.add_collection(patches)
        ax.set_xlim(-self.detector_width/2, self.detector_width/2)
        ax.set_ylim(-self.detector_height/2, self.detector_height/2)
        ax.set_aspect("equal")

    def get_fractions(self, ttree, entry):
        """
        Separate out the label fractions. This gives indices for the plot cells.
        """

        ttree.GetEntry(entry)

        x, y, values, fractions, labels, label_indices = self.read_tree_entry(ttree, entry)
        NCollections = len(set(labels))
        
        grouped = np.zeros(NCollections * len(values)).reshape(NCollections, -1)
        
        for icell, e in enumerate(label_indices):
            try:
                e_next = label_indices[icell+1]
            except IndexError:
                e_next = len(labels)

            for j in range(e, e_next):
                grouped[e-j][icell] = fractions[j]
        return grouped

    def major_label(self, ttree, entry):
        ttree.GetEntry(entry)
        npfracs = np.array(ttree.fractions, dtype=float)
        nplabels = np.array(ttree.labels, dtype=int)
        nplabelidx = np.array(ttree.label_indices, dtype=int)

        nplabels_thresh = np.zeros_like(nplabelidx)

        for i,e in enumerate(nplabelidx):
            try:
                e_next = nplabelidx[i+1]
            except IndexError:
                e_next = nplabelidx[-1]

            max_e = e
            for j in range(e, e_next):
                if npfracs[j] > npfracs[max_e]:
                    max_e = j
            nplabels_thresh[i] = nplabels[max_e]

        return nplabels_thresh

    
    def v_score(self, ttree, entry, repeat=False, non_cluster=False):
        ttree.GetEntry(entry)
        vals = np.array(ttree.value, dtype=int)
        nplabels = self.major_label(ttree, entry)
        npclusters = np.array(ttree.clusters, dtype=int)
        if non_cluster:
            npclusters = np.ones(len(npclusters))
        if repeat:
            nplabels = np.repeat(nplabels, vals)
            npclusters = np.repeat(npclusters, vals)
        return met.v_measure_score(nplabels, npclusters)


    def homogeneity_score(self, ttree, entry, repeat=False, non_cluster=False):
        ttree.GetEntry(entry)
        vals = np.array(ttree.value, dtype=int)
        nplabels = self.major_label(ttree, entry)
        npclusters = np.array(ttree.clusters, dtype=int)
        if non_cluster:
            npclusters = np.ones(len(npclusters))
        if repeat:
            nplabels = np.repeat(nplabels, vals)
            npclusters = np.repeat(npclusters, vals)
        return met.homogeneity_score(nplabels, npclusters)


    def completeness_score(self, ttree, entry, repeat=False, non_cluster=False):
        ttree.GetEntry(entry)
        vals = np.array(ttree.value, dtype=int)
        nplabels = self.major_label(ttree, entry)
        npclusters = np.array(ttree.clusters, dtype=int)
        if non_cluster:
            npclusters = np.ones(len(npclusters))
        if repeat:
            nplabels = np.repeat(nplabels, vals)
            npclusters = np.repeat(npclusters, vals)
        return met.completeness_score(nplabels, npclusters)


    def nmi(self, ttree, entry, non_cluster=False):
        ttree.GetEntry(entry)
        npclusters = np.array(ttree.clusters, dtype=int)
        if non_cluster:
            npclusters = np.ones(len(npclusters))
        return met.normalized_mutual_info_score(self.major_label(ttree, entry), npclusters)

    def ars(self, ttree, entry, non_cluster=False):
        ttree.GetEntry(entry)
        npclusters = np.array(ttree.clusters, dtype=int)
        if non_cluster:
            npclusters = np.ones(len(npclusters))
        return met.adjusted_rand_score(self.major_label(ttree, entry), npclusters)

        
    def center_of_masses(self, ttree, entry):
        """
        Compute center of masses for labels and clusters for an event.
        """
        pass

    def clusters_per_label(self, ttree, entry, non_cluster=False):
        """
        Compute number of clusters per particle for an event.
        """
        ttree.GetEntry(entry)
        npclusters = np.array(ttree.clusters, dtype=int)
        nplabels = np.array(ttree.labels, dtype=int)
        if non_cluster:
            npclusters = np.ones(len(npclusters))
        num_clusters = npclusters.max()
        num_labels = len(set(nplabels)) # Going to be incorrect if I e.g. remove cluster 2 after filtering
        #print(set(npclusters))
        return num_clusters / num_labels
    
    def fill(self, x, y, value):
        for i, path in enumerate(self.paths):
            if path.contains_point((x,y)):
                self.values[i] = value
                return
        print("Coordinates not within any polygon!")

    def search(self, x, y):
        for i, path in enumerate(self.paths):
            if path.contains_point((x,y)):
                return i
        print("Coordinates not within any polygon!")
        return -1

    def center_of_mass(self, values, dimension):
        """
        I want center-of-mass. May even be a good metric to evaluate based on. Match com as best and compare distance? Error on com?
        """
        com = (values * dimension).sum()
        if values.sum() == 0:
            pass
        else:
            com = com / values.sum()
        return com

    def labels_center_of_mass(self, ttree, entry):
        """
        Return the center of masses for x and y for each label.
        Can make a version which only takes dominant cells.
        """
        ttree.GetEntry(entry)
        npx = np.array(ttree.x, dtype=float)
        npy = np.array(ttree.y, dtype=float)
        npvals = np.array(ttree.value, dtype=float)
        
        fracs = self.get_fractions(ttree, entry)
        com = np.zeros(len(fracs) * 2).reshape(len(fracs), 2)
        for i in range(len(fracs)):
            com[i][0] = self.center_of_mass(fracs[i] * npvals, npx)
            com[i][1] = self.center_of_mass(fracs[i] * npvals, npy)
        return com

    def num_labels(self, ttree, entry):
        ttree.GetEntry(entry)
        return len(set(ttree.labels))

    def adc_sum(self, ttree, entry):
        ttree.GetEntry(entry)
        return np.array(ttree.value).sum()

    def cluster_sums(self, ttree, entry):
        ttree.GetEntry(entry)
        npvals = np.array(ttree.value)
        npclusters = np.array(ttree.clusters)
        unique_clusters = set(npclusters)
        sums = []
        for cl in unique_clusters:
            if cl==0:
                continue
            mask = npclusters == cl
            sums.append(npvals[mask].sum())
        return sums

    def get_energies(self, ttree, entry):
        ttree.GetEntry(entry)
        return np.array(ttree.energies)

    def num_saturated_cells(self, ttree, entry, sat_value):
        ttree.GetEntry(entry)
        npvals = np.array(ttree.value)
        mask = npvals >= sat_value
        return len(npvals[mask])


    def avg_com(self, com):
        if len(com) <= 1:
            return -1
        coms = []
        for i in range(len(com)):
            for j in range(i+1, len(com)):
                coms.append(np.sqrt( (com[j][0] - com[i][0])**2 + (com[j][1] - com[i][1])**2 ))
        return np.mean(coms)




def analyse_tree(ttree, N):
    f = FocalH()
    entries = ttree.GetEntries()
    if entries > N:
        entries = N
    V_SCORES = np.zeros(entries)
    NMI_SCORES = np.zeros(entries)
    ARS_SCORES = np.zeros(entries)
    CENTER_OF_MASSES = []
    CLUSTERS_PER_LABEL = np.zeros(entries)
    for i in range(entries):
        V_SCORES[i] = f.v_score(ttree, i)
        NMI_SCORES[i] = f.nmi(ttree, i)
        ARS_SCORES[i] = f.ars(ttree, i)
        CENTER_OF_MASSES.append(f.avg_com(f.labels_center_of_mass(ttree, i)))
        CLUSTERS_PER_LABEL[i] = f.clusters_per_label(ttree, i)
    return CENTER_OF_MASSES, V_SCORES, NMI_SCORES, ARS_SCORES, CLUSTERS_PER_LABEL



def plot_chunks(x, y, Nbins, ax=None, x_label = "", y_label = "", title=""):
    if ax is None:
        fig, ax = plt.subplots()
        
    bins = np.linspace(min(x), max(x), Nbins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mu_y = np.zeros(Nbins)
    sigma_y = np.zeros(Nbins)
    
    for i in range(Nbins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if True:#np.any(mask):
            mu_y[i] = np.mean(y[mask])
            sigma_y[i] = np.std(y[mask]) / np.sqrt(np.sum(mask))
    
    ax.errorbar(bin_centers, mu_y, yerr=sigma_y, marker='o', linestyle='-', capsize=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
