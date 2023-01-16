import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # This avoids crashes on the math cluster from using to many resources
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import time
import csv
from celluloid import Camera
from matplotlib.collections import PatchCollection
from scipy import spatial, ndimage
import copy
import networkx as nx
matplotlib.use("TkAgg")

class experiment:
    """Holds tracks of the experiment, as well as any calculated observables.
    :name Name of the experiment
    :tracks Array of numpy arrays containing the times and coordinates of each particle
    :tmax number of timesteps
    :x_max Biggest x coordinate found in a track
    :y_max Biggest y coordinate found in a track 
    :distance_travlled Array with one entry for every track, contains the total distance a 
        particle has travelled on the track
    :tortuosity Array with one entry for every track, contains the total normalised
        difference between distance_travelled and the total displacement
    :rng Random number generator used to create histogram normalisaiton by generating particles with 
        uniform density
    :uniform_histogram Stores histogram of radial density of uniform particles. Used
        for normalisation in plot_radial_density
    """
    def __init__(self,name,seed=None):
        self.periodic = False #The simulation as periodic, the experimental data is not
        self.name = name
        self.timeStep = 1 #time passed between each step of the track
        self.tracks = [] #tracks[particle][step][t,x,y]
        self.x_max = 0 
        self.y_max = 0
        self.max_distance = 0 #Maximum distance between particles based on x_max, y_max
        self.reference_histogram = np.array([None])
        self.color = [] #color[particle][step]['color']
        self.seed=seed
        self._measurementTimes=None
        if seed==None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed)

    def read_xml(self,fileName, min_length):
        """Reads the xml file in fileName and saves the tracks in it into fileName
        :fileName Path to the file
        :min_length All tracks shorter than that are excluded
        """
        tic = time.perf_counter()

        #Read xml file into beautiful soup:
        with open(fileName, 'r') as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")

        self.periodic = False #This .xml file only exists for experiments, which are not periodic

        #keep track of the biggest x and y values we find

        #Iterate through all tracks in the file
        for track in Bs_data.find_all('particle'):
            self.tracks.append([])
            #create numpy arrays to contain the track
            # nSpots = int(track['nSpots'])
            # if nSpots >=min_length:# filter short tracks
            #     self.tracks.append(np.zeros(shape=(nSpots,3)))
            #     n=0
                #Iterate through each spot detected for the particle
            nSpots = 0
            for spot in track.find_all('detection'):
                nSpots +=1
                self.tracks[-1].append([int(spot['t']),float(spot['x']),float(spot['y'])])
                # self.tracks[-1][n][0]=int(spot['t'])
                # self.tracks[-1][n][1]=float(spot['x'])
                # self.tracks[-1][n][2]=float(spot['y'])
                self.x_max = max(self.x_max,float(spot['x']))
                self.y_max = max(self.y_max,float(spot['y']))
        
        self.set_max_distance()
        toc = time.perf_counter()

        print(f"Read xml file in {toc - tic:f} seconds")
        
    def read_csv(self, fileName):
        """Reads the simulation data from a .csv file named fileName and saves the data in experiment.
        """
        tic = time.perf_counter()
        with open(fileName, mode="r") as file:
            csvFile = csv.reader(file)
            lineCount = 0
            for lines in csvFile: #Go through every timestep
                nParticles = int((len(lines)-1)/3)
                if (lineCount==0): #Initialise the tracks
                    self.tracks = [[] for i in range(nParticles)]
                    self.color =  [[] for i in range(nParticles)]
                t = float(lines[0])
                for particleIdx in range(nParticles):
                    self.color[particleIdx].append(lines[1+3*particleIdx])
                    self.tracks[particleIdx].append([t,
                                                    float(lines[1+3*particleIdx+1]), #x value
                                                    float(lines[1+3*particleIdx+2])])#y value
                    
                lineCount +=1
        toc = time.perf_counter()
        self.periodic = True #This .csv file only exists for simulations, which are always periodic
        print(f"Read .csv track file in {toc - tic:f} seconds")

    def read_parameter_file(self,file):
        params = {}
        """Reads the parameter .txt files used to initialise the C simulation
        """
        with open(file, 'r') as reader:
            for line in reader.readlines():
                #if line=="\n": break  #Stop when encountering an empty line
                idx1 = line.find(" ")
                idx2 = line.find(": ")
                if(idx2==-1): break #we reached the end of the parameter list
                name = line[:idx1]
                value = line[idx2+2:]
                params[name] = float(value)
                #print(name+" = %f"%params[name])
        self.timeStep = float(params["measurementInterval"])
        self.x_max = float(params["Length"])
        self.y_max = float(params["Length"])
        self.set_max_distance()
        return params

    def set_max_distance(self):
        if(self.periodic==False):
            self.max_distance = np.sqrt(self.x_max**2+self.y_max**2)*1.01 #Some wiggle room in case a point is exactly at max distane
        elif(self.periodic==True):
            self.max_distance = np.sqrt((self.x_max**2+self.y_max**2)/4)*1.01

    def find_t_max(self):
        tMax = 0
        for track in self.tracks:
            tMax = max(tMax, track[-1][0])
        return tMax

    def distance(self, r):
        """Calculates the length of the tangent vector r taking periodicity into account
        """
        if self.periodic == False:
            return np.linalg.norm(np.array(r))
        elif self.periodic == True:
            r[0] = min(abs(r[0]),abs(abs(r[0])-self.x_max)) #abs(r[0]) % self.x_max
            r[1] = min(abs(r[1]),abs(abs(r[1])-self.y_max)) #abs(r[1]) % self.y_max
            return np.linalg.norm(np.array(r))
        else:
            assert(False), "Invalid value for periodicity"  

    def MSD(self):
        """Calculates the mean squared displacement. Accounts for periodic boundary conditions in the simulation.
        Assumes displacement between steps doesn't exceed half the box.
        """
        MSD = np.zeros(len(self.tracks[0]))
        time = np.zeros(len(self.tracks[0]))
        for track in self.tracks:
            origin = np.array(track[0][1:]) #keeps track of where particle started
            offset = np.zeros(2) #keeps track of how many times the periodic boundary was crossed
            oldPosition = origin
            for tIdx in range(1,len(track)):
                newPosition = np.array(track[tIdx][1:])
                displacement = newPosition-oldPosition
                #check if particle passed boundary
                if(displacement[0]>self.x_max/2): #crossed lower x boundary
                    offset[0]-= self.x_max
                elif(displacement[0] < -self.x_max/2): #crossed upper x boundary
                    offset[0] += self.x_max       
                if(displacement[1]>self.y_max/2): #crossed lower y boundary
                    offset[1]-= self.y_max
                elif(displacement[1] < -self.y_max/2): #crossed upper y boundary
                    offset[1] += self.y_max    

                #Update MSD
                MSD[tIdx] += np.linalg.norm(newPosition+offset-origin)**2
                time[tIdx] = track[tIdx][0]

                oldPosition = newPosition

        return MSD/len(self.tracks), time

    def TAMSD(self,delta_t,min_length):
        """Returns the average TAMSD as well as all individual TAMSDs for all tracks in the experiment
         whose length exceed min_length. WARNING: Not suited for periodic boundary conditions (doesn't undo wraparound)
        :delta_t time interval (in timesteps, not minutes) for which the TAMASD will be computed
        :min_length minimum number of timesteps a track must have to be used for the average
        """
        assert delta_t < min_length, "The minimum track length must exceed the time average interval"
        nTracks = 0 # counts how many tracks are included/ have the minimum length
        average = 0
        individual_TAMSD = [] 
        for track in self.tracks:
            track_duration = len(track)
            total = 0
            if track_duration >= min_length:
                nTracks += 1
                #Calculate TAMSD for the track
                for idx in range(track_duration-delta_t):
                    distance = track[idx][1:]-track[idx+delta_t][1:]
                    total += np.dot(distance,distance)
                TAMSD= total/(track_duration-delta_t)
                individual_TAMSD.append(TAMSD)
                average += TAMSD
        assert nTracks > 0, 'None of the tracks have the minimum length'
        average = average/nTracks

        return average, individual_TAMSD

    def plot_TAMSD(self, tmax, min_length,axes):
        """Calculates the TAMSD as a function of the time intervall delta_t for all tracks 
        in the experiment that are longer than min_length.
        :tmax sets the biggest delta_t that is considered (in timesteps, not minutes). Has to 
            be an integer
        :min_length is used to filter all tracks that are shorter 
        :axes is the pyplot object into which the graph is drawn
        """
        delta_t_vec = np.linspace(1, tmax, tmax, dtype = int)
        avr_TAMSD = np.zeros(tmax)
        individual_TAMSD = []
        for delta_t in delta_t_vec:
            avr_TAMSD[delta_t-1], TAMSDs = self.TAMSD(delta_t,min_length)
            individual_TAMSD.append(TAMSDs)

        #Plot individual tracks
        individual_TAMSD = np.transpose(np.array(individual_TAMSD))
        for vec in individual_TAMSD:
            axes.loglog(delta_t_vec,vec,linewidth=0.1)    
        axes.set_xlabel(r"$\Delta$ t")
        axes.set_ylabel("TAMSD")
        axes.loglog(delta_t_vec,avr_TAMSD,label='Average', linewidth=2)

        #Plot dashed lines for comparison
        
        #Ballistic motion, starting on the left
        x0 = avr_TAMSD[0]
        x1 = avr_TAMSD[0]*(tmax)**2 
        axes.loglog([delta_t_vec[0],delta_t_vec[-1]],[x0,x1],color='black', 
                    linestyle='dashed', label='Ballistic motion')

        #Normal diffusive motion, 
        y0 = avr_TAMSD[0]
        y1 = avr_TAMSD[0]*(tmax) 
        axes.loglog([delta_t_vec[0],delta_t_vec[-1]],[y0,y1],color='blue', 
                    linestyle='dashed', label='Diffusive motion')

    def calculate_tortuosity_dun_method(self, delta_t, min_length):
        """calculates distance travelled and 
        tortuosity = (distance_travelled-displacement)/distance_travlled for every track and
        saves the values in two arrays, whose indices correspond to those of the tracks
        :delta_t Size of time interval for averaging (Dun method)
        :min_length Minimum size the track must have
        """
        assert delta_t < min_length, "Can't average unless delta_t<min_length"
        assert (delta_t/self.timeStep==int(delta_t/self.timeStep)), "delta_a has to be a mutiple of the time steps" 
        nTracks = len(self.tracks)
        self.distance_travelled = np.zeros(nTracks)
        self.tortuosity = np.zeros(nTracks)
        n=0 #track index
        for track in self.tracks:
            track_duration = int(len(track)*self.timeStep) #almost correct, some of the measured tracks have gaps in them
            if track_duration >= min_length:
                tortuosity = 0
                n_intervals = int(track_duration//delta_t)
                delta_idx = int(delta_t/self.timeStep)
                for intervalIdx in range(n_intervals): 
                    #Calculate Tortuosity in every interval
                    straight_line_distance = self.distance(np.array(track[intervalIdx*delta_idx][1:])-np.array(track[(intervalIdx+1)*delta_idx-1][1:]))
                    displacement = 0
                    for spot_idx in range(intervalIdx*delta_idx,(intervalIdx+1)*delta_idx-1):
                        displacement += self.distance(np.array(track[spot_idx+1][1:])-np.array(track[spot_idx][1:]))
                    tortuosity += (displacement-straight_line_distance)/displacement
                tortuosity = tortuosity/n_intervals

                self.distance_travelled[n]=displacement
                self.tortuosity[n] = tortuosity
                n+=1
        assert(n>0),"No tracks fulfill the requirements"
        #Extract non-zero entries
        self.distance_travelled = self.distance_travelled[self.tortuosity!=0]
        self.tortuosity = self.tortuosity[self.tortuosity!=0]

    def calculate_tortuosity(self, min_length): #Old version without dun method
        """calculates distance travelled and 
        tortuosity = (distance_travelled-displacement)/distance_travlled for every track and
        saves the values in two arrays, whose indices correspond to those of the tracks
        """
        nTracks = len(self.tracks)
        self.distance_travelled = np.zeros(nTracks)
        self.tortuosity = np.zeros(nTracks)
        n=0 #track index
        for track in self.tracks:
            if len(track)>= min_length:
                displacment = self.distance(np.array(track[0][1:])-np.array(track[-1][1:]))
                distance_travelled = 0
                for idx in range(len(track)-1):
                    distance_travelled += self.distance(np.array(track[idx+1][1:])-np.array(track[idx][1:]))
                tortuosity = (distance_travelled-displacment)/distance_travelled

                self.distance_travelled[n]=distance_travelled
                self.tortuosity[n] = tortuosity
                n+=1
        assert(n>0),"No tracks fulfill the requirements"
        #Extract non-zero entries
        self.distance_travelled = self.distance_travelled[self.tortuosity!=0]
        self.tortuosity = self.tortuosity[self.tortuosity!=0]

    def plot_tortuosity(self,axes, min_length, label, color, n_bins=None, method='Normal', delta_t=None):
        """Plots tortuosity either by considering the entire track at once, or by splitting 
        the track into intervals of length delta_t and averaging over those (Dun method).
        """
        if method=='Normal':
            self.calculate_tortuosity(min_length)
        elif method == 'Dun':
            self.calculate_tortuosity_dun_method(delta_t, min_length)
        else:
            raise Exception("Unknown method for computing tortuosity")
        #axes.scatter(self.distance_travelled,self.tortuosity, s=0.4)
        if n_bins == None:
            axes.hist(self.tortuosity, density=True, alpha=0.5, label=label, color=color)
        else: 
            axes.hist(self.tortuosity, density=True, alpha=0.5, label=label, color=color, bins=n_bins)
        axes.set_xlabel(r'Tortuosity = (L-I)/L')
        axes.set_ylabel(r'Probability Density')
        axes.legend()

    def generate_reference_histogram(self, n_particles, n_bins):
        '''Generates a radial density histogram for uniform distributed particles in a box of 
        size self.x_max self.y_max. This serves as a normalisation for the measured density.
        :n_particles Number of particles that are generated
        :n_bins Number of bins to be used. The bins are evenly distributed between 0 and max_distance
        '''
        #Check if we already created the distogram with this number of particles
        if (all(self.reference_histogram==None ) or len(self.reference_histogram)!= n_bins):
            #Generate points
            x_values = self.rng.random((n_particles))*self.x_max
            y_values = self.rng.random((n_particles))*self.y_max
            points = np.transpose(np.array([x_values,y_values]))

            if(self.periodic==False): #Faster Computation for non-periodic data
                distances = spatial.distance.pdist(points)
            elif(self.periodic==True): #Manual compuatation for periodic data
                distances = np.zeros(int(n_particles*(n_particles-1)/2))
                cnt = 0
                for i in range(n_particles):
                    for j in range(i+1,n_particles):
                        distances[cnt]=self.distance(points[i]-points[j])
                        cnt+=1

            self.reference_histogram = ndimage.histogram(distances, 0, self.max_distance,n_bins)

        return self.reference_histogram

    def plot_radial_density(self,axes, t ,n_bins, label, color, no_plot=False, cutoff_percentange = 10, 
                            n_reference_points = 10000):
        """Goes through all tracks to find spots in the time slice t and finds all distances 
        between them. The distances are assigned to the n_bin histogram bins.
        :axes Histogram will be plotted here
        :t Time at which the correlation function will taken
        :n_bins number of bins in the histogram
        :no_plot Does not plot on axes
        :cutoff_percentage The percentage of bins that will be cutoff at the end.
            The tail end 
        :n_reference_points Number of points used for the reference histogramm. High values 
            tax performance, but smoothen the reference density
        """
        #Prepare histogram
        bins = np.linspace(0, self.max_distance, n_bins+1)[:-1] #left edge of every bin
        histogram = np.zeros(n_bins)
        n_points = 0

        #Search all tracks for points
        points = []
        for track in self.tracks:
            tIdx, = np.where(np.array(track)[:,0]==t)
            
            if len(tIdx)>0: #Does the tracked particle exist at t?
                tIdx = tIdx[0] #Want first (and only) occurence
                new_point = track[tIdx][1:]
                n_points += 1
                #Compute all distances to the new point and place them in the histogram
                for old_point in points: 
                    distance = self.distance(np.array(new_point)-np.array(old_point))
                    bin_idx = int(np.floor(distance/self.max_distance*n_bins))
                    histogram[bin_idx]+=1
                points.append(new_point)
        assert len(points)>0, 'No points found at this time.'

        #Normalise the density
        n_distances = (n_points-1)*(n_points)/2
        n_reference_distances = (n_reference_points-1)*(n_reference_points)/2
        

        tic = time.perf_counter()
        self.reference_histogram = self.generate_reference_histogram(n_reference_points,n_bins)
        toc = time.perf_counter()
        print(f"Reference Histogram with {n_reference_points:.0f} particles was computed in {toc - tic:f} seconds")
            
        #Multiply by the ratio of the numbers of distances between points in each histogram
        reference_density = (n_distances/n_reference_distances)*self.reference_histogram
        cutoff = int(np.floor(n_bins*cutoff_percentange/100)) #Cutoff tail end of the histogarm to avoid dividing by 0
        radial_density = np.divide(histogram[:-cutoff],reference_density[:-cutoff])
        # radial_density = histogram[:-cutoff]

        #Plot Histogram
        if no_plot==False:
            bin_midpoints = bins+(bins[1]-bins[0])/2
            axes.plot(bin_midpoints[:-cutoff], radial_density, label=label, color = color)
            axes.set_xlabel(r'Distance')
            axes.set_ylabel(r'Radial distribution function')
            axes.legend()
        
        return histogram, bins, points

    def get_measurementTimes(self):
        if self._measurementTimes==None:
            self._measurementTimes=[self.tracks[0][idx][0] for idx in range(len(self.tracks[0]))]
        return self._measurementTimes

    measurementTimes = property(get_measurementTimes)

##################################### End of experiment class ############################################

def plot_mixed_particle_radial_density(axes, experiments, t ,n_bins, n_reference_points, no_plot=False, cutoff_percentage=0):
    """Calculates a histogram of the the distances between the points in the two experiments, i.e. cross radial distribution 
    function. Then plots them to axes.
    :axes Figure on which the plot is drawn
    :experiments List containing two experiment structs
    :t Time slice at which the radial distribution function will be conputed
    :n_bins Number of bins in the histogram
    :n_reference_points Number of points used in the computation of the reference histogram used to normalise the density. 
        Comutatiois skipped if such a histogram is already safed in experiment[0]
    :no_plot If true no plot is drawn
    :cutoff_percentage Percentage of the histogram that won't be shown in the plot, 
        useful when the tail end is high fluctuation
    """
    #Barricades
    assert(len(experiments)==2), "Expect exactly two experiments"
    assert(experiments[0].periodic == experiments[1].periodic), "Mixing periodic and non-periodic data"

    max_distance = max(experiments[0].max_distance,experiments[1].max_distance)
    bins = np.linspace(0, max_distance, n_bins+1)[:-1] #left edge of every bin
    histogram = np.zeros(n_bins)
    n_points = 0
    
    position_array = [] 
    for exp in experiments:
        points = []
        for track in exp.tracks:
            tIdx, = np.where(np.array(track)[:,0]==t)
            
            if len(tIdx)>0: #Does the tracked particle exist at t?
                tIdx = tIdx[0] #Only want first (and only) occurence
                points.append(track[tIdx][1:])
        position_array.append(points) #has spots from both experiments
    points1 = np.array(position_array[0])
    points2 = np.array(position_array[1])


    if(experiments[0].periodic==False): #Faster Computation for non-periodic data
        distances = spatial.distance.cdist(points1,points2)
    elif(experiments[0].periodic==True): #Manual compuatation for periodic data
        distances = np.zeros(len(points1)*len(points2))
        cnt = 0
        for pointA in points1:
            for pointB in points2:
                distances[cnt]=experiments[0].distance(pointA-pointB)
                cnt+=1
  
    histogram = ndimage.histogram(distances, 0, max_distance,n_bins)

    reference_histogram = experiments[0].generate_reference_histogram(n_reference_points,n_bins)
    #Multiply by the ratio of the numbers of links between points in each histogram
    n_distances = len(points1)*len(points2)
    n_reference_distances = n_reference_points*(n_reference_points-1)/2
    reference_histogram = (n_distances/n_reference_distances)*reference_histogram
    cutoff = int(np.floor(n_bins*cutoff_percentage/100)) #Cutoff tail end of the histogarm to avoid dividing by 0
    radial_density = np.divide(histogram[:-cutoff],reference_histogram[:-cutoff])

    #Plot Histogram
    if no_plot==False:
        bin_midpoints = bins+(bins[1]/2)
        axes.plot(bin_midpoints[:-cutoff], radial_density, label= 'Red-green RDF')
        axes.legend()
        axes.set_xlabel(r'Distance')
        axes.set_ylabel(r'Radial distribution function')

def load_simulation_data(PATH, trackIdx=""):
    #Load the data
    green = experiment(PATH)
    green.read_csv(f"{PATH}_tracks{trackIdx}.csv")
    params = green.read_parameter_file(PATH)
    nGreenParticles = int(params["nGreenParticles"])
    nRedParticles = int(params["nRedParticles"])


    #Don't exclude any tracks
    min_length = 0 #Tracks shorter than that are excluded

    #split up red and green particles (the simulation has green particles first and red particles second in its list)
    red = copy.deepcopy(green)
    green.tracks = green.tracks[:nGreenParticles]
    green.color = green.color[:nGreenParticles]
    red.tracks = red.tracks[nGreenParticles:nGreenParticles+nRedParticles]
    red.color = red.color[nGreenParticles:nGreenParticles+nRedParticles]

    return green, red, params

def get_positions_and_particle_types(experiments, time):
    """Extracts positions and particle types from experiment. 
    0 is red, 1 is green and greenPlus"""
    points=[]
    colorIdx=[]
    for expIdx, exp in enumerate(experiments):
            for pIdx, track in enumerate(exp.tracks):
                    tIndices, = np.where(np.array(track)[:,0]==time)

                    if len(tIndices)>0: #Does the tracked particle exist at t?
                            tIdx = tIndices[0] #first occurence
                            colorIdx.append(0 if exp.color[pIdx][tIdx]==" red" else 1)
                            points.append(np.array(track[tIdx][1:]))
    return points, colorIdx

def get_distance_based_neighbourhood_graph(points, color, experiment, neighbourDistance):
    # Creates graph whose connected components are the clusters, defined by two particles being within neighbourhood distance
    #Initialise the graph
    G =  nx.Graph()
    #Add nodes
    for nodeIdx in range(len(points)):
        G.add_node(nodeIdx)
    #Add edges
    for pIdx1 in range(len(points)):
        for pIdx2 in range(pIdx1):
            dist = experiment.distance((points[pIdx1]-points[pIdx2]))
            if(dist < neighbourDistance):
                if(color[pIdx1]==color[pIdx2]):
                    G.add_edge(pIdx1, pIdx2)
                    G.add_edge(pIdx2, pIdx1)
    return G

def get_voronoi_based_neighbourhood_graph(points, color, experiment):
    """
    Creates graph whose connected components are the clusters, defined by two particles being within neighbourhood distance
    """
    # Create Voronoi tesselation
    embeddedCoordinates = [[point[0], point[1], 0.5] for point in points]
    tess = Container(embeddedCoordinates, limits=(experiment.x_max, experiment.y_max,1), periodic=(True, True, False))    
    #Initialise the graph
    G =  nx.Graph()
    #Add nodes
    for nodeIdx in range(len(points)):
        G.add_node(nodeIdx)
    for idx, point in enumerate(tess):
        for neighbourIdx in point.neighbors():
            if(neighbourIdx>=0):
                if(color[idx]==color[neighbourIdx]):
                    G.add_edge(idx, neighbourIdx)
    return G

def calculate_mixing_index(experiments, t, neighbourDistance):
    """Calculates the mixing index of the two cell types. 
    :experiments List containing two experiments with the tracks
    :t Time slice at which the mixing index function will be conputed
    :neighbourDistance How far cells can be apart and still count as 
    neighbours. Giving None results in using Voronoi neighbourhood.
    """
    #Barricades
    assert(len(experiments)==2), "Expect exactly two experiments"
    assert(experiments[0].periodic == experiments[1].periodic), "Mixing periodic and non-periodic data"

    points, colorIdx = get_positions_and_particle_types(experiments, t)
    
    # Get colorblind neighbourhood graph
    if neighbourDistance==None:
        G = get_voronoi_based_neighbourhood_graph(points, np.zeros(len(points)), experiments[0])
    else:
        G = get_distance_based_neighbourhood_graph(points, np.zeros(len(points)), experiments[0], neighbourDistance)

    # Find how many same-type and cross-type neighbours each particle has
    sameTypeCnt = np.zeros(len(points))        
    crossTypeCnt = np.zeros(len(points))       

    for idx1 in range(len(points)):
        for idx2 in range(idx1):
            if G.has_edge(idx1, idx2): #Are they neighbours
                if colorIdx[idx1]==colorIdx[idx2]: #Same type?
                    sameTypeCnt[idx1] += 1
                    sameTypeCnt[idx2] += 1
                elif colorIdx[idx1] != colorIdx[idx2]: # Cross type
                    crossTypeCnt[idx1] += 1
                    crossTypeCnt[idx2] += 1

    #Discard 0 values and calculate mixing index for each particle (For Voronoi there should be no 0 values)
    neighbourCnt = sameTypeCnt+crossTypeCnt
    sameTypeCnt = np.array([sameTypeCnt[idx] for idx in range(len(neighbourCnt)) if neighbourCnt[idx]>0])
    crossTypeCnt = np.array([crossTypeCnt[idx] for idx in range(len(neighbourCnt)) if neighbourCnt[idx]>0])
    neighbourCnt = np.array([neighbourCnt[idx] for idx in range(len(neighbourCnt)) if neighbourCnt[idx]>0])
    colorIdx = [colorIdx[idx] for idx in range(len(neighbourCnt)) if neighbourCnt[idx]>0]

    mixing = (sameTypeCnt)/(neighbourCnt)
    mixingGreen = np.array([mixing[idx] for idx in range(len(mixing)) if colorIdx[idx]==1])
    mixingRed = np.array([mixing[idx] for idx in range(len(mixing)) if colorIdx[idx]==0])

    #Average
    mixingIdx = np.sum(mixing)/len(mixing)
    mixingIdxGreen = np.sum(mixingGreen)/len(mixingGreen)
    mixingIdxRed = np.sum(mixingRed)/len(mixingRed)

    return mixingIdx, mixingIdxGreen, mixingIdxRed

def write_avr_mixing_index(parameters, mixingIdx, FILE_PATH):
    """Appends the mixing index with it's parameters to FILE_PATH (creates new file if none is there).
    OUTDATED: Use write_mixing_index_to_file instead, it considers multiple realisations and time points.
    """
    file = open(FILE_PATH, "a") #append mode
    for name in parameters.keys():
        file.write(name+", %f, "%parameters[name])
    file.write("mixingIdx, %f \n"%mixingIdx)
    file.close()

def read_avr_mixing_index(FILE_PATH):
    """Reads all mixing indices and parameter values at FILE_PATH. Assumes format from write_mixing_index
    and two parameter values.
    OUTDATED: Use read_mixing_index_file instead, it works on different files that consider multiple realisations and time points.
    """
    names = []
    xValues = []
    yValues = []
    mixingIndices = []
    namesEmpty = True
    with open(FILE_PATH, 'r') as reader:
        for line in reader.readlines():
            idx1 = line.find(", ")
            idx2 = line.find(", ",idx1+1)
            idx3 = line.find(", ",idx2+1)
            idx4 = line.find(", ",idx3+1)
            idx5 = line.find(", ",idx4+1)
            idx6 = line.find(" ",idx5+2)
            if(namesEmpty):
                names.append(line[:idx1])
                names.append(line[idx2+2:idx3])
                namesEmpty=False
            xValues.append(float(line[idx1+2:idx2]))
            yValues.append(float(line[idx3+2:idx4]))
            mixingIndices.append(float(line[idx5+2:idx6]))
    return xValues, yValues, mixingIndices, names

def max_cluster_size(experiments, t, neighbourDistance = 1.05):
    # Get points
    points=[]
    color=[]
    for expIdx, exp in enumerate(experiments):
            for pIdx, track in enumerate(exp.tracks):
                    tIdx, = np.where(np.array(track)[:,0]==t)

                    if len(tIdx)>0: #Does the tracked particle exist at t?
                            tIdx = tIdx[0] #first occurence
                            color.append(expIdx)
                            points.append(np.array(track[tIdx][1:]))

    G = get_distance_based_neighbourhood_graph(points, color, experiments[0], neighbourDistance)
    
    #get connected components
    connectedComponentsSize = [len(comp) for comp in nx.connected_components(G)]
    return max(connectedComponentsSize)

def plot_cluster_histogram(axes, clusterSizes, label ,color, nBins=None):

    # G = get_distance_based_neighbourhood_graph(points, colorList, experiments[0], neighbourDistance)
    
    # #get connected components
    # connectedComponentsSize = [len(comp) for comp in nx.connected_components(G)]

    maxSize = max(clusterSizes)
    if (nBins==None):
        nBins = maxSize

    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(1),np.log10(maxSize),num=nBins)

    # axes.hist(clusterSizes, bins=logbins, label = label, color=color, alpha=0.5) 
    axes.hist(clusterSizes, bins=nBins, label = label, color=color, alpha=0.4)
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel("Cluster size")
    axes.set_ylabel("Number of clusters")

    # Return histogram
    return ndimage.histogram(clusterSizes, 1, maxSize, nBins)

def write_clusters_to_file(output_file, experiments, time, neighbourDistance, newRealisation=False):
    """Finds neighbourhood graph and saves the cluster distribution by appending output_file. 
    Also records the time. If newRealisation=True it will create a line to indicate that.
    Assumes colorIdx = 0 is green, colorIdx=1 is red. Files will be of the form
    ## Realisation 2 (neighbour distance = 1.05)
    # t 0
    nClustersSize1 nGreenClustersSize1 nRedClustersSize1
    nClustersSize2 nGreenClustersSize2 nRedClustersSize2
    ...
    """
    # Find neigbourhood graph
    points, colorIdx = get_positions_and_particle_types(experiments, time)
    if neighbourDistance==None:
        graph = get_voronoi_based_neighbourhood_graph(points, colorIdx, experiments[0])
    else:
        graph = get_distance_based_neighbourhood_graph(points, colorIdx, experiments[0], neighbourDistance)
    
    # Write clustering to file
    with open(output_file, 'a') as f:
            if newRealisation!=False:
                f.write(f"## Realisation {newRealisation} (neighbour distance = {neighbourDistance})\n")
            f.write('# t %f\n' % time)
            
            connectedComponents = [comp for comp in nx.connected_components(graph)]
            clusterSizes = [len(comp) for comp in connectedComponents]
            clusterColor = [colorIdx[comp.pop()] for comp in connectedComponents] # .pop removes item from set
            # for size in range(1, max(clusterSizes)+1):
            #     f.write("%d \n" % clusterSizes.count(size))
            for clusterIdx in range(len(clusterSizes)):
                color = "red" if clusterColor[clusterIdx]==0 else "green"
                f.write(f"{clusterSizes[clusterIdx]} {color}\n")

def write_mixing_index_to_file(output_file, experiments, time, neighbourDistance, newRealisation=False):
    """Saves the cluster distribution save in the graph by appending the output_file. 
    Also records the time. If newRealisation=True it will create a line to indicate that.
    """
    with open(output_file, 'a') as f:
            if newRealisation!=False:
                f.write(f"## Realisation {newRealisation} (neighbour distance = {neighbourDistance})\n")
            f.write('# t %f\n' % time)
            
            mixingIdx, mixingIdxGreen, mixingIdxRed = calculate_mixing_index(experiments, time, neighbourDistance)

            f.write(f"{mixingIdx} {mixingIdxGreen} {mixingIdxRed}\n")

def write_data_for_all_times(observable, output_file, greenData, redData, neighbourDistance, fileIdx):
    """Calculate the cluster distributions for the realisation described by green_data and red_data
    and writes it to output_file. If neighbourDistance=None, it will find the clusters 
    using the Voronoi-based neighbourhood measure, otherwise it will use the distance-based method.
    "observable" can be either "mixingIdx" or "clustering": This determines which observable is calculated
    and written into the file.
    """
    newRealisation=fileIdx
    for time in greenData.measurementTimes:
        # Calculate clustering
        if observable == "clustering":
            write_clusters_to_file(output_file, [greenData, redData], time, neighbourDistance, newRealisation=newRealisation)
        # Calculate mixing index
        elif observable == "mixingIdx":
            write_mixing_index_to_file(output_file, [greenData, redData], time, neighbourDistance, newRealisation=newRealisation)
        else: 
            raise ValueError("Unknown observable type; Can't write data.")

        newRealisation=False
    print("Data written for one realisation")

def write_data_for_all_realisations(observable, parameterFile, outputFile, neighbourDistance):
    """Goes through all tracks belonging to parameterFile and writes their cluster sizes into 
    output for every time.
    "observable" can be either "mixingIdx" or "clustering": This determines which observable is calculated
    and written into the file.
    """

    outOfFiles = False
    fileIdx=0
   
    while(not outOfFiles):
        try: # See if file with this index exists
            if fileIdx==0: # See if unnumbered track file exists
                try:
                    green, red, params = load_simulation_data(parameterFile)
                except:
                    fileIdx += 1
            if fileIdx > 0: # numbered track file
                green, red, params = load_simulation_data(parameterFile, trackIdx=f"_{fileIdx}")
        except:
            outOfFiles = True
        if not outOfFiles:
            write_data_for_all_times(observable, outputFile, green, red, neighbourDistance, fileIdx)
            fileIdx += 1

def read_clustering_file(inputFile):
    """Reads out a file that contains cluster sizes for multiple time steps and realisations. 
    Adds the numbers up over the different realisations and returns the accumulated numbers.
    """
    with open(inputFile, mode="r") as file:
        csvFile = csv.reader(file, delimiter=" ")

        measurementTimes = []
        clusterSizes = [] #clusterSizes[timeIdx][clusterIdx]
        clusterSizesGreen = [] #Clusters of only green particles
        clusterSizesRed = [] # Clusters of only red particles
        timeIdx = -1

        for line in csvFile:
            if line[0] =="##": #Next realisation?
                timeIdx = -1
            elif line[0]=="#": #Next time step?
                timeIdx += 1  
                if len(measurementTimes) == timeIdx: # Never had this time step before?
                    measurementTimes.append(float(line[2]))
                    clusterSizes.append([])
                    clusterSizesGreen.append([])
                    clusterSizesRed.append([])
                assert(measurementTimes[timeIdx]==float(line[2]))  
                
            else: # This line contains a cluster size
                clusterSizes[timeIdx].append(int(line[0]))
                if line[1]=="green":
                    clusterSizesGreen[timeIdx].append(int(line[0]))
                elif line[1]=="red":
                    clusterSizesRed[timeIdx].append(int(line[0]))
                else:
                    raise ValueError(f"Found unknown particle type in file {inputFile}")

    return clusterSizes, clusterSizesGreen, clusterSizesRed, measurementTimes

def read_mixing_index_file(inputFile):
    """Reads out a file that contains cluster sizes for multiple time steps and realisations. 
    Averages the indeices over all realisations
    """
    nRealisations = 0 
    measurementTimes = []
    mixingIdx = []
    mixingIdxGreen = []
    mixingIdxRed = []
    with open(inputFile, mode="r") as file:
        csvFile = csv.reader(file, delimiter=" ")
        for line in csvFile:
            if line[0] =="##": #Next realisation?
                timeIdx = -1 #Will be incremented before first use
                nRealisations += 1
            elif line[0]=="#": #Next time step?
                timeIdx += 1  
                if len(measurementTimes) == timeIdx: # Never had this time step before?
                    measurementTimes.append(float(line[2]))
                    mixingIdx.append(0)
                    mixingIdxGreen.append(0)
                    mixingIdxRed.append(0)
            else: # This line contains a mixing index
                assert(len(measurementTimes)==len(mixingIdx))
                mixingIdx[timeIdx] += float(line[0])
                mixingIdxGreen[timeIdx] += float(line[1])
                mixingIdxRed[timeIdx] += float(line[2])
    
    # Average 
    mixingIdx = np.array(mixingIdx)/nRealisations
    mixingIdxGreen = np.array(mixingIdxGreen)/nRealisations
    mixingIdxRed = np.array(mixingIdxRed)/nRealisations
    
    return mixingIdx, mixingIdxGreen, mixingIdxRed, measurementTimes
                


########################################### Animation ########################################################

def readTracksFile(filename):
    """This is similar to read_csv in analysis library, but has a different output structure and is
    only used here"""
    measurementTimes = []
    positionMeasurements = []
    colorMeasurements = []
    with open(filename, mode="r") as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            measurementTimes.append(float(lines[0])) 
            positionMeasurements.append([])
            colorMeasurements.append([])
            nParticles = int((len(lines)-1)/3)
            for particleIdx in range(nParticles):
                colorMeasurements[-1].append(lines[1+3*particleIdx])
                positionMeasurements[-1].append([float(lines[1+3*particleIdx+1]), #x value
                                              float(lines[1+3*particleIdx+2])])   #y value

    return measurementTimes, colorMeasurements, np.array(positionMeasurements)
     
def getColors(colorMeasurement): ##ToDo: Get right colors
    c = []
    for color in colorMeasurement:
        if color==" red":
            c.append('red')
        elif color==" green":
            c.append('green')
        elif color==" greenPlus":
            c.append('green')
        else:
            assert False, "Invalid color value"
    return c

def animate_tracks(PATH):
    #Get parameters
    sim = experiment("test")
    params = sim.read_parameter_file(PATH)
    xLength = params["Length"] 
    yLength = params["Length"]
    R = 0.5 #WARNING: Radius of the cells since we set r0=1
    transparency = 0.7 #transparency of circles 

    # Get tracks
    try: # Single realisation
        measurementTimes, colorMeasurements, positionMeasurements = readTracksFile(PATH+"_tracks.csv")
    except: # Multiple realisations
        measurementTimes, colorMeasurements, positionMeasurements = readTracksFile(PATH+"_tracks_1.csv")
    positionMeasurements = np.array(positionMeasurements)

    positionMeasurements = np.array(positionMeasurements)


    #Set up figure 
    fig,ax = plt.subplots()
    plt.axis([0, xLength, 0, yLength])
    plt.gca().set_aspect('equal', adjustable='box') #Makes both axis have some scaling while keeping the set limits

    #Record the animation
    camera = Camera(fig)
    for frameIdx in range(len(measurementTimes)):
        c = getColors(colorMeasurements[frameIdx])
        for particleIdx in range(len(positionMeasurements[0])):
            circle = plt.Circle((positionMeasurements[frameIdx][particleIdx][0],
                                        positionMeasurements[frameIdx][particleIdx][1]), 
                                        radius=R, linewidth=0, color = c[particleIdx], alpha = transparency)
            plt.gca().add_artist(circle)
        plt.gca().set_title("Time = %f"%measurementTimes[frameIdx])    
        camera.snap()
        #Delete all artists before next frame
        for artist in plt.gca().lines + plt.gca().collections:
            artist.remove()
    animation = camera.animate()
    animation.save(PATH+".mov")

def animate_tracks_same_col(PATH):
    #Get parameters
    sim = experiment("test")
    params = sim.read_parameter_file(PATH)
    xLength = params["Length"] 
    yLength = params["Length"]
    exponentialExpRate = params["exponentialExpRate"] 
    stepLimit = params["stepLimit"] 
    stepDuration = params["stepDuration"] 
    
    R = 0.5 #WARNING: Radius of the cells since we set r0=1
    transparency = 0.7 #transparency of circles 

    # Get tracks
    try: # Single realisation
        measurementTimes, colorMeasurements, positionMeasurements = readTracksFile(PATH+"_tracks.csv")
    except: # Multiple realisations
        measurementTimes, colorMeasurements, positionMeasurements = readTracksFile(PATH+"_tracks_1.csv")
    positionMeasurements = np.array(positionMeasurements)

    positionMeasurements = np.array(positionMeasurements)


    #Set up figure 
    fig,ax = plt.subplots()
    # plt.axis([0, xLength, 0, yLength])
    # plt.axis([0, xLength*np.exp(stepLimit*stepDuration*exponentialExpRate), 0, yLength*np.exp(stepLimit*stepDuration*exponentialExpRate)])  
    plt.gca().set_aspect('equal', adjustable='box') #Makes both axis have some scaling while keeping the set limits

    #Record the animation
    camera = Camera(fig)
    for frameIdx in range(len(measurementTimes)):
        c = getColors(colorMeasurements[frameIdx])
        for particleIdx in range(len(positionMeasurements[0])):
            circle = plt.Circle((positionMeasurements[frameIdx][particleIdx][0],
                                        positionMeasurements[frameIdx][particleIdx][1]), 
                                        radius=R, linewidth=0, color = 'black', alpha = transparency)
            plt.gca().add_artist(circle)
        plt.gca().set_title("Time = %f"%measurementTimes[frameIdx]) 
        plt.axis([0, xLength*np.exp(measurementTimes[frameIdx]*exponentialExpRate), 0, yLength*np.exp(measurementTimes[frameIdx]*exponentialExpRate)])  
        
        camera.snap()
        #Delete all artists before next frame
        for artist in plt.gca().lines + plt.gca().collections:
            artist.remove()
    animation = camera.animate()
    animation.save(PATH+".mov")

def final_snapshot(PATH):
    #Get parameters
    sim = experiment("test")
    params = sim.read_parameter_file(PATH)
    xLength =params["Length"] 
    yLength =params["Length"]
    R = 0.5 #WARNING: Radius of the cells since we set r0=1
    transparency = 0.7 #transparency of circles

    #Get tracks
    try: # Single realisation
        measurementTimes, colorMeasurements, positionMeasurements = readTracksFile(PATH+"_tracks.csv")
    except: # Multiple realisations
        measurementTimes, colorMeasurements, positionMeasurements = readTracksFile(PATH+"_tracks_1.csv")
    positionMeasurements = np.array(positionMeasurements)


    #Set up figure 
    fig,ax = plt.subplots()
    plt.axis([0, xLength, 0, yLength])
    plt.gca().set_aspect('equal', adjustable='box') #Makes both axis have some scaling while keeping the set limits

    frameIdx = -1 # last frame
    c = getColors(colorMeasurements[frameIdx])
    for particleIdx in range(len(positionMeasurements[0])):
        circle = plt.Circle((positionMeasurements[frameIdx][particleIdx][0],
                            positionMeasurements[frameIdx][particleIdx][1]), 
                            radius=R, linewidth=0, color = c[particleIdx], alpha = transparency)
        plt.gca().add_artist(circle)
    plt.gca().set_title("Time = %f"%measurementTimes[frameIdx])    
    plt.savefig(PATH+"_final_frame.png")




# from numba import jit
# @jit(nopython=True)
# def generate_reference_histogram(n_particles, n_bins, x_max,y_max, seed=None):
#         '''Generates a radial density histogram for uniform distributed particles in a box of 
#         size x_max y_max. This serves as a normalisation for the measured density.
#         :n_particles Number of particles that are generated
#         :n_bins Number of bins to be used. The bins are evenly distributed between 0 and max_distance
#             where max_distance is calculated from x_max and y_max
#         :x_max, y_max Dimensions of the rectangle for which we generate the density
#         :rng Random number generator to be used
#         '''

#         #Numba compatible r andom number genration
#         if seed!=None:
#             np.random.seed(seed)

#         #Generate all points
#         max_distance = np.sqrt(x_max**2+y_max**2)
#         x_values = np.random.random((n_particles))*x_max
#         y_values = np.random.random((n_particles))*y_max
#         points = np.stack((x_values,y_values),axis=1)


#         # distances = spatial.distance.pdist(points)
#         # reference_histogram = ndimage.histogram(distances, 0, max_distance,n_bins)

#         #Prepare histogram
#         bins = np.linspace(0, max_distance, n_bins+1)[:-1] #left edge of every bin
#         histogram = np.zeros(n_bins)

#         #calculate distances between all points
#         for n in range(n_particles):
#             for m in range(n):
#                 distance = self.distance(points[n]-points[m])
#                 bin_idx = int(np.floor(distance/max_distance*n_bins))
#                 histogram[bin_idx]+=1
        
#         reference_histogram = histogram

#         return reference_histogram

