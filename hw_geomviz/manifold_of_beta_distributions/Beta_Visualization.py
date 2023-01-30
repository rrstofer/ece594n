from multiprocessing.sharedctypes import Value
from re import I
import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions

import geomstats.visualization as visualization
from geomstats.geometry.special_euclidean import SpecialEuclidean

import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt

SE2_GROUP = SpecialEuclidean(n=2, point_type="matrix")

beta = BetaDistributions()


class Beta:
    
    """ Class for the visualization of beta manifold
    
    Parameters
    ----------
    points : array-like, shape=[..., 2]
            Point representing a beta distribution.
    
    Returns
    -------
    Plots
    """

    def __init__(self):
        pass

    def process_points(self,points,**kwargs):
        """ Confirms that points passed into function lie on manifold and prepares them for plotting

        by Marianne Arriola

        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Beta manifold points to be plotted.
    	"""
        # TODO: figure out how this function works
        # points = gs.to_numpy(points)
        points = np.array(points)
        if len(points.shape) == 1:
            points = np.expand_dims(points, axis=0)

        if not len(points) > 0: raise ValueError("No points given")
        if not np.all(points > 0): raise ValueError("Points must be in the upper-right quadrant of Euclidean space")
        if not ((points.shape[-1] == 2 and len(points.shape) == 2)): raise("Points must lie in 2D space")
        limit = np.max(points)
        limit += (limit/10)
        return points, limit

    def plot(self,points,size=None,**kwargs):
        """ Draws the beta manifold

        by Yiliang Chen & Marianne Arriola & Ryan Stofer

        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Beta manifold points to be plotted.
        size : array-like, shape=[..., 2]
            Defines the range of the manifold to be shown.
            Optional, default: None
    	"""
        points,limit = self.process_points(points)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        if not size:
            ax.set(xlim=(0, limit), ylim=(0, limit))
        else:
            ax.set(xlim=(0, size[0]), ylim=(0, size[1]))
        ax.scatter(points[:,0],points[:,1],**kwargs)
        plt.title('Points on 2D Manifold of Beta Distributions')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        #fig.show()

    def plot_rendering(self,initial_point=[0,0],size=[10,10],sampling_period=1):
        """ Draws the beta manifold

        by Yiliang Chen & Allen Wang 

        Parameters
        ----------
        Initial_point : array-like, shape=[1, 2]
            Defines initial point for plot rendering
            Optional, default: [0,0]
        size : array-like, shape=[..., 2]
            Defines the range of the samples to be shown
            Optional, default: [10,10]
        sampling_period : float, >0
            Defines the sampling period of the sampled data
            Optional, default: 1
        """
        for value in initial_point:
                if value < 0:
                    raise ValueError("Initial Point {} is not in the first quadrant".format(initial_point))

        sz = gs.array(size)
        if sz.size != 2:
            raise ValueError(
                "size should be a 1*2 array"
            )
        x = gs.linspace(initial_point[0], (initial_point[0]+size[0]-1)*sampling_period, size[0])
        y = gs.linspace(initial_point[1], (initial_point[1]+size[1]-1)*sampling_period, size[1])
        points = [[i,j] for i in x for j in y]
        points_x = [i[0] for i in points]
        points_y = [i[1] for i in points]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        
        ax.scatter(points_x,points_y)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')

    
    def plot_grid(self,size,initial_point=[0,0],n_steps=100,n_points=10,step=1,**kwargs):
        """ Draws the grids of beta manifold

        by Yiliang Chen

        Parameters
        ----------
        size : array-like, shape=[..., 2]
            Defines the range of the grids to be shown.
        initial_point : array-like, shape=[1,2]
            Defines the initial point for plotting the beta manifold grid.
            Optional, default: [0,0]
        n_steps : int, >0
            Defines the number of steps for integration.
            Optional, default: 100
        n_points : int, >0
            Defines the number of points for interpolation.
            Optional, default: 10
        step : float, >0
            Defines the length of a step for the grid
            Optional, default: 1
        """
        for value in initial_point:
            if value < 0:
                raise ValueError("Initial Point {} is not in the first quadrant".format(initial_point))

        sz = gs.array(size)
        if sz.size != 2:
            raise ValueError(
                "size should be a 1*2 array"
            )
        b = [(initial_point[0]+i*step) for i in range(size[0])]
        gF = [(initial_point[1]+i*step) for i in range(size[1])]

        t = gs.linspace(0, 1, n_points)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        for i in b:
            for j in gF:
                start = [i,j]
                end_h = [i+step,j]
                end_v = [i,j+step]
                if i < (size[0] + initial_point[0] - 1):
                    grid_h=beta.metric.geodesic(initial_point=start,
                                                end_point=end_h,
                                                n_steps=n_steps)
                    ax.plot(*gs.transpose(gs.array([grid_h(k) for k in t])))
                if j < (size[1] + initial_point[1] - 1):
                    grid_v=beta.metric.geodesic(initial_point=start,
                                                end_point=end_v,
                                                n_steps=n_steps)
                    ax.plot(*gs.transpose(gs.array([grid_v(k) for k in t])))
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')

    def scatter(self,points,**kwargs):
        """ Scatter plot of beta manifold
        
        by Sunpeng Duan & Marianne Arriola
    
        Parameters
        ----------
        points : array-like, shape=[..., 2]
            Manifold point representing a beta distribution.
        """
        points,limit = self.process_points(points)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.set(xlim=(0, limit), ylim=(0, limit))
        ax.scatter(points[:,0],points[:,1],**kwargs)
        ax.set_title("Scatter plot of beta manifolds")
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        
    def plot_geodesic(self,
                      initial_point,
                      end_point = None,
                      initial_tangent_vec = None,
                      n_steps = 100,
                      n_points = 10,
                      **kwargs):
        """ Geodesic plot of beta manifold
        
        by Sunpeng Duan & Allen Wang 
    
        Parameters
        ----------
        initial_point : array-like, shape=[1, 2]
            Starting point representing a beta distribution.
        end_point : array-like, shape=[1, 2]
            Ending point representing a beta distribution.
            Optional, default: None.
        initial_tangent_vec : array-like, shape=[1, 2]
            Initial tangent vector for the starting point.
            Optional, default: None.
        n_steps : int, >0
            Number of steps for integration.
            Optional, default: 100.
        n_points : int, >0
            Number of points for interpolation.
            Optional, default: 10.
        """
    
        
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
                 
        t = gs.linspace(0, 1, n_points)
            
        if end_point is not None:

            for point in [initial_point, end_point]:
                x, y = point
                if x < 0 or y < 0:
                    raise ValueError("Point {} is not in the first quadrant".format(point))

            upperLimit = np.max(list(zip(initial_point, end_point))) + 1
            lowerLimit = np.min(list(zip(initial_point, end_point))) - 1
            geod = beta.metric.geodesic(initial_point=initial_point, 
                                        end_point=end_point,
                                        n_steps=n_steps)(t)
            
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.set(xlim=(lowerLimit, upperLimit), ylim=(lowerLimit, upperLimit))
            ax.scatter(geod[:,0],geod[:,1],**kwargs)
            ax.set_title("Geodesic between two beta distributions for the Fisher-Rao metric")
            plt.xlabel(r'$\alpha$')
            plt.ylabel(r'$\beta$')
            #fig.show()
            
        if initial_tangent_vec is not None:
            if (initial_point < 0).any():
                raise ValueError("The initial point is not in the first quadrant")
            geod = beta.metric.geodesic(initial_point=initial_point,
                                        initial_tangent_vec=initial_tangent_vec,
                                        n_steps=n_steps)(t)
            upperLimit = np.max(geod) + 1
            lowerLimit = np.min(geod) - 1
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.set(xlim=(lowerLimit, upperLimit), ylim=(lowerLimit, upperLimit))
            ax.scatter(geod[:,0],geod[:,1],**kwargs)
            ax.set_title("Geodesic between two beta distributions for the Fisher-Rao metric")
            plt.xlabel(r'$\alpha$')
            plt.ylabel(r'$\beta$')
        
    def plot_geodestic_ball(self,
                      initial_point,
                      n_rays,
                      ray_length,
                      n_steps = 100,
                      n_points = 10,
                      **kwargs):
        """ Geodesic ball plot of beta manifold
        
        by Sunpeng Duan & Allen Wang & Marianne Arriola
    
        Parameters
        ----------
        inital_point : array-like, shape=[1, 2]
            Point representing a beta distribution.
        tangent_vecs : array-like, shape=[..., 2]
            Set of tangent vectors for geodesic ball.
        n_steps : int, >0
            Number of steps for integration.
            Optional, default: 100.
        n_points : int, >0
            Number of points for interpolation.
            Optional, default: 10.
        """

        theta = gs.linspace(-gs.pi, gs.pi, n_rays)
        directions = gs.transpose(gs.stack((gs.cos(theta), gs.sin(theta))))
        direction_norms = beta.metric.squared_norm(directions, initial_point) ** (1 / 2)
        unit_vectors = directions / gs.expand_dims(direction_norms, 1)
        tangent_vecs = ray_length * unit_vectors
        
        t = gs.linspace(0, 1, n_points)
        
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        for j in range(len(tangent_vecs)):
            geod = beta.metric.geodesic(initial_point=initial_point, 
                                        initial_tangent_vec=tangent_vecs[j, :],
                                        n_steps = n_steps)
            if j == 0:
                allPoint = geod(t)
            else:
                allPoint = np.concatenate((allPoint, geod(t)))
        
            ax.plot(*gs.transpose(gs.array([geod(k) for k in t])))

        x_lowerLimit = np.min(allPoint[:,0]) ; x_lowerLimit -= x_lowerLimit/10
        x_upperLimit = np.max(allPoint[:,0]) ; x_upperLimit += x_upperLimit/10
        y_lowerLimit = np.min(allPoint[:,1]) ; y_lowerLimit -= y_lowerLimit/10
        y_upperLimit = np.max(allPoint[:,1]) ; y_upperLimit += y_upperLimit/10
        
        ax.set(xlim=(x_lowerLimit, x_upperLimit), ylim=(y_lowerLimit, y_upperLimit))
        ax.set_title("Geodesic ball of the space of beta distribution")
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        