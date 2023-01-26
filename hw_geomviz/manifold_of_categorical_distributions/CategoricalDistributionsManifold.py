from cmath import nan
from tokenize import endpats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
import numpy as np
import pandas as pd
from geomstats.information_geometry.categorical import CategoricalDistributions, CategoricalMetric
class CategoricalDistributionsManifold:
    r""" Class for visualizing the manifold of categorical distributions.

    This is the set of $n+1$-tuples of positive reals that sum up to one,
    i.e. the $n$-simplex. Each point is the parameter of a categorical
    distribution, i.e. gives the probabilities of $n$ different outcomes
    in a single experiment.

    Attributes:
    -----------
    dim : integer
        Dimension of the manifold.
    points: array-like, [[..., dim + 1], [..., dim + 1], ... ]
        Discrete points to be plotted on the manifold.

    Notes:
    ------
    The class only implements visualization methods for 2D and 3D manifolds.
    """
    def __init__(self, dim):
        """ Construct a CategoricalDistributionsManifold object.

        Construct a CategoricalDistributionsManifold with a given dimension.

        Parameters:
        -----------
        dim : integer
            Dimension of the manifold
        
        Returns:
        --------
        None.

        Notes:
        ------
        dim should be a positive integer. 
        The methods only support visualization of 2-D and 3-D manifolds. 
        """
        self.dim = dim
        self.points = []
        self.ax = None
        self.elev, self.azim = None, None
        self.metric = CategoricalMetric(dim = self.dim)
        self.dist = CategoricalDistributions(dim = self.dim)


    def plot(self):
        """ Plot the 2D or 3D Manifold.

        Plot the 2D Manifold as a regular 2-simplex(triangle) or
        the 3D Manifold as a regular 3-simplex(tetrahedral). 

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        Notes
        -----
        This method only works properly if the dimension is 2 or 3.
        
        References
        ----------
        Simplex: https://en.wikipedia.org/wiki/Simplex
        """
        min_limit = 0
        max_limit = 1
        plt.figure(dpi = 100)
        self.set_axis(min_limit, max_limit)
        if self.dim == 3:
            self.set_view()
            x = [0, 1, 0, 0]
            y = [0, 0, 1, 0]
            z = [0, 0, 0, 1]
            vertices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            tupleList = list(zip(x, y, z))
            poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
            self.ax.add_collection3d(Poly3DCollection(poly3d, edgecolors='k', facecolors=(0.9, 0.9, 0.9, 1.0), linewidths=3, alpha=0.2))

        elif self.dim == 2:
            X = np.linspace(start = min_limit, stop = max_limit, num = 101, endpoint = True)
            Y = 1 - X
            self.ax.fill_between(X, Y, color = (0.9, 0.9, 0.9, 1.0))
            self.ax.set_title("2 Dimension Categorical Manifold")
            
    def plot3D(self):
        """ Plot the 3D Manifold using Plotly.

        Plot the 3D Manifold as a regular 3-simplex(tetrahedral). 

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        Notes
        -----
        This method only works properly if the dimension is 3.
        
        """
        if self.dim != 3:
            print('Invalid Dimension')
            return 
        fig = go.Figure(data=[go.Mesh3d(
                                x = [0, 1, 0, 0],
                                y = [0, 0, 1, 0],
                                z = [0, 0, 0, 1],
                                colorbar_title='z',
                                colorscale=[[0, 'gold'],
                                            [0.5, 'mediumturquoise'],
                                            [1, 'magenta']],
                                # Intensity of each vertex, which will be interpolated and color-coded
                                intensity=[0, 0.33, 0.66, 1],
                                # i, j and k give the vertices of triangles
                                # here we represent the 4 triangles of the tetrahedron surface
                                i=[0, 0, 0, 1],
                                j=[1, 2, 3, 2],
                                k=[2, 3, 1, 3],
                                name='y',
                                showscale=False,
                                opacity=0.25
                                )
                            ]).update(layout=dict(title=dict(x=0.5)))
        fig.update_layout(title_text='3 Dimension Categorical Manifold')
        fig.show()
        
    def set_points(self, points):
        self.points = points


    def scatter(self, n_samples, **scatter_kwargs):
        """ Scatter plot some randomly sampled points in the manifold.

        Plot the manifold along with some randomly sampled points
        lying on the manifold. 

        Parameters:
        -----------
        n_samples : integer
            The number of randomly sampled points.

        **scatter_kwargs: optional
            Inherits the matplotlib scatter function parameters.
        
        Returns:
        --------
        None.

        Notes:
        ------
        This method internally calls the plot method. 
        """
        self.set_points(self.dist.random_point(n_samples=n_samples))
        if self.dim == 3:
            # Plot 3D Mesh with Sample Scatter Points in Plotly
            df = pd.DataFrame(self.points, columns = ['x1', 'x2','x3','x4'])
            scatter = px.scatter_3d(df,x = 'x1', 
                                    y = 'x2', 
                                    z = 'x3',
                                    color = 'x4',
                                    title = f'3D Scatterplot for {n_samples} Samples',
                                    opacity = 0.5)
            scatter.update_traces(marker_size = 3)

            mesh = go.Figure(data=[
                                go.Mesh3d(
                                    x = [0, 1, 0, 0],
                                    y = [0, 0, 1, 0],
                                    z = [0, 0, 0, 1],
                                    colorbar_title='z',
                                    colorscale=[[0, 'gold'],
                                                [0.5, 'mediumturquoise'],
                                                [1, 'magenta']],
                                    # Intensity of each vertex, which will be interpolated and color-coded
                                    intensity=[0, 0.33, 0.66, 1],
                                    # i, j and k give the vertices of triangles
                                    # here we represent the 4 triangles of the tetrahedron surface
                                    i=[0, 0, 0, 1],
                                    j=[1, 2, 3, 2],
                                    k=[2, 3, 1, 3],
                                    name='y',
                                    showscale=False,
                                    opacity=0.25,
                                    hoverinfo='skip',
                                    hovertemplate=None
                                )
                            ])
            mesh.update_traces(
               hovertemplate=None,
               hoverinfo='skip'
            )
            fig = go.Figure(data=scatter.data+mesh.data)
            fig.update_layout(title_text=f'3D Scatterplot for {n_samples} Samples')
            fig.show()
                
                
        elif self.dim == 2: 
            self.plot()
            for point in self.points:
                self.ax.scatter(point[0], point[1], **scatter_kwargs)
            self.ax.set_title(f'2 Dimension Categorical Manifold with {n_samples} Samples')
        self.clear_points()


    def plot_geodesic(self, initial_point, end_point = None, tangent_vector = None):
        """ Plot a geodesic on the manifold.

        Plot a geodesic that is either specified with 
        1) an initial_point and an end_point, or
        2) an initial point and an initial tangent vector
        on the manifold.

        Parameters:
        -----------
        initial_point: array-like, shape = [..., dim + 1]
            Initial point on the manifold.

        end_point: optional, array-like, shape = [..., dim + 1]
            End point on the manifold.

        tangent_vector: optional, array-like, shape = [..., dim + 1]
            Initial tangent vector at the initial point. 

        Returns:
        --------
        None.

        Notes:
        ------
        Either end_point or tangent_vector needs to be specified.
        The initial point will be marked red.
        The initial tangent vector will also be plotted starting from the initial point.
        """
        self.plot()
        geodesic = self.metric.geodesic(initial_point=initial_point, end_point = end_point, initial_tangent_vec = tangent_vector)
        num_samples = 200
        if self.dim == 3:
            for i in range(num_samples):
                point = geodesic(i/num_samples)
                self.ax.scatter(point[0], point[1], point[2], color='blue', s = 2)
            self.ax.scatter(geodesic(0)[0], geodesic(0)[1], geodesic(0)[2], color='red', s = 30)
            if tangent_vector is not None:
                normalized_tangent_vector = tangent_vector/np.sum(np.power(tangent_vector, 2))
                self.ax.quiver(
                    initial_point[0],
                    initial_point[1],
                    initial_point[2],
                    normalized_tangent_vector[0],
                    normalized_tangent_vector[1],
                    normalized_tangent_vector[2],
                    color = 'red',
                    length = 0.1,
                    normalize = True
                )
        geodesic = self.metric.geodesic(initial_point=initial_point, end_point = end_point, initial_tangent_vec = tangent_vector)
        num_samples = 100
        geodesic_points = np.zeros(shape=(num_samples,4))
        if self.dim == 3:
            for i in range(num_samples):
                point = geodesic(i/num_samples)
                geodesic_points[i] = point
            df = pd.DataFrame(geodesic_points, columns = ['x1', 'x2','x3','x4'])
            geodesic_plot = px.scatter_3d(df,x = 'x1', 
                                    y = 'x2', 
                                    z = 'x3',
                                    color = 'x4',
                                    title = '3D Scatterplot for Geodesic',
                                    opacity = 0.5)
            geodesic_plot.update_traces(marker_size = 2)

            mesh = go.Figure(data=[
                                go.Mesh3d(
                                    x = [0, 1, 0, 0],
                                    y = [0, 0, 1, 0],
                                    z = [0, 0, 0, 1],
                                    colorbar_title='z',
                                    colorscale=[[0, 'gold'],
                                                [0.5, 'mediumturquoise'],
                                                [1, 'magenta']],
                                    # Intensity of each vertex, which will be interpolated and color-coded
                                    intensity=[0, 0.33, 0.66, 1],
                                    # i, j and k give the vertices of triangles
                                    # here we represent the 4 triangles of the tetrahedron surface
                                    i=[0, 0, 0, 1],
                                    j=[1, 2, 3, 2],
                                    k=[2, 3, 1, 3],
                                    name='y',
                                    showscale=False,
                                    opacity=0.25
                                )
                            ])
            mesh.update_traces(
               hovertemplate=None,
               hoverinfo='skip'
            )
            
            # if tangent_vector is not None:
            normalized_tangent_vector = tangent_vector/np.sum(np.power(tangent_vector, 2))
                # self.ax.quiver(
                #     initial_point[0],
                #     initial_point[1],
                #     initial_point[2],
                #     normalized_tangent_vector[0],
                #     normalized_tangent_vector[1],
                #     normalized_tangent_vector[2],
                #     color = 'red',
                #     length = 0.1,
                #     normalize = True
                # )
                
            pt1 = np.array([initial_point[0]])
            pt2 = np.array([initial_point[1]])
            pt3 = np.array([initial_point[2]])
            arr1 = np.array([normalized_tangent_vector[0]])
            arr2 = np.array([normalized_tangent_vector[1]])
            arr3 = np.array([normalized_tangent_vector[2]])
            cone = go.Figure(data=go.Cone(x=pt1, y=pt2, z=pt3, u=arr1, v=arr2, w=arr3, sizeref=0.2))

            fig = go.Figure(data=geodesic_plot.data + mesh.data + cone.data).update(layout=dict(title=dict(x=0.5)))
            fig.update_layout(title_text='3 Dimension Categorical Manifold with Geodesic')
            fig.show()
        
        elif self.dim == 2:
            for i in range(num_samples):
                point = geodesic(i/num_samples)
                self.ax.scatter(point[0], point[1], color='blue', s = 2)
            self.ax.scatter(geodesic(0)[0], geodesic(0)[1], color='red', s = 30)
            if tangent_vector is not None:
                normalized_tangent_vector = tangent_vector/np.sum(np.power(tangent_vector, 2))
                self.ax.quiver(
                    initial_point[0],
                    initial_point[1],
                    normalized_tangent_vector[0],
                    normalized_tangent_vector[1],
                    color = 'red',
                    angles = 'xy',
                    scale_units = 'xy',
                    scale = 10,
                )


    def plot_log(self, end_point, base_point):
        """ Plot the result of taking the logarithm of two points on the manifold. 

        Plot the tangent vector calculated from taking the logarithm between the two input points on the manifold.

        Parameters:
        -----------
        end_point: array-like, shape = [..., dim + 1]
            End point on the manifold.
        
        base_point: array-like, shape = [..., dim + 1]
            Base point on the manifold.
        
        Returns:
        --------
        None.
        """
        tangent_vec = self.metric.log(point=end_point,base_point=base_point)
        self.plot_helper(end_point=end_point,base_point=base_point, tangent_vec=tangent_vec, operation='Log')


    def plot_exp(self, tangent_vec, base_point):
        """ Plot the result of taking the exponential of one point with one of its tangent vector. 

        Plot the end point resulting from taking the exponential of the base point with respect to a tangent vector.

        Parameters:
        -----------
        tangent_vec: array-like, shape = [..., dim + 1]
            A tangent vector at the base point.
        
        base_point: array-like, shape = [..., dim + 1]
            Base point on the manifold.
        
        Returns:
        --------
        None.
        """
        end_point = self.metric.exp(tangent_vec=tangent_vec,base_point=base_point)
        self.plot_helper(end_point=end_point,base_point=base_point, tangent_vec=tangent_vec, operation='Exp')


    def plot_helper(self,end_point, base_point, tangent_vec, operation):
        """ Plot two points and a vector together on a manifold.

        Helper function used by plot_exp and plot_log methods.

        Parameters:
        -----------
        end_point: array-like, shape = [..., dim + 1]
            End point on the manifold.
        
        base_point: array-like, shape = [..., dim + 1]
            Base point on the manifold.
        
        tangent_vec: array-like, shape = [..., dim + 1]
            A tangent vector to the manifold.

        Returns:
        --------
        None.

        Notes:
        ------
        The base point and the tangent vector will be marked red.
        THe end point will be marked blue.
        
        """
        self.plot()
        self.ax.set_title(f'{operation} Operation with {self.dim} Dimension Categorical Manifold')
        if self.dim == 3:
            # Plot in Matplotlib
            self.ax.scatter(base_point[0], base_point[1], base_point[2], color='red', s = 30)
            self.ax.scatter(end_point[0], end_point[1], end_point[2], color='blue', s = 30)
            self.ax.quiver(
                base_point[0],
                base_point[1],
                base_point[2],
                tangent_vec[0],
                tangent_vec[1],
                tangent_vec[2],
                color = 'red',
                length = 0.1,
                normalize = True
                )
            
            # Plot in Plotly
            mesh = go.Figure(data=[
                                go.Mesh3d(
                                    x = [0, 1, 0, 0],
                                    y = [0, 0, 1, 0],
                                    z = [0, 0, 0, 1],
                                    colorbar_title='z',
                                    colorscale=[[0, 'gold'],
                                                [0.5, 'mediumturquoise'],
                                                [1, 'magenta']],
                                    # Intensity of each vertex, which will be interpolated and color-coded
                                    intensity=[0, 0.33, 0.66, 1],
                                    # i, j and k give the vertices of triangles
                                    # here we represent the 4 triangles of the tetrahedron surface
                                    i=[0, 0, 0, 1],
                                    j=[1, 2, 3, 2],
                                    k=[2, 3, 1, 3],
                                    name='y',
                                    showscale=False,
                                    opacity=0.25
                                )
                            ])
            
            # if tangent_vector is not None:
            normalized_tangent_vector = tangent_vec/np.sum(np.power(tangent_vec, 2))
                
            pt1 = np.array([base_point[0]])
            pt2 = np.array([base_point[1]])
            pt3 = np.array([base_point[2]])
            arr1 = np.array([normalized_tangent_vector[0]])
            arr2 = np.array([normalized_tangent_vector[1]])
            arr3 = np.array([normalized_tangent_vector[2]])
            cone = go.Figure(data=go.Cone(x=pt1, y=pt2, z=pt3, u=arr1, v=arr2, w=arr3, sizeref=0.1))
            
            arr = [[ end_point[0] ,end_point[1],end_point[2],end_point[3]]]
            df = pd.DataFrame(arr, columns = ['x1', 'x2','x3','x4'])
            print(df)
            point = px.scatter_3d(df,x = 'x1', 
                                    y = 'x2', 
                                    z = 'x3',
                                    color = 'x4',
                                    opacity = 0.5)
            point.update_traces(marker_size = 6)
            fig = go.Figure(data=mesh.data + cone.data + point.data).update(layout=dict(title=dict(x=0.5)))
            fig.update_layout(title_text=f'{operation} Operation with {self.dim} Dimension Categorical Manifold')
            fig.show()
                
        elif self.dim == 2:
            self.ax.scatter(base_point[0], base_point[1], color='red', s = 30)
            self.ax.scatter(end_point[0], end_point[1], color='blue', s = 30)
            self.ax.quiver(
                base_point[0],
                base_point[1],
                tangent_vec[0],
                tangent_vec[1],
                color = 'red',
                angles = 'xy',
                scale_units = 'xy',
                scale = 5,
                )


    def plot_grid(self):
        """ Plot the manifold with a geodesic grid.

        Plot some geodesic grid lines on top of a 2D manifold.

        Parameters:
        -----------
        None.

        Returns:
        --------
        None. 

        Notes:
        ------
        This function only works for 2D manifold. 
        """
        self.plot()
        points = [
        np.array([0.5,0,0.5]),
        np.array([0,0.5,0.5]),
        np.array([0.5,0.5,0]),
        np.array([0.25,0,0.75]),
        np.array([0,0.25,0.75]),
        np.array([0.75,0,0.25]),
        np.array([0,0.75,0.25]),
        ]

        num_samples = 100
        curves = [(0,1),(0,2),(1,2),(3,2),(4,2),(3,4),(5,2),(6,2),(5,6)]
        for curve in curves:
            geodesic = self.metric.geodesic(initial_point=points[curve[0]], end_point= points[curve[1]])
            for i in range(num_samples):
                point = geodesic(i/num_samples)
                self.ax.scatter(point[0], point[1], color='black', s = 1)

    def clear_points(self):
        """ Clear the points stored in the object.

        Clear the points vector stored as an attribute of an object.

        Parameters:
        -----------
        None.

        Returns:
        --------
        None.
        """
        self.points = []

    def set_axis(self, min_limit, max_limit):
        """ Set the axes in 2D or 3D Euclidean space. 

        Set the boundary of each axis for plotting
        as specified in the input for plotting the manifold.

        Parameters:
        -----------
        min_limit : float
            Lower limit for each axis.
        min_limit : float
            Upper limit for each axis.

        Returns:
        --------
        None.

        Notes:
        ------
        This method is not intended to be called externally. 
        """
        if self.dim == 3: 
            ax = plt.subplot(111, projection="3d")
            plt.setp(
                ax,
                xlim = (min_limit, max_limit),
                ylim = (min_limit, max_limit),
                zlim = (min_limit, max_limit),
                anchor = (0,0),
                xlabel = "X",
                ylabel = "Y",
                zlabel = "Z",
            )
            
        elif self.dim == 2:
            ax = plt.subplot(111)
            plt.setp(
                ax,
                xlim = (min_limit, max_limit),
                ylim = (min_limit, max_limit),
                xlabel = "X",
                ylabel = "Y",
                aspect = "equal")

        self.ax = ax


    def set_view(self, elev = 30.0, azim = 20.0):
        """ Set the viewing angle for plotting the 3D manifold.
        
        Set the elevation and azimuthal angle of viewing the 3D manifold.

        Parameters
        ----------
        elev : float
            Angle of elevation from the x-y plane in degrees (default: 30.0).
        azim : float
            Azimuthal angle in the x-y plane in degrees (default: 20.0).

        Returns
        -------
        None.
        """
        if self.dim == 3:
            if self.ax is None:
                self.set_axis()
            self.elev, self.azim = elev, azim
            self.ax.view_init(elev, azim)

