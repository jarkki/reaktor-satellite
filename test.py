import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def radians(degrees):
    return degrees * np.pi / 180.0

def spherical_to_cartesian(radius, theta, phi):
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    return (x,y,z)

def latlon_to_cartesian(radius, lat, lon):
    x = radius * np.cos(radians(lat)) * np.cos(radians(lon))
    y = radius * np.cos(radians(lat)) * np.sin(radians(lon))
    z = radius * np.sin(radians(lat))
    return (x,y,z)

def plot_earth(ax, radius = 6371.0):
    theta, phi = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x,y,z = spherical_to_cartesian(radius, theta, phi)
    return ax.plot_surface(x, y, z, alpha=0.05, color='#fff1d0', rstride=1, cstride=1, linewidth=1, shade=1)

def plot_point(x,y,z, ax, col='#086788', size=50):
    return ax.scatter([x],[y],[z],c=col,s=size,edgecolor='none')

def plot_line(xyz1, xyz2, ax, col='r', width=1.0):
    x1,y1,z1 = xyz1
    x2,y2,z2 = xyz2
    return ax.plot([x1,x2],[y1,y2],[z1,z2],c=col, linewidth=width)

def line_sphere_intersect(radius, xyz1, xyz2, xyz3=(0.0,0.0,0.0)):
    """ Calculates whether a line and a sphere intersect

    Assumes the sphere is centered at origin.

    Args:
    -----
    radius --  The radius of the sphere (centered at origo)
    xyz1   --  Tuple of cartesian coordinates for the first point: (x1,y1,z1)
    xyz2   --  Tuple of cartesian coordinates for the second point: (x2,y2,z2)
    xyz3   --  Tuple of cartesian coordinates for center of the sphere: (x3,y3,z3)

    Returns:
    --------
    List with the intersection coordinates or empty list if no intersection
    """
    x1,y1,z1 = xyz1
    x2,y2,z2 = xyz2
    x3,y3,z3 = xyz3

    a = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    b = 2.0 * ((x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3))
    c = x3**2 + y3**2 + z3**2 + x1**2 + y1**2 + z1**2 - 2.0*(x3*x1 + y3*y1 + z3*z1) - radius**2

    i = b * b - 4.0 * a * c

    if i < 0.0:
        # No intersection
        return []
    elif i == 0.0:
        # Tangent intersection
        mu = -b/2.0*a
        return [(x1 + mu * (x2-x1), y1 + mu * (y2-y1), z1 + mu * (z2-z1))]
    elif i > 0.0:
        # Two intersection points
        intersects = []
        mu = (-b + np.sqrt(i))/2.0*a
        intersects.append((x1 + mu * (x2-x1), y1 + mu * (y2-y1), z1 + mu * (z2-z1)))
        mu = (-b - np.sqrt(i))/2.0*a
        intersects.append((x1 + mu * (x2-x1), y1 + mu * (y2-y1), z1 + mu * (z2-z1)))
        return intersects

def distance(xyz1, xyz2):
    x1,y1,z1 = xyz1
    x2,y2,z2 = xyz2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def equal(xyz1, xyz2, tol=1e-5):
    x1,y1,z1 = xyz1
    x2,y2,z2 = xyz2

    if abs(x1-x2) < tol and abs(y1-y2) < tol and abs(z1-z2) < tol:
        return True
    else:
        return False

def satellite_graph(coord, earth_radius):
    """ Create graph for satellites that have un-obstructed view to each other and call locations.

    Args:
    -----
    coord     -- A dict of satellite coordinates (Cartesian) including call locations with names 'Source and 'Target'.

    Returns:
    --------
    A dict of dicts graph representation of the connections between satellites (and call locations) with distances.

    """
    nsats = len(coord)-2
    graph = {}

    for key in coord.keys():
        graph[key] = {}

    for key1 in coord.keys():
        for key2 in coord.keys():
            if key1 != key2:
                intersect = line_sphere_intersect(earth_radius, coord[key1], coord[key2])
                if key1 == 'Source' or key1 == 'Target':
                    if len(intersect) > 0:
                        if equal(intersect[0], coord[key1]):
                            dist = distance(coord[key1], coord[key2])
                            graph[key1][key2] = dist
                            graph[key2][key1] = dist
                else:
                    if len(intersect) == 0:
                        graph[key1][key2] = distance(coord[key1], coord[key2])

    # Remove nodes with no vertices
    graph = dict((k, v) for k, v in graph.items() if v)

    if not 'Source' in graph.keys() or not 'Target' in graph.keys():
        raise ValueError("There is no path from source to target! (maybe increase tolerance for the equal'-function)")

    return graph

def min_dist_vertex(Q, dist):

    min_vertex = Q[0]

    for vertex in Q[1:len(Q)]:
        if dist[vertex] < dist[min_vertex]:
            min_vertex = vertex

    return min_vertex

def dijkstra(graph, source, target):
    """ The Dijkstra algorithm for finding the shortest path between two nodes in an undirected weighted graph

    Args:
    -----
    graph       --  A dict of dicts representation of the graph with weights (distances).
    source      --  Name of the starting node
    destination --  Name of the destination node

    Returns:
    --------
    A list with node names ordered to give the shortest path from source to destination
    """
    dist = dict() # Distance from source to vertex v
    prev = dict() # Previous node in the optimal path from source

    vertices = list(graph.keys())
    for vertex in vertices:
        dist[vertex] = float('Inf')
        prev[vertex] = None

    dist[source] = 0.0
    Q = vertices

    while len(Q) > 0:
        u = min_dist_vertex(Q,dist)
        if u == target or dist[u] == float('Inf'):
            break
        Q.remove(u)

        for neighbour in graph[u]:
            alt = dist[u] + graph[u][neighbour]
            if alt < dist[neighbour]:
                dist[neighbour] = alt
                prev[neighbour] = u

    # Recover the optimal path
    path = []
    u = target
    while prev[u]:
        path.insert(0,u)
        u = prev[u]
    path.insert(0,u) # Source

    return path

def plot_solution(graph, coord, path):

    # Init plotting
    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    # Plot colors
    ax.set_axis_bgcolor('#f4f4f8')
    sat_col = '#086788'
    link_col = '#118ab2'
    source_col = '#f038ff'
    target_col = '#06d6a0'
    opt_path_col = '#ef709d'

    # Plot earth
    plot_earth(ax)

    # Plot the satellites and call locations
    for key, val in coord.items():
        x,y,z = val
        if key is 'Source':
            plot_point(x, y, z, ax, col=source_col, size=100)
        elif key is 'Target':
            plot_point(x, y, z, ax, col=target_col, size=100)
        else:
            plot_point(x, y, z, ax, col=sat_col)

    # Plot the graph edges
    for key1 in graph.keys():
        for key2 in graph[key1].keys():
            plot_line(coord[key1], coord[key2], ax, col=link_col)

    # Plot the path
    for i in range(len(path)-1):
        plot_line(coord[path[i]], coord[path[i+1]], ax, col=opt_path_col, width=2.0)

    plt.axis('off')
    plt.show()

def solve_challenge(filename='data.csv'):

    earth_radius = 6371.0

    # Read in data
    sats = pd.read_csv(filename, header=None, skiprows=[0,21], names=['ID','lat','long','alt'])
    call_coords = pd.read_csv(filename, header=None, skiprows=range(21), names=['lat1', 'long1', 'lat2', 'long2'])

    # Transform from lat and long to Cartesian coordinates
    coord = {}
    for i,sat in sats.iterrows():
        coord[sat['ID']] = latlon_to_cartesian(earth_radius + sat['alt'], sat['lat'], sat['long'])

    # Add call locations
    coord['Source'] = latlon_to_cartesian(earth_radius, call_coords['lat1'].values[0], call_coords['long1'].values[0])
    coord['Target'] = latlon_to_cartesian(earth_radius, call_coords['lat2'].values[0], call_coords['long2'].values[0])

    # Create the graph (connections between satellites and locations that have clear view to each other) with distances as edges
    graph = satellite_graph(coord, earth_radius)

    # Calculate the shortest path from source to target
    shortest_path = dijkstra(graph, 'Source', 'Target')

    # Print the path
    path_str = "{}".format(shortest_path[1:len(shortest_path)-1]).replace("'","")
    print("Shortest path: {}".format(path_str))

    # Plot
    plot_solution(graph, coord, shortest_path)

solve_challenge('data3.csv')
