#
#  Solution to the Reaktor Orbital Challenge
#    https://reaktor.com/orbital-challenge/
#
#  Copyright (C) 2016  Jarno Kiviaho <jarkki@kapsi.fi>
#
#
import sys
import numpy as np

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

def plot_earth(ax, radius = 6371.0, col='#fff1d0'):
    """ Plot earth as a perfect sphere """
    theta, phi = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x,y,z = spherical_to_cartesian(radius, theta, phi)
    return ax.plot_surface(x, y, z, alpha=0.05, color=col, rstride=1, cstride=1, linewidth=1, shade=1)

def plot_point(x,y,z, ax, col='#086788', size=50):
    return ax.scatter([x], [y], [z], c=col, s=size, edgecolor='none')

def plot_line(xyz1, xyz2, ax, col='r', width=1.0):
    x1,y1,z1 = xyz1
    x2,y2,z2 = xyz2
    return ax.plot([x1,x2], [y1,y2], [z1,z2], c=col, linewidth=width)

def line_sphere_intersect(radius, xyz1, xyz2, xyz3=(0.0,0.0,0.0)):
    """ Calculates whether a line and a sphere intersect

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

    dx = x2-x1
    dy = y2-y1
    dz = z2-z1

    a = dx**2 + dy**2 + dz**2
    b = 2.0 * (dx*(x1-x3) + dy*(y1-y3) + dz*(z1-z3))
    c = x3**2 + y3**2 + z3**2 + x1**2 + y1**2 + z1**2 - 2.0*(x3*x1 + y3*y1 + z3*z1) - radius**2

    i = b**2 - 4.0 * a * c

    if i > 0.0:
        # The intersection nearest to point xyz1
        mu = (-b - np.sqrt(i))/(2.0 * a)
        inter1 = (x1 + mu * dx, y1 + mu * dy, z1 + mu * dz)
        # The other intersection
        mu = (-b + np.sqrt(i))/(2.0 * a)
        inter2 = (x1 + mu * dx, y1 + mu * dy, z1 + mu * dz)
        return [inter1,inter2]
    elif i == 0.0:
        # Tangent intersection
        mu = -b/(2.0 * a)
        return [(x1 + mu * dx, y1 + mu * dy, z1 + mu * dz)]
    else:
        # No intersection
        return []

def distance(xyz1, xyz2):
    """ Distance between two points in 3D-space"""
    x1,y1,z1 = xyz1
    x2,y2,z2 = xyz2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def equal(xyz1, xyz2, tol=1e-8):
    """ Checks if two points in 3D-space have the same location, within tolerance."""
    if distance (xyz1, xyz2) < tol:
        return True
    else:
        return False

def satellite_graph(coord, earth_radius):
    """ Create graph for satellites and call locations.

    Two satellites are connected if there is an un-obstructed view from one to another. Edge weights are the distances between satellites. Call locations are included in the graph.

    Args:
    -----
    coord        -- A dict of satellite coordinates (Cartesian) including call locations with names 'Source and 'Target'.
    earth_radius -- The radius of earth.

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
            if key1 not in ['Source', 'Target', key2]:
                # Check if the line created by the two points (satellites/call locations) intersects the planet sphere
                intersect = line_sphere_intersect(earth_radius, coord[key1], coord[key2])
                if key2 == 'Source' or key2 == 'Target':
                    if len(intersect) >= 0:
                        # Two intersections or one tangent.
                        # Check if the intersection closest to key1 is the call location
                        # If so, the satellite has un-obstructed view to the call location.
                        if equal(intersect[0], coord[key2]):
                            # Un-obstructed view
                            dist = distance(coord[key1], coord[key2])
                            graph[key1][key2] = dist
                            graph[key2][key1] = dist
                else:
                    if len(intersect) == 0:
                        # No intersection, two satellites have un-obstructed view
                        graph[key1][key2] = distance(coord[key1], coord[key2])

    # Remove nodes with no connections
    graph = dict((k, v) for k, v in graph.items() if v)

    if not 'Source' in graph.keys() or not 'Target' in graph.keys():
        raise ValueError("There is no path from source to target!")

    return graph

def min_dist_vertex(Q, dist):
    """ Search for the vertex with the minimum distance from source, in vertices given in Q.

    Args:
    -----
    Q     --  List of vertice names.
    dist  --  Dict, representing the distance from source to vertex v.

    Returns:
    --------
    The name of the vertex with the minimum distance from source.

    """
    min_vertex = Q[0]

    for vertex in Q[1:len(Q)]:
        if dist[vertex] < dist[min_vertex]:
            min_vertex = vertex

    return min_vertex

def dijkstra(graph, source, target):
    """ The Dijkstra algorithm for finding the shortest path between two vertices in an undirected weighted graph.

    Args:
    -----
    graph    --  A dict of dicts representation of the graph with weights (distances).
    source   --  Name of the starting vertex
    target   --  Name of the destination vertex

    Returns:
    --------
    An ordered list with node names to give the shortest path from source to target
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
        if u == target:
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

def plot_solution(graph, coord, path, col):
    """ Use matplotlib to plot the solution

    Note: the plot blocks the script execution until the it is closed.

    Args:
    -----
    graph  --  Dict of dicts graph representation
    coord  --  Coordinates for each satellite and call location
    path   --  The shortest path from call source to target
    col    --  Dict of colors for plotting
    """

    # Check if matplotlib is installed
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError as e:
        print(e)
        print("Matplotlib not installed, can't plot.")
        return

    # Init plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    ax.set_axis_bgcolor(col['bg'])

    # Plot earth
    plot_earth(ax)

    # Plot the satellites and call locations
    for key, val in coord.items():
        x,y,z = val
        if key is 'Source':
            plot_point(x, y, z, ax, col=col['source'], size=100)
        elif key is 'Target':
            plot_point(x, y, z, ax, col=col['target'], size=100)
        else:
            plot_point(x, y, z, ax, col=col['sat'])

    # Plot the graph edges
    for key1 in graph.keys():
        for key2 in graph[key1].keys():
            plot_line(coord[key1], coord[key2], ax, col=col['link'])

    # Plot the path
    for i in range(len(path)-1):
        plot_line(coord[path[i]], coord[path[i+1]], ax, col=col['opt_path'], width=2.0)

    plt.axis('off')
    plt.show()

def dl_data(url='http://space-fast-track.herokuapp.com/generate',
            filename='data.csv'):
    """ Downloads the problem data and writes to csv file.

    Does not handle exceptions.

    Args:
    -----
    url       --  The url for the data
    filename  --  Name of the csv file to write to.

    """
    # Handle different urllib implementation for python 2 and 3
    py3 = sys.version_info[0] == 3
    if py3:
        import urllib.request
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
    else:
        import urllib
        response = urllib.urlopen(url)
        data = response.read()

    file = open(filename, 'w')
    file.write(data)
    file.close()

def read_data(filename='data.csv'):
    """ Read csv data from file

    Does not handle exceptions.

    Args:
    -----
    filename  --  Name of the csv file.

    Returns:
    --------
    Dictionary with the seed(float), call locations(dict) and the satellite locations(list(dict))

    """
    # Open file and read the data
    file = open(filename, 'r')
    data = file.read()
    file.close()

    # Split the rows
    rows = data.split('\n')

    # Seed
    seed = float(rows[0].split(':')[1])

    # Satellite data
    sats = []
    for row in rows[1:len(rows)-1]:
        id, lat, lon, alt = row.split(',')
        sats.append({'ID': id, 'lat': float(lat), 'lon': float(lon), 'alt': float(alt)})

    # Call location data
    _, lat1, lon1, lat2, lon2 = rows[len(rows)-1].split(',')
    call_loc = {'lat1': float(lat1), 'lon1': float(lon1), 'lat2': float(lat2), 'lon2': float(lon2)}

    res = {'sats': sats, 'seed': seed, 'call_loc': call_loc}
    return res

def solve_challenge(dl_new_data=False):

    earth_radius = 6371.0

    # Plot colors
    plot_col = {'bg'       : '#f4f4f8',
                'sat'      : '#086788',
                'link'     : '#118ab2',
                'source'   : '#f038ff',
                'target'   : '#06d6a0',
                'opt_path' : '#ef709d'}

    if dl_new_data:
        dl_data()    # Download new dataset and save to 'data.csv'

    # Read in data from 'data.csv'
    data = read_data()

    # Transform from lat and long to Cartesian coordinates
    coord = {}
    for sat in data['sats']:
        coord[sat['ID']] = latlon_to_cartesian(earth_radius + sat['alt'], sat['lat'], sat['lon'])

    # Add source and target call locations
    coord['Source'] = latlon_to_cartesian(earth_radius, data['call_loc']['lat1'], data['call_loc']['lon1'])
    coord['Target'] = latlon_to_cartesian(earth_radius, data['call_loc']['lat2'], data['call_loc']['lon2'])

    # Create the graph (connections between satellites (and locations) that have clear view to each other) with distances as edges
    graph = satellite_graph(coord, earth_radius)

    # Calculate the shortest path from source to target
    shortest_path = dijkstra(graph, 'Source', 'Target')

    # Print the path and seed
    path_str = "{}".format(shortest_path[1:len(shortest_path)-1]).replace("'","")
    print("Shortest path: {}, seed: {}".format(path_str, data['seed']))

    # Plot
    plot_solution(graph, coord, shortest_path, plot_col)

solve_challenge(dl_new_data=False)
