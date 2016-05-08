import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

earth_radius = 6371.0
#earth_radius = 6378.1370

# Init plotting
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

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

def plot_earth():
    theta, phi = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x,y,z = spherical_to_cartesian(earth_radius, theta, phi)
    return ax.plot_surface(x, y, z, color="b", alpha=0.2, rstride=1, cstride=1, linewidth=0)

def plot_point(x,y,z, col='r', size=50):
    return ax.scatter([x],[y],[z],c=col,s=size)

def plot_line(xyz1, xyz2, col='r'):
    x1,y1,z1 = xyz1
    x2,y2,z2 = xyz2
    return ax.plot([x1,x2],[y1,y2],[z1,z2],c=col)

def line_sphere_intersect(radius, xyz1, xyz2, xyz3=(0.0,0.0,0.0)):
    """ Calculates whether a line and a sphere intersect

    Assumes the sphere is centered at origo.

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

def satellite_graph(coord):
    """ Create graph for which satellites have un-obstructed view to each other

    Args:
    -----
    coord -- A list of satellite coordinates (Cartesian) with call locations in the last two spots.

    Returns:
    --------
    A list of lists graph representation of the connections between satellites (and call locations)

    """
    nsats = len(coord)
    conn = []

    # For satellites
    for i in range(nsats-2):
        sati_conn = []
        for j in range(nsats-2):
            if i != j:
                intersect = line_sphere_intersect(earth_radius, coord[i], coord[j])
                if len(intersect) == 0:
                    sati_conn.append(j)
        conn.append(sati_conn)

    # For call locations check if the first intersection point is at call location
    for i in range(nsats-2,nsats):
        calli_conn = []
        for j in range(nsats-2):
            intersect = line_sphere_intersect(earth_radius, coord[i], coord[j])
            if len(intersect) == 2:
                # print("{}\n{}".format(intersect,coord[i]))
                if equal(intersect[0], coord[i]):
                    calli_conn.append(j)
                    conn[j].append(i)
        conn.append(calli_conn)
    return conn

# Read in data
sats = pd.read_csv("data2.csv", header=None, skiprows=[0,21], names = ['ID','lat','long','alt'])
call_coords = pd.read_csv("data2.csv", header=None, skiprows=range(21), names=['lat1', 'long1', 'lat2', 'long2'])

# Transform lat and long to Cartesian coordinates
coord = [latlon_to_cartesian(earth_radius + sat['alt'], sat['lat'], sat['long']) for i,sat in sats.iterrows()]

# Add call locations
coord.append(latlon_to_cartesian(earth_radius, call_coords['lat1'].values[0], call_coords['long1'].values[0]))
coord.append(latlon_to_cartesian(earth_radius, call_coords['lat2'].values[0], call_coords['long2'].values[0]))

# Create graph connections and calculate distances
conn = satellite_graph(coord)

# Calculate distances
dist = []
for i in range(len(coord)):
    dist.append([distance(coord[i], coord[j]) for j in conn[i]])

# Plot earth
plot_earth()

# Plot the satellites
for i in range(len(conn)-2):
    x,y,z = coord[i]
    plot_point(x,y,z)

# Plot call coordinates
x1,y1,z1 = coord[len(coord)-2]
x2,y2,z2 = coord[len(coord)-1]
plot_point(x1,y1,z1, col='yellow')
plot_point(x2,y2,z2, col='orange')

# Plot connections between satellites that have clear view to each other
for i in range(len(conn)):
    for j in conn[i]:
        plot_line(coord[i], coord[j])

plt.axis('off')
plt.show()
