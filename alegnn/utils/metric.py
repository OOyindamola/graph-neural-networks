# Implement Cluster and Radial Metrics
# -*- coding: utf-8 -*-

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from shapely.validation import make_valid
# from numba import jit
"""
- O Algoritmo é bem simples para contar os clusters, mas testei ele bastante e me pareceu confiável;
Basicamente, ele itera para cada robô do mesmo tipo de forma a achar um caminho de passo menor ou
igual a dAA (distancia minima entre robos do mesmo tipo). Primeiramente, removo um robô na lista e
adiciono em uma lista de cluster e busco o vizinho mais proximo. Caso ele satisfaça a condição de
distancia minima, ele é adicionando a lista de cluster, e dentre os robôs da lista, busco o proximo
vizinho mais perto, e repito o processo até remover todos os robos da lista; caso ele não satisfaça
a condição de distancia minima eu fecho a lista de cluster e crio um nova e continuo o processo.
"""


class NClusterMetric(object):
    """    This class implement a metric to count the number of clusters"""

    def __init__(self, GROUPS, ROBOTS, threshold):
        self.GROUPS = GROUPS
        self.ROBOTS = ROBOTS
        self.threshold = threshold

    def compute(self, q):
        # Compute for each combination of groups of robots the intersection area and accumulate it
        clusters = []
        n_groups = self.ROBOTS/self.GROUPS
        for i in range(self.GROUPS):
            idx_i = (int(math.floor((i) * self.ROBOTS/self.GROUPS)),
                     int(math.floor((i+1) * self.ROBOTS/self.GROUPS)))
            qi = q[idx_i[0]:idx_i[1]]  # robots of group i
            cluster = []
            cluster.append(qi[0, :])
            qi = np.delete(qi, 0, axis=0)
            while(qi.shape[0]):
                for j in range(len(cluster)):
                    mink = 0
                    minDist = 10000000000
                    for k in range(len(qi)):
                        dist = self.euclidean(cluster[j], qi[k])
                        if minDist > dist:
                            minDist = dist
                            mink = k
                if minDist < self.threshold:
                    cluster.append(qi[mink])
                    qi = np.delete(qi, mink, axis=0)
                else:
                    clusters.append(cluster)
                    cluster = []
                    cluster.append(qi[0])
                    qi = np.delete(qi, 0, axis=0)
            clusters.append(cluster)

        return len(clusters)

    def euclidean(self, x, y):
        dx = x[0] - y[0]
        dy = x[1] - y[1]
        # print("x: ", x.shape)
        if x.shape[0] ==3:
            dz = x[2] - y[2]
            return np.sqrt(dx*dx + dy*dy +dz*dz)
        return np.sqrt(dx*dx + dy*dy)


class AverageMetric(object):
    """    This class implement a metric to evaluate the average distance between the robots"""

    def __init__(self, GROUPS, ROBOTS):
        self.GROUPS = GROUPS
        self.ROBOTS = ROBOTS
        x = np.array(range(1, self.ROBOTS+1))
        i, j = np.meshgrid(x, x)
        gpr = float(self.GROUPS)/float(self.ROBOTS)
        self.AA = (np.floor(gpr*(i-1.0)) == np.floor(gpr*(j-1.0)))*1.0
        self.AB = (np.floor(gpr*(i-1.0)) != np.floor(gpr*(j-1.0)))*1.0



    def compute(self, q):
        # Compute for each combination of groups of robots the intersection area and accumulate it

        # print(q.shape)

        if q.shape[1] ==3:
            dq = np.repeat(q, self.ROBOTS, axis=0).reshape(
                (self.ROBOTS, self.ROBOTS, 3)) - q
            dist = np.sqrt(np.power(dq[:, :, 0], 2) + np.power(dq[:, :, 1], 2) + np.power(dq[:, :, 2], 2))
        else:
            dq = np.repeat(q, self.ROBOTS, axis=0).reshape(
                (self.ROBOTS, self.ROBOTS, 2)) - q
            dist = np.sqrt(np.power(dq[:, :, 0], 2) + np.power(dq[:, :, 1], 2))
        n = self.GROUPS * (self.ROBOTS/self.GROUPS) **2

        # M = int(self.ROBOTS/self.GROUPS)
        # dig = [x[i*M:(i+1)*M,i*M:(i+1)*M].flatten().mean() for i in range(int(x.shape[0]/M))]


        return [(self.AA * dist).mean(), (self.AB * dist).mean()]

    def euclidean(self, x, y):
        dx = x[0] - y[0]
        dy = x[1] - y[1]
        # print(x.shape)
        if x.shape[0] ==3:
            dz = x[2] - y[2]
            return np.sqrt(dx*dx + dy*dy +dz*dz)
        return np.sqrt(dx*dx + dy*dy)

class ClusterMetric(object):
    """    This class implement a metric to evaluate the pairwise intersection area
    between the convex hulls of the robots acoording to their type partition.
    That is, the convex hull of all robots of type k is computed and
    intersected with each other. """

    def __init__(self, GROUPS, ROBOTS):
        self.GROUPS = GROUPS
        self.ROBOTS = ROBOTS

    def compute(self, q, DEAD_ROBOTS=[]):
        # Compute for each combination of groups of robots the intersection area and accumulate it
        mclu = 0.0
        for i in range(self.GROUPS):
            idx_i = (int(math.floor((i) * self.ROBOTS/self.GROUPS)),
                     int(math.floor((i+1) * self.ROBOTS/self.GROUPS)))
            qi = []
            for k in range(idx_i[0], idx_i[1]):
                if not k in DEAD_ROBOTS:
                    qi.append(q[k, :])
            qi = np.array(qi)
            #qi = q[idx_i[0]:idx_i[1]]
            for j in range(self.GROUPS):
                idx_j = (int(math.floor((j) * self.ROBOTS/self.GROUPS)),
                         int(math.floor((j+1) * self.ROBOTS/self.GROUPS)))
                #qj = q[idx_j[0]:idx_j[1]]
                qj = []
                for k in range(idx_j[0], idx_j[1]):
                    if not k in DEAD_ROBOTS:
                        qj.append(q[k, :])
                qj = np.array(qj)
                if idx_i != idx_j:
                    mclu += self.compute_area(qi, qj)
        return mclu

    # def feature(self):
    # Return the sum of area of all combinations of intersection between convex hulls formed by group of robots.
    #    return self.mclu

    def compute_area(self, A, B):
        # Compute the area between two polygons formed by two convex hull.
        # Evaluate a convex hull for two groups of robots

        A_hull = ConvexHull(A)
        B_hull = ConvexHull(B)


        # Get convex hull points
        A_hull = A[A_hull.vertices]
        B_hull = B[B_hull.vertices]

        # Transform points to polygon
        A_poly = make_valid(Polygon(A_hull))#.buffer(0)
        B_poly = make_valid(Polygon(B_hull))#.buffer(0)

        # Compute the intersections points between the two polygons
        AB_inter = A_poly.intersection(B_poly)

        # Then return the area of the intersection
        return AB_inter.area


class RadialMetric(object):
    """This class implement Gross et al. (2009) metric.
    Their metric requires robots to segregate around a particulary stationary point c."""

    def __init__(self, GROUPS, ROBOTS, const):
        self.GROUPS = GROUPS
        self.ROBOTS = ROBOTS
        self.const = const

    def compute(self, q):
        # This method implement the radial segregation metric. It counts the number of robots outrside the
        # correct spherical shells according to the values of each dAA_k and normalizes the result into the unit interval.
        self.mrad = 0.0
        self.q = q
        # Compute the stationary point c
        self.n = float(len(q))
        self.c = np.sum(self.q, axis=0)/self.n

        # for i in range(self.ROBOTS):
        #    for j in range(self.ROBOTS):
        #        self.mrad += self.radial_error(i, j)

        self.mrad = fast(self.ROBOTS, self.const, self.q, self.c)
        self.mrad /= (self.n**2)
        return self.mrad

    def feature(self):
        # Return the feature of the current segregation state
        return self.mrad

    def radial_error(self, i, j):
        # Apply a cost when robots which sould be "inside" the other and closer to the stationary point
        dAAk = self.const[i, i]
        dAAl = self.const[j, j]
        if dAAk != dAAl:  # if dissimilar groups
            if dAAk < dAAl and self.distToC(i) >= self.distToC(j):
                return 1.0
            if dAAk > dAAl and self.distToC(i) < self.distToC(j):
                return 1.0
        return 0.0

    # Compute the Euclidean distance from robot r to the stationary point c
    def distToC(self, r):
        d = self.q[r] - self.c
        dist = np.sqrt(d[0]**2 + d[1]**2)
        return dist


# @jit
def fast(ROBOTS, const, q, c):
    mrad = 0.0
    for i in range(ROBOTS):
        for j in range(ROBOTS):
            mrad += radial_error(i, j, const, q, c)
    return mrad


# @jit
def radial_error(i, j, const, q, c):
    # Apply a cost when robots which sould be "inside" the other and closer to the stationary point
    dAAk = const[i, i]
    dAAl = const[j, j]
    if dAAk != dAAl:  # if dissimilar groups
        if dAAk < dAAl and distToC(i, q, c) >= distToC(j, q, c):
            return 1.0
        if dAAk > dAAl and distToC(i, q, c) < distToC(j, q, c):
            return 1.0
    return 0.0


# @jit
def distToC(r, q, c):
    d = q[r] - c
    dist = np.sqrt(d[0]**2 + d[1]**2)
    return dist
