# Este fichero deriva del paper1 con la intención de implementar el algoritmo genético
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import matplotlib.colors as colors
import matplotlib.colors as mcolors
import random
from collections import namedtuple
#import traceback
import math
from enum import IntEnum
import json
import time
import copy
import multiprocessing
import os
import re
import argparse
from typing import List, Tuple, Callable
import itertools
from functools import partial
import datetime
#from scipy.optimize import minimize
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
import gc
#import psutil
#import pickle
#import cProfile
#import pstats
#import h5py



class Parameters:
	"""
	Parameters of the simulation

	Attributes:
		verticalBlocks (int): Number of vertical blocks
		horizontalBlocks (int): Number of horizontal blocks
		numberCarsPerBlock (int): Number of cars per block
		numberStations (int): Number of charging stations
		numberChargingPerStation (int): Number of charging per station
		carMovesFullDeposity (int): Number of moves when the car is full
		carRechargePerTic (int): Number of moves that the car recharge per tic
		opmitimizeCSSearch (int): Number of charging stations to store in bifurcation cell to optimize the search
		viewDrawCity (bool): If true, draw the city
	"""
	def __init__(self):
		self.verticalBlocks=3
		self.horizontalBlocks=3
		self.numberBlocks=self.verticalBlocks*self.horizontalBlocks
		self.numberStationsPerBlock=1# tipical 1/(numberBlocks), 1, 4
		self.yellowBox=True

		self.numberStations=self.numberStationsPerBlock*self.numberBlocks
		
		self.densityEV=0.5
		self.densityDiesel=0.1
		#self.densityPetrol=1-self.densityEV-self.densityDiesel
		self.buildings=True
		self.distributionCS=1
		self.windV=(0.1,0)
		self.pollutionRouting = False
		self.ring = True

		self.numberChargersPerBlock=1

		self.densityCars=0.1
		self.carMovesFullDeposity=27000
		self.stepsToRecharge=960 
		self.carRechargePerTic=self.carMovesFullDeposity/self.stepsToRecharge

		self.introduceCarsInCSToStacionaryState=True

		self.timeStop=10

		# A* optimization
		self.opmitimizeCSSearch=3 # bigger is more slow
		self.aStarDeep=100 # Number of positions to search in the aStar algorithm
		self.aStarRemainderWeight=2 #* weight of lineal distance to target to time

		# A* optimization Time
		self.aStarCalculateEach=10 # The aStar calculation is slow, so we can calculate it each n bifurcation cells
		self.aStarTimeOutCacheBifulcation=10 # In the bifulcation cell the calculos of time to CS is valid for n tics

		self.carSizeTarget=20 # The target of each car is a secuence of random cells. This parameter is the size of the secuence

		self.aStarMethod="Time" # Time or Distance
		self.aStarRandomCS=False
		self.aStarCSQueueQuery=0. # HACK percentage of EV than use the web to see the queue of the CS (time)
		self.aStarCSReserve=0. # HACK percentage of EV than reserve a slot OF THE CSQUEUEQUERY 

		# when aStarMethod is Time
		self.aStarAddRoadCarAsTimeSteps=0
		self.aStarUseCellAverageVelocity=True # false=time of the street. works in combination with aStarUseCellExponentialWeight
		self.aStarUseCellExponentialWeight=0.95 #* 0 disable, 0-1 weight of old velocity data
		# self.aStarStepsPerCar=100000 # bigger is more slow, more precision

		# interface parameters
		self.viewWarning = True
		self.viewDrawCity = False
		#self.statsFileName="data/stats_" # paper1
		self.statsFileName="paper2/stats_" 
		self.metastatsFileName="paper2/metastats/"
		self.dataSave="simulationData32Last"


	def clone(self):
		"""
		Creates a deep copy of the Parameters object.
		"""
		return copy.deepcopy(self)
	
	def metaExperiment(self,**m):
		"""
		Take a map [parameter] -> [values] and generte al cartesian product of the values.
		"""
		# generate an array of index
		index=[0]*len(m)
		keys=list(m.keys())
		r=[] # result
		end=False
		while True:
			# r.append(index.copy())
			p=self.clone()
			p.fileName=""
			p.legendName=""
			for i in range(len(index)):
				setattr(p,keys[i],m[keys[i]][index[i]])
				p.fileName=p.fileName+keys[i]+str(m[keys[i]][index[i]])+"_"
				p.legendName=p.legendName+keys[i]+":"+str(m[keys[i]][index[i]])+" "
			# remove last character
			p.fileName=p.fileName[:-1]
			p.legendName=p.legendName[:-1]
			r.append(p)

			i=0 # index to increment
			while True:
				if i==len(index):
					end=True
					break
				index[i]+=1
				# if reseach the end of the array
				size=len(m[keys[i]])
				if index[i]==size:
					index[i]=0
					i+=1
				else:
					break
			if end:
				break
		return r

class CarPriority(IntEnum):
	'''
	CarPriority is used to define the priority of the execution in the grid.
	Our Cellular Automata is asynchronous. Some cells (with cars) are executed before than others.
	'''
	StopedNoPriority = -2
	StopedPriority = -1
	NoAsigned=0
	Priority = 1
	NoPriority = 2

class CarState(IntEnum):
	'''
	CarState is used to evaluate the efficiency of the ubication of the charging stations.
	'''
	Destination = 0
	Waiting = 1
	Driving = 2
	Charging = 3
	Queueing = 4
	ToCharging = 5

class CarType(IntEnum):
	'''
	The EV type go to charging station when the battery is low. 
	'''
	EV = 0
	Petrol = 1
	Diesel = 2
'''
poll_coefs = {
	'CO2': {
		CarType.Petrol: [0, 0.553, 0.161, -0.00289, 0.266, 0.511, 0.183],
		CarType.Diesel: [0, 0.324, 0.0859, 0.00496, -0.0586, 0.448, 0.23],
		CarType.EV: [0, 0, 0, 0, 0, 0, 0]
	},
	'NOx': {
		CarType.Petrol: [0, 0.000619, 8e-05, -4.03e-06, -0.000413, 0.00038, 0.000177],
		CarType.Diesel: [0, 0.00241, -0.000411, 6.73e-05, -0.00307, 0.00214, 0.0015],
		CarType.EV: [0, 0, 0, 0, 0, 0, 0]
	},
	'NOxdecel': {
		CarType.Petrol: [0, 2.17e-4, 0, 0, 0, 0, 0],
		CarType.Diesel: [0, 1.68e-03, -6.62e-05, 9.00e-06, 2.50e-04, 2.91e-04, 1.20e-04],
		CarType.EV: [0, 0, 0, 0, 0, 0, 0]
	},
	'VOC': {
		CarType.Petrol: [0, 0.00447, 7.32e-07, -2.87e-08, -3.41e-06, 4.94e-06, 1.66e-06],
		CarType.Diesel: [0, 9.22e-05, 9.09e-06, -2.29e-07, -2.2e-05, 1.69e-05, 3.75e-06],
		CarType.EV: [0, 0, 0, 0, 0, 0, 0]
	},
	'VOCdecel': {
		CarType.Petrol: [0, 2.63e-3, 0, 0, 0, 0, 0],
		CarType.Diesel: [0, 5.25e-05, 7.22e-06, -1.87e-07, 0, -1.02e-05, -4.22e-06],
		CarType.EV: [0, 0, 0, 0, 0, 0, 0]
	},
	'PMexhaust': {
		CarType.Petrol: [0, 0.0, 1.57e-05, -9.21e-07, 0.0, 3.75e-05, 1.89e-05],
		CarType.Diesel: [0, 0.0, 0.000313, -1.84e-05, 0.0, 0.00075, 0.000378],
		CarType.EV: [0, 0, 0, 0, 0, 0, 0]
	},
	'PMexhaustprueba': {
		CarType.Petrol: [0, 0, 3.1*0.000001, 0, 0, 0, 0],#*cellsize*timestepvalue
		CarType.Diesel: [0, 0, 2.4*0.000001, 0, 0, 0, 0],
		CarType.EV: [0, 0, 0, 0, 0, 0, 0]
	},
	'PMnonexhaust25': {
		CarType.Petrol: [0, 0, (23.2-3)*0.000001, 0, 0, 0, 0],
		CarType.Diesel: [0, 0, (22.6-2.4)*0.000001, 0, 0, 0, 0],
		CarType.EV: [0, 0, 22.4*0.000001, 0, 0, 0, 0]
	},
	'PMnonexhaust10': {
		CarType.Petrol: [0, 0, (66-3.1)*0.000001, 0, 0, 0, 0],
		CarType.Diesel: [0, 0, (65.3-2.4)*0.000001, 0, 0, 0, 0],
		CarType.EV: [0, 0, 65.7*0.000001, 0, 0, 0, 0]
	}
}
'''

poll_coefs = {
    CarType.Petrol: {
        'CO2': [0, 0.553, 0.161, -0.00289, 0.266, 0.511, 0.183],
        'NOx': [0, 0.000619, 8e-05, -4.03e-06, -0.000413, 0.00038, 0.000177],
        'NOxdecel': [0, 2.17e-4, 0, 0, 0, 0, 0],
        'VOC': [0, 0.00447, 7.32e-07, -2.87e-08, -3.41e-06, 4.94e-06, 1.66e-06],
        'VOCdecel': [0, 2.63e-3, 0, 0, 0, 0, 0],
        'PMexhaust': [0, 0.0, 1.57e-05, -9.21e-07, 0.0, 3.75e-05, 1.89e-05],
        'PMexhaustprueba': [0, 0, 3.1*0.000001, 0, 0, 0, 0],
        'PMnonexhaust25': [0, 0, (23.2-3)*0.000001, 0, 0, 0, 0],
        'PMnonexhaust10': [0, 0, (66-3.1)*0.000001, 0, 0, 0, 0]
    },
    CarType.Diesel: {
        'CO2': [0, 0.324, 0.0859, 0.00496, -0.0586, 0.448, 0.23],
        'NOx': [0, 0.00241, -0.000411, 6.73e-05, -0.00307, 0.00214, 0.0015],
        'NOxdecel': [0, 1.68e-03, -6.62e-05, 9.00e-06, 2.50e-04, 2.91e-04, 1.20e-04],
        'VOC': [0, 9.22e-05, 9.09e-06, -2.29e-07, -2.2e-05, 1.69e-05, 3.75e-06],
        'VOCdecel': [0, 5.25e-05, 7.22e-06, -1.87e-07, 0, -1.02e-05, -4.22e-06],
        'PMexhaust': [0, 0.0, 0.000313, -1.84e-05, 0.0, 0.00075, 0.000378],
        'PMexhaustprueba': [0, 0, 2.4*0.000001, 0, 0, 0, 0],
        'PMnonexhaust25': [0, 0, (22.6-2.4)*0.000001, 0, 0, 0, 0],
        'PMnonexhaust10': [0, 0, (65.3-2.4)*0.000001, 0, 0, 0, 0]
    },
    CarType.EV: {
        'PMnonexhaust25': [0, 0, 22.4*0.000001, 0, 0, 0, 0],
        'PMnonexhaust10': [0, 0, 65.7*0.000001, 0, 0, 0, 0]
    }
}

for car_type, pollutants in poll_coefs.items():
    for pollutant, values in pollutants.items():
        poll_coefs[car_type][pollutant] = [np.float32(v) if v != 0 else 0 for v in values]


class CarProperties:
	def __init__(self,typeCar,needsCharging):
		self.typeCar = typeCar
		self.needsCharging = needsCharging
		self.density = 0
		self.number = 0
		self.contaminationCoef = poll_coefs[self.typeCar]
		#AÑADIR CONTAMINACIÓN AQUÍ

CAR_PROPERTIES = {CarType.EV: CarProperties(CarType.EV,True), CarType.Petrol: CarProperties(CarType.Petrol,False), CarType.Diesel: CarProperties(CarType.Diesel,False)}

class ChargingStation:
	"""
	The distance and fuel consumption in this version are the same. Astar will have to adapt when this simplification is changed.
	Charging station is a cell that can charge cars. It has a queue of cars and a number of charging slots.
	The chargins statation (CS) alson has a route map to all cells of the city. This route map is used to calculate the distance to the CS.

	Attributes:
		p (Parameters): Parameters of the simulation
		grid (Grid): Grid of the city
		cell (Cell): Cell of the grid where the CS is located
		numberCharging (int): Number of charging slots
		queue (List[Car]): Queue of cars
		car (List[Car]): List of cars in the charging slots
	"""
	def __init__(self,p,grid,cell,numPlugins=-1):
		self.p=p
		self.grid=grid
		self.cell=cell
		if numPlugins == -1:
			#self.numPlugins = p.numberChargingPerStation
			self.numPlugins = p.numberChargersPerBlock
		else:
			self.numPlugins = numPlugins
		cell.cs=self
		# Note: factorizable
		self.queue=[]
		self.carsInCS=0
		self.reserve=[]

		#self.insertInRouteMap(cell)
	def createChargins(self):
		p=self.p
		numberCharging=int(self.numPlugins)
		self.numberCharging=numberCharging
		self.car=[None for i in range(numberCharging)]

	def TicksStepsToAttend(self):
		total=0
		for car in self.queue:
			total+=(car.p.carMovesFullDeposity-car.moves)/car.p.carRechargePerTic
		for car in self.car:
			if car!=None:
				total+=(car.p.carMovesFullDeposity-car.moves)/car.p.carRechargePerTic
		for car in self.reserve:
			total+=(car.p.carMovesFullDeposity-car.moves)/car.p.carRechargePerTic	
		return total/self.numberCharging	


	def moveCS(self,t):
		if self.carsInCS==0:
			return
		# If there is a car in the queue, and there is a gap in the station, then the car enters the station
		while 0<len(self.queue) and None in self.car:
			car=self.queue.pop(0)
			i=self.car.index(None)
			self.car[i]=car
			car.state=CarState.Charging

		# Recharge the cars that are in the charger
		candidateToLeave=-1
		for i in range(self.numberCharging):
			if self.car[i]!=None:
				self.car[i].moves+=self.p.carRechargePerTic
				#print("Recharge percent:",self.car[i].moves/self.p.carMovesFullDeposity*100,"%")
				if self.car[i].moves>self.p.carMovesFullDeposity:
					self.car[i].moves=self.p.carMovesFullDeposity
					candidateToLeave=i

		#candidateToLeave=-1
		if 0<=candidateToLeave and self.cell.car==None:
			car=self.car[candidateToLeave]
			self.car[candidateToLeave]=None
			car.state=CarState.Driving
			car.target2=None
			self.cell.car=car
			self.carsInCS-=1
			car.cell=self.cell
			car.calculateRoute(car.cell,t)

	def insertInRouteMap(self):
		'''
		Insert the CS in the route map of the city.
		This a reverse version of the A* algorithm. 
		It is used to calculate the distance to the CSs near on the bifurcation cells. 
		'''
		cell=self.cell
		visited = []
		distance = 0
		current_level = [cell]

		while current_level:
			next_level = []

			for current_cell in current_level:
				visited.append(current_cell)

				allowInsert=False
				if len(current_cell.destination) > 1:
					# sort h2cs
					# if worst is better than the current distance break propagation
					if len(current_cell.h2cs)+1<self.p.opmitimizeCSSearch:
						allowInsert=True
					else:
						if distance<current_cell.h2cs[-1].distance:
							allowInsert=True
					if allowInsert:
						current_cell.h2cs.append(HeuristicToCS(self, distance))
						current_cell.h2cs.sort(key=lambda x: x.distance)
						current_cell.h2cs=current_cell.h2cs[:self.p.opmitimizeCSSearch]
				else:
					allowInsert=True

				if allowInsert:
					# Add the origins to the list of the next level
					for origin in current_cell.origin:
						if origin not in visited:
							next_level.append(origin)

			# Increase the distance and move to the next level
			distance += 1
			current_level = next_level

class HeuristicToCS:
	"""
	Heuristic to Charging Station is a class that stores the distance to a CS in a bifurcation cell.
	It is a first version of the route map. 
	"""
	def __init__(self,cs:ChargingStation,distance:int):
		self.cs=cs
		self.distance=distance

class Cell:    
	"""
	Cell is a class that represents a cell of the grid. It can be a street, a bifurcation or free. 
	It contains a maximun of a car and a CS. 
	When it is a street, it has a velocity and a direct link to the nexts cells. 
	The time (t) is used to ensure that the cars respect the security distance. It is like a snake game. 
	Same t represents the tail of the snake.
	"""

	# factorizable
	ONE=1
	TWO=2
	FREE = 0

	def __init__(self, initial_state):
		self.h2cs=[]
		self.state = initial_state
		self.next_state = None
		# factorizable
		self.neighbors = [[0,0,0],[0,0,0],[0,0,0]]
		self.origin=[]
		self.destination=[]
		self.velocity=0
		self.x=-1
		self.y=-1

		self.car=None
		self.cs=None
		self.t=0
		self.t0=0
		self.tCache=0 # time in with cached is calculated
		self.timeCache=None # time of the calculation
		self.semaphore=[] # if there is a car over then close the cells in list
		self.occupation=0
		self.exponentialOccupation=0
		self.exponentialLastT=0
		self.pollutionLevel=0

	
	def get_cnnData_value(self) -> list:
			# get the type of the car of the cell
			carTypes = {None: 0, CarType.EV: 1, CarType.Diesel: 2, CarType.Petrol: 3}
			carType_ = carTypes[self.car]
			# get the number of cars per street
			numCars = 0
			if self.car is not None:
					numCars += 1
			if self.cs is not None:
					numCars += self.cs.carsInCS
			# elif dataType == "buildings":
			
			# get if is street/cs or not
			street_ = self.cell.origin or self.cell.destination or self.cell.cs is not None
			# get the number of chargers for each cs
			num_cs = self.cell.cs.numberCharging if self.cell.cs is not None else 0
					
			return [carType_, numCars, street_, num_cs]


	def add_neighbor(self, neighbor):
		self.neighbors.append(neighbor)

	def set_next_state(self):
		total = sum(sum(objeto.state for objeto in fila) for fila in self.neighbors) 
		total -= self.neighbors[1][1].state
		total = total // Cell.STREET
		
		if self.state == Cell.STREET:
			if (total < 2) or (total > 3):
				self.next_state = Cell.FREE
			else:
				self.next_state = self.state
		else:
			if total == 3:
				self.next_state = Cell.STREET
			else:
				self.next_state = self.state

	def update_state(self):
		self.state = self.next_state
	
	# factorizable
	def updateColor(self):
		count=len(self.origin+self.destination)
		if count==0:
			self.state=Cell.FREE
		elif count==1 or count==2:
			self.state=Cell.ONE
		elif count==2:
			self.state=Cell.TWO

	def update(self,t): 
		# when it is an intersection
		stop=False
		for origin in self.origin: # reversed(self.origin):
			if not stop:
				if origin.car is None:
					continue
				stop=True
			else:
				origin.t=t+1

	def color(self,city):
		cell=self
		
		if cell.state==Cell.FREE:
			return 0
		else:
			if len(cell.destination)==0 or len(cell.origin)==0:
				r=3
			elif len(cell.destination)==2:
				r=2
			else:
				r=1
		
		if cell.t==city.t and cell.car==None:
			r=3
		if cell.car!=None: # and cell.car.id==24:
			if cell.car.p.type==CarType.EV:
				r=5
			else:
				r=6
		if cell.cs!=None:
			r=4
		
		# car 0 target is red
		if 0<len(city.cars) and city.cars[0].target==cell:
			r=3

		# artificial origin of semaphores to test location
		# if 0<len(cell.semaphore):
		# 	r=3

		return r

class Street:
	"""
	Street is used as sugar syntax to define a street.

	Attributes:
		path (List[tuple]): List of points of the street
		velocity (int): Velocity of the street
		lanes (int): Number of lanes of the street
	"""
	def __init__(self, path, velocity, lanes, csUbicationable=False):
		self.path = path
		self.velocity = velocity
		self.lanes = lanes
		self.csUbicationable = csUbicationable

class Block:
	"""
	Block is used as sugar syntax to define the streets. 
	The direction of the streets is important because the cars can only move in the direction of the streets.
	At same time you draw the block connet the cells of the grid.
	The construction is a list of streets that is rotated 90 degrees to fill the mosaique of the block.
	"""

	def __init__(self,p,allocate_CS = True):
		self.p=p
		self.allocate_CS = allocate_CS
		r = 1
		self.lanes=[]
		self.velocities=[]
		self.csUbicable=[]
		self.sugar(
			Street([ (-1,3), (3,3), (3,-1) ], 3,2), # roundabout
			#Street([ (2,6), (6,6), (6,2) ], 3,1), # New roundabout lane
			# parametrizable
			#Street([(r,3), (r,-1)],1,2), # cross
			#Street([(-1,r),(3,r)],1,2),

			Street([(r,47),(r,3)],5,2,False), # Avenues # HACK: VELOCIDAD 2 -> 5
			Street([(3,r),(47,r)],5,2), 

			Street([(47,15),(r+1-1,15)],3,1,True), # Streets #incorporación
			Street([(r+1-1,36), (47,36) ],3,1,True), 			
			Street([ (15,r+1-1), (15,47) ],3,1,True),
			Street([ (36,47), (36,r+1-1), ],3,1,True),

			# Street([(r+1-1,15),(47,15)],1,1), # inverse the direction
			# Street([(47,36),(r+1-1,36)  ],1,1), 
			# Street([(15,47), (15,r+1-1)  ],1,1),
			# Street([(36,r+1-1), (36,47)  ],1,1),

		)
		self.semaphores=[]
		self.laneWithCS=7
		self.numberOfCS=0
		# self.semaphores=[
		# 	(2,3,1,4),
		# 	(3,3,2,4),
		# ]
		# self.semaphores=[
		# 	(0,3,1,4),
		# 	(1,3,2,4),
		# ]
		# self.semaphores=[ # peor que autosemaforo
		# 	(-1,3,1,4),
		# 	(0,3,2,4),
		# ]

		max_width = 0
		max_height = 0
		for lane in self.lanes:
			for point in lane:
				if point[0] > max_width:
					max_width = point[0]
				if point[1] > max_height:
					max_height = point[1]
		self.width = (max_width+1)*2
		self.height = (max_height+1)*2
		self.ubiqueCSRest=0

	def semaphore(self,grid,x,y):
		''' 
		If a car stays in (x1,y1) then the semaphore is red in (x2,y2)
		'''
		def semaphore2(x1,y1,x2,y2):
			presed=grid.grid[(y1+y)%grid.heigh,(x1+x)%grid.width]
			inred=grid.grid[(y2+y)%grid.heigh,(x2+x)%grid.width]
			presed.semaphore.append(inred)
			if len(presed.semaphore)!=1:
				print("Error: semaphore")

		# for x1,y1,x2,y2 in self.semaphores:
		# 	semaphore2(x1+1,y1+1,x2+1,y2+1)
		# 	semaphore2(y1+1,-x1-1,y2+1,-x2-1)
		# 	semaphore2(-x1-1,-y1-1,-x2-1,-y2-1)
		# 	semaphore2(-y1-1,x1+1,-y2-1,x2+1)

	def pathPluslane(self,path,lane):
		name=["" for i in range(len(path))]
		for i in range(len(path)-1):
			source=path[i]
			target=path[i+1]
			if source[1]==target[1]: #vertical
				if source[0]<target[0]: #up
					key="u"
				else: #down
					key="d"
			else: #horizontal
				if source[1]<target[1]: #right
					key="r"
				else: #left
					key="l"
			name[i]+=key
			name[i+1]+=key
			
		
		switch={				
			"u":(0,1),
			"d":(0,-1),
			"r":(-1,0),
			"l":(1,0),
			"ur":(-1,1),
			"ul":(1,1),
			"dr":(1,1),
			"dl":(-1,1),
			"ru":(-1,1),
			"rd":(-1,-1),
			"lu":(1,1),
			"ld":(1,-1),
		}

		newPath=[]
		for i,p in enumerate(path):
			delta=switch[name[i]]
			newPath.append((p[0]+delta[0]*lane,p[1]+delta[1]*lane))
		return newPath

	def sugar(self,*streets):
		for street in streets:
			for lane in range(street.lanes):
				self.lanes.append(self.pathPluslane(street.path,lane))
				self.velocities.append(street.velocity)
				self.csUbicable.append(street.csUbicationable)

	def draw2(self,grid,lastx,lasty,xx,yy,velocity,csUbicationable):
		if lastx is None:
			return
		if lasty is None:
			return
		last=None
		xmax = self.p.horizontalBlocks * self.width-1
		ymax = self.p.verticalBlocks * self.height-1
		used=False
			
		def csUbication(current):
			nonlocal used
			if 0<=self.ubiqueCSRest and csUbicationable and not used and self.allocate_CS:
				current.cs=ChargingStation(self.p,grid,current)
				grid.cs.append(current.cs)
				self.ubiqueCSRest-=1
				used=True

		if lastx == xx:
			inc = -1
			if lasty < yy:
				inc = 1
			last_y = lasty%grid.heigh
			for i in range(lasty,yy+inc,inc):
				current_y = i%grid.heigh
				current=grid.grid[i%grid.heigh,xx%grid.width]
				if (abs(last_y-current_y)>1) and self.p.ring:
					pass
				else:
					# csUbication(current)
					grid.link(last,current,velocity)			
				last=current
				last_y = i%grid.heigh
				yield
		if lasty == yy:
			inc = -1
			if lastx < xx:
				inc = 1
			last_x = lasty%grid.width
			for i in range(lastx,xx+inc,inc):
				current_x = i%grid.width
				current=grid.grid[yy%grid.heigh,i%grid.width]
				if (abs(last_x-current_x)>1) and self.p.ring:
					pass
				else:
					# csUbication(current)
					grid.link(last,current,velocity)
				last=current
				last_x = i%grid.width
				yield
		used=False
		csUbication(current)

	def draw(self,grid,x,y):
		self.ubiqueCSEach=self.p.numberStationsPerBlock/len(self.lanes)
		for i,lane in enumerate(self.lanes):
			self.ubiqueCSRest+=self.ubiqueCSEach
			csUbicationable=self.csUbicable[i]
			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[0]+1
				yy=y+point[1]+1
				#grid.grid[xx][yy].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy

			lastx=None
			lasty=None
			for point in lane:
				xx=x+point[1]+1
				yy=y-point[0]-1
				#grid.grid[x+point[1]+1,y-point[0]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[0]-1
				yy=y-point[1]-1
				#grid.grid[x-point[0]-1,y-point[1]-1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy
			lastx=None
			lasty=None
			for point in lane:
				xx=x-point[1]-1
				yy=y+point[0]+1
				#grid.grid[x-point[1]-1,y+point[0]+1].state=Cell.STREET
				for _ in self.draw2(grid,lastx,lasty,xx,yy,self.velocities[i],csUbicationable):
					yield
				lastx=xx
				lasty=yy
		#if 0<self.ubiqueCSRest:
			#print("There are more CS than expected to ubicate")
		#aqui anillo NO
		
# separable interfaz y modelo
class City:
	"""
	City is a general holder of the simulation. It encapsules low level details of the graphics representation.
	generators are used to draw the buildings of the city and the simulation. It uses the yield instruction.
	Yield can stop the execution of the container function and can be used recursively.
	"""
	def __init__(self,p,indiv=None):		
		self.p=p
		self.indiv = indiv
		self.block=Block(p,allocate_CS = indiv == None)
		self.grid=Grid(p,p.verticalBlocks*self.block.height,p.horizontalBlocks*self.block.width) #Hacked: añadido "p,"

		self.t=0

		city = self
		self.city_generator = city.generator()
		self.g=city.grid
		self.cars=[]
		#g=Grid(100,100)

		# Animation
		#next(self.city_generator)
		#next(self.city_generator)

		#self.valid_coordinates = [(cellX,cellY) for (cellX,cellY) in itertools.product(range(len(self.grid.grid[0])),range(len(self.grid.grid))) if self.grid.grid[cellY][cellX].state != self.grid.grid[cellY][cellX].FREE]    # HACKED
		#print(self.valid_coordinates)
	def shell(self):
		while True:
			shell=input("Simtravel3> ")
			if shell=="plot":
				self.plot()
			else:
				# split shell
				shell2=shell.split(" ")
				if self.p.__dict__.get(shell2[0])!=None:
					setattr(self.p,shell2[0],int(shell2[1]))

	def plot(self,block=False):
		fig, ax = plt.subplots()

		bounds = [0, 1, 2, 3, 4, 5, 6, 7]
		cmap = mcolors.ListedColormap(['black',  'green', 'blue','red', 'yellow', 'white','orange'])
		norm = mcolors.BoundaryNorm(bounds, cmap.N)
	

		def extract_color(cell_obj):
			return cell_obj.color(self)

		img = ax.imshow(np.vectorize(extract_color)(self.g.grid), interpolation='nearest', cmap=cmap, norm=norm)
		self.ani = animation.FuncAnimation(fig, self.update, fargs=(img, self.g.grid, self.g.heigh,self.g.width, ), frames=50,interval=1)

		def on_click(event):
			# Evento al hacer click
			x, y = int(event.xdata), int(event.ydata)  # Convertir las coordenadas a enteros para obtener la posición en la matriz
			print(f"X: {x}, Y: {y}")
			car=self.g.grid[y,x].car
			if car!=None:
				print("id car",car.id)

		fig.canvas.mpl_connect('button_press_event', on_click)

		plt.show(block=block) #HACKED ANTES MOSTRABA



	def runWithoutPlot(self, times, returnFits = False, contaminationExp = False, delta = 0.1, corner_factor = 1, gamma = 0.1, acc = None, dif_matrix = None, wind = None, names = None):
		notSaveGif=True
		if notSaveGif:
			initial_time = time.time()
			savedTimes=10
			next(self.city_generator)#GENERARLO MUCHAS VECES
			# print("\t\tNum cars: ", len(self.cars))
			if returnFits or contaminationExp:
				timestepvalue=np.float32(1.8)
				cellsize=5
				num_cs = len(self.grid.cs)
				loc_fits = [0] * num_cs
				carsAtDestination = 0
				width = len(acc)
				height = len(acc[0])
				positions = {cartype : {car.id: [(car.cell.x,car.cell.y)]*(times+1) for car in self.cars if car.p.type == cartype and car.cell is not None} for cartype in CAR_PROPERTIES}
				positions[CarType.EV].update({car.id: [(car.target2.x,car.target2.y)]*(times+1) for car in self.cars if car.p.type == CarType.EV and car.cell is None}) #CORREGIR A LOS QUE PUEDAN CARGAR
				velocities = {cartype : {car.id: [0]*(times+1) for car in self.cars if car.p.type == cartype} for cartype in CAR_PROPERTIES}
				mean_velocities = []
				mean_all_cars_vel = []
				accelerations = {cartype : {car.id: [0]*(times+1) for car in self.cars if car.p.type == cartype} for cartype in CAR_PROPERTIES}
				def substract_tuples(a,b,factor):
					aux = ((a[0]-b[0])*factor,(a[1]-b[1])*factor)
					if a[0]-b[0]>0.5*width:
						if p.ring:
							print('this should not happen')
						aux = (aux[0]-width*factor,aux[1])
					if a[0]-b[0]<-0.5*width:
						if p.ring:
							print('this should not happen')
						aux = (aux[0]+width*factor,aux[1])
					if a[1]-b[1]>0.5*height:
						if p.ring:
							print('this should not happen')
						aux = (aux[0],aux[1]-height*factor)
					if a[1]-b[1]<-0.5*height:
						if p.ring:
							print('this should not happen')
						aux = (aux[0],aux[1]+height*factor)
					return aux
				def tuple_norm(a):
						return np.sqrt(a[0]**2+a[1]**2)

				G = {f"{cartype}_{pollutant}": np.zeros((width, height), dtype=np.float32)
							for cartype in CAR_PROPERTIES
							for pollutant in poll_coefs[cartype]
							if pollutant != 'NOxdecel' and pollutant != 'VOCdecel'}

				P = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2), dtype=np.float32)
							for cartype in CAR_PROPERTIES
							for pollutant in poll_coefs[cartype]
							if pollutant != 'NOxdecel' and pollutant != 'VOCdecel'}

				savedG = {f"{cartype}_{pollutant}": np.zeros((width, height, savedTimes+1), dtype=np.float32)
							for cartype in CAR_PROPERTIES
							for pollutant in poll_coefs[cartype]
							if pollutant != 'NOxdecel' and pollutant != 'VOCdecel'}

				savedP = {f"{cartype}_{pollutant}": np.zeros((width, height,savedTimes+1), dtype=np.float32)
							for cartype in CAR_PROPERTIES
							for pollutant in poll_coefs[cartype]
							if pollutant != 'NOxdecel' and pollutant != 'VOCdecel'}
				
				averages = {}
				for key in P:
					averages[f"P_{key}"] = np.zeros((width, height))
					averages[f"G_{key}"] = np.zeros((width, height))

				map_cars = {}

				# Populate the dictionary using a loop
				for cartype in CAR_PROPERTIES:
					map_cars[f"{cartype}_positions"] = np.zeros((width, height, savedTimes+1))
					map_cars[f"{cartype}_velocities"] = np.zeros((width, height, savedTimes+1))
					map_cars[f"{cartype}_accelerations"] = np.zeros((width, height, savedTimes+1))
					map_cars[f"{cartype}_braking"] = np.zeros((width, height, savedTimes+1))

				for key in map_cars:
					averages[f"cars_{key}"] = np.zeros((width, height))

				aux=np.zeros_like(P[f"1_CO2"])
				#Arand=np.random.rand(width,height)
				Arand=np.zeros((width,height))#np.ones((width,height))
			for i in range(times+1):
				try:
					#print(i)
					n = i-times//4
					isave = i%(times//savedTimes)==0
					isaved = min(savedTimes,i//(times//savedTimes))
					for key in G:
						G[key][:,:]=0
					if i==0:
						for k in range(width):
							for j in range(height):
								self.grid.grid[k,j].pollutionLevel = Arand[k,j]
					#for pollutant in P:
					#	for evType in P[pollutant]:
					#		Psum[:,:,i] += P[cartype][pollutant][:,:,i]
					if names[-1]:
						for k in range(width):
							for j in range(height):
								if self.grid.grid[k,j].pollutionLevel <= Arand[k,j]:
									self.grid.grid[k,j].pollutionLevel = max(Arand[k,j], sum([P[f"{cartype}_CO2"][k+1,j+1] for cartype in CAR_PROPERTIES if cartype != CarType.EV])) #CAMBIAR A NEEDSCHARGING
								else:
									self.grid.grid[k,j].pollutionLevel = 0.5*(sum([P[f"{cartype}_CO2"][k+1,j+1] for cartype in CAR_PROPERTIES if cartype != CarType.EV]) + self.grid.grid[k,j].pollutionLevel) #CAMBIAR A NEEDSCHARGING
					if i>0:
						next(self.city_generator)
					if returnFits or contaminationExp:
						if n>=0:
							for k in range(num_cs):
								current_cs = self.grid.cs[k]
								numChargingCars = len([1 for car in current_cs.car if car!=None])
								assert current_cs.numPlugins > 0, "Error, CS without plugins (runWithouPlot, calculate Fitness)."
								loc_fits[k] = n/(n+1)*loc_fits[k]+ 1/(n+1)*(current_cs.numPlugins+len(current_cs.queue)) / (current_cs.numPlugins+numChargingCars)
							mean_all_cars_vel = []
						for cartype in positions:
							if n>=0:
								averages[f"cars_{cartype}_positions"]*=n/(n+1)
								averages[f"cars_{cartype}_velocities"]*=n/(n+1)
								averages[f"cars_{cartype}_accelerations"]*=n/(n+1)
								averages[f"cars_{cartype}_braking"]*=n/(n+1)
							for key in positions[cartype]:
								if self.cars[key].cell is None:
									positions[cartype][key][i] = (self.cars[key].target2.x,self.cars[key].target2.y)
								else:
									positions[cartype][key][i] = (self.cars[key].cell.x,self.cars[key].cell.y)#[(car.cell.x,car.cell.y) for car in self.cars if car.id == key][0]
								if i>0:
									velocities[cartype][key][i] = tuple_norm(substract_tuples(positions[cartype][key][i],positions[cartype][key][i-1],cellsize/timestepvalue)) # Factor due to Amaro
								vel = velocities[cartype][key][i]
								if i>1:
									accelerations[cartype][key][i] = (vel-velocities[cartype][key][i-1])/timestepvalue
								accel = accelerations[cartype][key][i]
								a,b=positions[cartype][key][i]
								if isave:
									map_cars[f"{cartype}_positions"][a,b,isaved]+=1
									map_cars[f"{cartype}_velocities"][a,b,isaved]+=vel
									if accel>0:
										map_cars[f"{cartype}_accelerations"][a,b,isaved]+=accel
									else:
										map_cars[f"{cartype}_braking"][a,b,isaved]-=accel
								if n>=0:
									averages[f"cars_{cartype}_positions"][a,b] += 1/(n+1)
									averages[f"cars_{cartype}_velocities"][a,b] += vel/(n+1)
									if accel>0:
										averages[f"cars_{cartype}_accelerations"][a,b] += accel/(n+1)
									else:
										averages[f"cars_{cartype}_braking"][a,b] -= accel/(n+1)
									cell_vel = self.grid.grid[a, b].velocity
									mean_all_cars_vel.append(min(vel/cell_vel, 1.))
								if self.cars[key].cell is not None:
									for key in G:
										if key.startswith(f"{cartype}_"):
											pollutant = key.split(f"{cartype}_")[1]
											pollutant2 = pollutant
											if accel < -0.5:
												if pollutant == 'NOx':
													pollutant2 = 'NOxdecel'
												if pollutant == 'VOC':
													pollutant2 = 'VOCdecel'
											G[f"{cartype}_{pollutant}"][a,b]=max(poll_coefs[cartype][pollutant2][0], poll_coefs[cartype][pollutant2][1] + poll_coefs[cartype][pollutant2][2]*vel + poll_coefs[cartype][pollutant2][3]*vel**2 + poll_coefs[cartype][pollutant2][4]*accel + poll_coefs[cartype][pollutant2][5]*accel**2 + poll_coefs[cartype][pollutant2][6]*vel*accel)*timestepvalue

						for key in P:
							diffusion_edge = (
								P[key][0:-2, 1:-1] + P[key][2:, 1:-1] +    # Left and Right
								P[key][1:-1, 0:-2] + P[key][1:-1, 2:]      # Up and Down
							)
							diffusion_corner = (
								P[key][0:-2, 0:-2] + P[key][2:, 2:] +      # Diagonals
								P[key][2:, 0:-2] + P[key][0:-2, 2:]        # Diagonals
							)
							
							#P[key][1:-1, 1:-1] = acc * (dif_matrix * P[key][1:-1, 1:-1] + delta * (diffusion_edge + diffusion_corner * corner_factor) / (4 + 4 * corner_factor))# ORDEN DE LAS OPERACIONES NO ESTA CLARO??
							aux[1:-1,1:-1] = acc * (dif_matrix * P[key][1:-1, 1:-1] + delta * (diffusion_edge + diffusion_corner * corner_factor) / (4 + 4 * corner_factor))# ORDEN DE LAS OPERACIONES NO ESTA CLARO??
							if isave:
								#savedP[key][:,:,isaved] = copy.deepcopy(P[key][1:-1, 1:-1])
								savedP[key][:,:,isaved] = copy.deepcopy(aux[1:-1, 1:-1])
								savedG[key][:,:,isaved] = copy.deepcopy(G[key])
							if n>=0:
								#averages[f"P_{key}"] = n/(n+1) * averages[f"P_{key}"] + 1/(n+1) * P[key][1:-1,1:-1]								
								averages[f"P_{key}"] = n/(n+1) * averages[f"P_{key}"] + 1/(n+1) * aux[1:-1,1:-1]								
								averages[f"G_{key}"] = n/(n+1) * averages[f"G_{key}"] + 1/(n+1) * G[key]								
							#P[key][1:-1, 1:-1] = (1-gamma) * acc * (wind[0]*P[key][1:-1, 2:] + wind[1]*P[key][1:-1, :-2] + wind[2]*P[key][:-2, 1:-1] + wind[3]*P[key][2:, 1:-1] + wind[4]*P[key][:-2, 2:] + wind[5]*P[key][2:, 2:] + wind[6]*P[key][:-2, :-2] + wind[7]*P[key][2:, :-2] + wind[8]*P[key][1:-1, 1:-1] + G[key][:, :])
							P[key][1:-1, 1:-1] = (1-gamma) * acc * (wind[0]*aux[1:-1, 2:] + wind[1]*aux[1:-1, :-2] + wind[2]*aux[:-2, 1:-1] + wind[3]*aux[2:, 1:-1] + wind[4]*aux[:-2, 2:] + wind[5]*aux[2:, 2:] + wind[6]*aux[:-2, :-2] + wind[7]*aux[2:, :-2] + wind[8]*aux[1:-1, 1:-1] + G[key][:, :])
						if returnFits or contaminationExp:
							if n>=0:
								mean_velocities.append(sum(mean_all_cars_vel)/len(mean_all_cars_vel))

				except StopIteration:
					#print("\ncity_generator has no more items to generate.")
					break
			if contaminationExp and not returnFits:
				del P
				del G
				np.savez_compressed(os.path.join(dataSaveDir, f'P_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **savedP)
				del savedP
				np.savez_compressed(os.path.join(dataSaveDir, f'G_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **savedG)
				del savedG
				np.savez_compressed(os.path.join(dataSaveDir, f'A_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **averages)
				del averages
				np.savez_compressed(os.path.join(dataSaveDir, f'C_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **map_cars)
				del map_cars
				gc.collect()
			if returnFits:
				Psum = averages["P_1_CO2"]+averages["P_2_CO2"]
				# new_acc = np.ones((width+2, height+2))
				# new_acc[1:-1,1:-1] = acc

				mean_vel = sum(mean_velocities)/len(mean_velocities) # [0, 1] de todos los coches. función por partes? 
				mean_vel_ = mean_vel if mean_vel < 0.8 else 1
				global_fit = math.sqrt(Psum[acc==1].mean() * Psum[acc==1].std())/max(0.0001, mean_vel_)
				return global_fit, loc_fits

		else:#f runw
			width = len(acc)
			height = len(acc[0])
			'''
			wind=list(wind)
			for k in range(len(wind)):
				print('Here: ', k)
				aux = np.zeros((width,height,times), dtype = np.float32)
				if len(np.shape(wind[k]))!=3:
					for t in range(times):
						aux[:,:,t] = wind[k]
					wind[k] = aux
			wind=tuple(wind)
			del aux
			gc.collect()
			'''
					
			initial_time = time.time()
			next(self.city_generator)
			if returnFits or contaminationExp:
				timestepvalue=np.float32(1.8)
				cellsize=5
				num_cs = len(self.grid.cs)
				loc_fits = [0] * num_cs
				carsAtDestination = 0
				#keys = [car.id for car in self.cars if car.p.type == CarType.ICEV]
				positions = {cartype : {car.id: [(car.cell.x,car.cell.y)]*(times+1) for car in self.cars if car.p.type == cartype and car.cell is not None} for cartype in CAR_PROPERTIES}
				#positions = {car.id: [(car.cell.x,car.cell.y)]*times for car in self.cars if car.p.type == CarType.ICEV and car.cell is not None}
				positions[CarType.EV].update({car.id: [(car.target2.x,car.target2.y)]*(times+1) for car in self.cars if car.p.type == CarType.EV and car.cell is None}) #CORREGIR A LOS QUE PUEDAN CARGAR
				#print('Number of wrong ICEV cars', len([car for car in self.cars if car.p.type != CarType.EV and car.cell is None]))
				#diesel = list(positions.keys())[:len(positions)//3]
				#print('Number of cars: ', len(self.cars))
				#print('Should be: ', len(self.cars))
				#print('Number of EV: ', CAR_PROPERTIES[CarType.EV].number)
				#print('Number of Petrol: ', CAR_PROPERTIES[CarType.Petrol].number)
				#print('Number of diesel: ', CAR_PROPERTIES[CarType.Diesel].number)
				velocities = {cartype : {car.id: [(0,0)]*(times+1) for car in self.cars if car.p.type == cartype} for cartype in CAR_PROPERTIES}
				accelerations = {cartype : {car.id: [0]*(times+1) for car in self.cars if car.p.type == cartype} for cartype in CAR_PROPERTIES}
				#velocities = {car.id: [(0,0)]*times for car in self.cars if car.p.type == CarType.ICEV}
				#accelerations = {car.id: [0]*times for car in self.cars if car.p.type == CarType.ICEV}
				def substract_tuples(a,b,factor):
					aux = ((a[0]-b[0])*factor,(a[1]-b[1])*factor)
					if a[0]-b[0]>0.5*width:
						aux = (aux[0]-width*factor,aux[1])
					if a[0]-b[0]<-0.5*width:
						aux = (aux[0]+width*factor,aux[1])
					if a[1]-b[1]>0.5*height:
						aux = (aux[0],aux[1]-height*factor)
					if a[1]-b[1]<-0.5*height:
						aux = (aux[0],aux[1]+height*factor)
					return aux
				def tuple_norm(a):
						return np.sqrt(a[0]**2+a[1]**2)

				G = {cartype : {pollutant : np.zeros((width, height, times+1), dtype=np.float32) for pollutant in poll_coefs[cartype] if pollutant != 'NOxdecel' and pollutant != 'VOCdecel'} for cartype in CAR_PROPERTIES}
				P = {cartype : {pollutant : np.zeros((width+2, height+2, times+1), dtype=np.float32) for pollutant in poll_coefs[cartype] if pollutant != 'NOxdecel' and pollutant != 'VOCdecel'} for cartype in CAR_PROPERTIES}

				'''
				G = {pollutant: {CarType.Petrol: np.zeros((width+2, height+2, times+1)),
								CarType.Diesel: np.zeros((width+2, height+2, times+1)),
								CarType.EV: np.zeros((width+2, height+2, times+1))}
					for pollutant in ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMexhaustprueba', 'PMnonexhaust25', 'PMnonexhaust10']}#'PMexhaustprueba', 

				P = {pollutant: {CarType.Petrol: np.zeros((width+2, height+2, times+1)),
								CarType.Diesel: np.zeros((width+2, height+2, times+1)),
								CarType.EV: np.zeros((width+2, height+2, times+1))}
					for pollutant in ['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMexhaustprueba', 'PMnonexhaust25', 'PMnonexhaust10']}
				#print(wind)
				'''
			for i in range(times):
				print('time = ', i)
				try:
					if i==0:
						for k in range(width):
							for j in range(height):
								self.grid.grid[k,j].pollutionLevel = 0
					#for pollutant in P:
					#	for evType in P[pollutant]:
					#		Psum[:,:,i] += P[cartype][pollutant][:,:,i]
					if names[-1]:
						for k in range(width):
							for j in range(height):
								self.grid.grid[k,j].pollutionLevel = 0.5*(sum([P[cartype]['CO2'][k+1,j+1,i] for cartype in CAR_PROPERTIES if cartype != CarType.EV]) + self.grid.grid[k,j].pollutionLevel)
					if i>0:
						next(self.city_generator)
					for k in range(num_cs):
						current_cs = self.grid.cs[k]
						loc_fits[k] += -(len([1 for car in current_cs.car if car!=None])-len(current_cs.queue))/current_cs.numPlugins
						#self.cars[523]
					if returnFits or contaminationExp:
						#aux.append(len([car for car in self.cars if car.p.type == CarType.ICEV and car.cell is None]))
						for cartype in positions:
							for key in positions[cartype]:# for c in cars, positions[c]=...
								if self.cars[key].cell is None:
									positions[cartype][key][i] = (self.cars[key].target2.x,self.cars[key].target2.y)
								else:
									positions[cartype][key][i] = (self.cars[key].cell.x,self.cars[key].cell.y)#[(car.cell.x,car.cell.y) for car in self.cars if car.id == key][0]
								if i>0:
									velocities[cartype][key][i] = (substract_tuples(positions[cartype][key][i],positions[cartype][key][i-1],cellsize/timestepvalue)) # Factor due to Amaro
								vel = tuple_norm(velocities[cartype][key][i])
								if i>1:
									accelerations[cartype][key][i] = (vel-tuple_norm(velocities[cartype][key][i-1]))/timestepvalue
								accel = accelerations[cartype][key][i]
								a,b=positions[cartype][key][i]
								if self.cars[key].cell is not None:
									for pollutant in G[cartype]:
										pollutant2 = pollutant
										if accel < -0.5:
											if pollutant == 'NOx':
												pollutant2 = 'NOxdecel'
											if pollutant == 'VOC':
												pollutant2 = 'VOCdecel'
										G[cartype][pollutant][a,b,i]=max(poll_coefs[cartype][pollutant2][0], poll_coefs[cartype][pollutant2][1] + poll_coefs[cartype][pollutant2][2]*vel + poll_coefs[cartype][pollutant2][3]*vel**2 + poll_coefs[cartype][pollutant2][4]*accel + poll_coefs[cartype][pollutant2][5]*accel**2 + poll_coefs[cartype][pollutant2][6]*vel*accel)*timestepvalue


						for cartype in P:
							for pollutant in P[cartype]:
								diffusion_edge = (
									P[cartype][pollutant][0:-2, 1:-1, i] + P[cartype][pollutant][2:, 1:-1, i] +    # Left and Right
									P[cartype][pollutant][1:-1, 0:-2, i] + P[cartype][pollutant][1:-1, 2:, i]      # Up and Down
								)
								diffusion_corner = (
									P[cartype][pollutant][0:-2, 0:-2, i] + P[cartype][pollutant][2:, 2:, i] +      # Diagonals
									P[cartype][pollutant][2:, 0:-2, i] + P[cartype][pollutant][0:-2, 2:, i]        # Diagonals
								)
								#aux=np.sum(P[cartype][pollutant][1:-1, 1:-1, i])
								#print(aux*(1-delta)+delta*np.sum(diffusion_corner+diffusion_edge)/8-np.sum(P[cartype][pollutant][1:-1, 1:-1, i]))
								P[cartype][pollutant][1:-1, 1:-1, i] = acc * (dif_matrix * P[cartype][pollutant][1:-1, 1:-1, i] + delta * (diffusion_edge + diffusion_corner * corner_factor) / (4 + 4 * corner_factor))# ORDEN DE LAS OPERACIONES NO ESTA CLARO??
								#print(np.sum(P[cartype][pollutant][1:-1, 1:-1, i])<=aux)
								#print((dif_matrix==acc*dif_matrix).all())
								#print('iter',i)
								#print(np.sum(P[cartype][pollutant][:,:,i]))
								#print(delta/(4 + 4 * corner_factor))
								#print((dif_matrix+acc*(delta/(4 + 4 * corner_factor))>1).any())
								#aux=np.sum(P[cartype][pollutant][1:-1, 1:-1, i])
								if i < times:
									P[cartype][pollutant][1:-1, 1:-1, i+1] = (1-gamma) * acc * (wind[0]*P[cartype][pollutant][1:-1, 2:, i] + wind[1]*P[cartype][pollutant][1:-1, :-2, i] + wind[2]*P[cartype][pollutant][:-2, 1:-1, i] + wind[3]*P[cartype][pollutant][2:, 1:-1, i] + wind[4]*P[cartype][pollutant][:-2, 2:, i] + wind[5]*P[cartype][pollutant][2:, 2:, i] + wind[6]*P[cartype][pollutant][:-2, :-2, i] + wind[7]*P[cartype][pollutant][2:, :-2, i] + wind[8]*P[cartype][pollutant][1:-1, 1:-1, i] + G[cartype][pollutant][:, :, i])
									#print(2, np.sum(P[cartype][pollutant][1:-1, 1:-1, i+1])<=np.sum(aux+G[cartype][pollutant][1:-1, 1:-1, i]))
						
									
					elapsed_time = time.time() - initial_time
					percentage_done = (i + 1) / times * 100
					# Estimación del tiempo total basado en el progreso actual
					total_estimated_time = elapsed_time / (percentage_done / 100)
					estimated_completion_time = initial_time + total_estimated_time
					#print(
					#	f"Progress: {percentage_done:.2f}% End: {time.strftime('%H:%M:%S', time.localtime(estimated_completion_time))} Total: {int(round(total_estimated_time))} seconds  ",
					#	end="\r"
					#)
				except StopIteration:
						#print("\ncity_generator has no more items to generate.")
						break
				if returnFits or contaminationExp:
						carsAtDestination += len([car for car in self.cars if car.state == CarState.Destination])
						#print(carsAtDestination, '\n')


			#plt.plot(range(len(aux)),aux)
			#plt.show()
			if (returnFits or contaminationExp) and True:
				'''
				def func(x):
					print(time.time())
					E0 = x[0]
					f1 = x[1]
					f2 = x[2]
					f3 = x[3]
					f4 = x[4]
					f5 = x[5]
					f6 = x[6]
					Paux = np.zeros_like(P)
					Gaux = np.zeros_like(G)
					for i in range(times):
						for key in positions:
							vel = tuple_norm(velocities[key][i])
							accel = tuple_norm(accelerations[key][i])
							a,b=positions[key][i]
							Gaux[a,b,i] = max(E0, f1 + f2*vel + f3*vel**2 + f4*accel + f5*accel**2 + f6*vel*accel)
						for key in positionsEV:
							a,b=positionsEV[key][i]
							vel=0
							accel=0
							if i>0:
								vel=tuple_norm(substract_tuples(positionsEV[key][i],positionsEV[key][i-1],5/1.8))
							if i>1:
								accel=tuple_norm(substract_tuples(substract_tuples(positionsEV[key][i],positionsEV[key][i-1],5/1.8),substract_tuples(positionsEV[key][i-1],positionsEV[key][i-2],5/1.8),1/1.8))
							Gaux[a,b,i] += max(E0, f1 + f2*vel + f3*vel**2 + f4*accel + f5*accel**2 + f6*vel*accel)
						diffusion_edge = (
							Paux[0:-2, 1:-1, i] + Paux[2:, 1:-1, i] +    # Up and down
							Paux[1:-1, 0:-2, i] + Paux[1:-1, 2:, i]      # Left and right
						)
						diffusion_corner = (
							Paux[0:-2, 0:-2, i] + Paux[2:, 2:, i] +      # Upper left and lower right diagonal
							Paux[2:, 0:-2, i] + Paux[0:-2, 2:, i]        # Lower left and upper right diagonal
						)
						Paux[1:-1, 1:-1, i] = acc * (dif_matrix * Paux[1:-1, 1:-1, i] + delta / (4 + 4 * corner_factor) * (diffusion_edge + diffusion_corner * corner_factor))
						Paux[1:-1, 1:-1, i+1] = (1-gamma) * acc * (wind[0][:,:,i]*Paux[2:, 1:-1, i] + wind[1][:,:,i]*Paux[:-2, 1:-1, i] + wind[2][:,:,i]*Paux[1:-1, :-2, i] + wind[3][:,:,i]*Paux[1:-1, 2:, i] + wind[4][:,:,i]*Paux[2:, :-2, i] + wind[5][:,:,i]*Paux[2:, 2:, i] + wind[6][:,:,i]*Paux[:-2, :-2, i] + wind[7][:,:,i]*Paux[:-2, 2:, i] + wind[8][:,:,i]*Paux[1:-1, 1:-1, i] + Gaux[1:-1, 1:-1, i])
					return np.sqrt(np.sum((Paux[1:-1,1:-1,:]-P[1:-1,1:-1,:])**2))
				x0 = np.array([0, 5.53e-1, 1.61e-1, -2.89e-3, 2.66e-1, 5.11e-1, 1.83e-1])
				res = minimize(func, x0)

				print(f"Resultado de la optimización: {res}")
				print(f"Valor mínimo encontrado en: {res.x}")
				print(f"Valor mínimo de la función: {res.fun}")


			if returnFits or contaminationExp:
				for cartype in P:
					for pollutant in P[cartype]:
						#if pollutant == 'PMexhaustprueba':
					#		pass
				#		else:
							plt.plot([a*1.8/60 for a in range(times+1)],[np.sum(P[cartype][pollutant][:,:,t]) for t in range(times+1)])
							plt.xlabel('$t (min)$')
							plt.ylabel('$P (g)$')
							#filename = f"total_pollution_gamma_{gamma}.png"
							filename = f"total_pollution_cartype_{str(cartype)}_pollutant_{pollutant}.png"
							plt.savefig(filename)
							#plt.show()
							plt.close()
							#print(np.min(P[cartype][pollutant][:,:,:])>=0)
				'''
				#print(G)
				#print(P)
				#print(positions)
				Psum = P[1]['CO2'] + P[2]['CO2']
				max_color_value = np.max(Psum)#/10
				print(max_color_value)
				#positionMatrix = np.zeros_like(Psum)
				positionMatrixEV = np.zeros((width,height,times))
				positionMatrixPetrol = np.zeros_like(positionMatrixEV)
				positionMatrixDiesel = np.zeros_like(positionMatrixEV)
				for i in range(times):
					for key in positions[0]:
						a, b= positions[0][key][i]
						#a-=1
						#b-=1
						positionMatrixEV[a,b,i] += 1
					for key in positions[1]:
						a, b= positions[1][key][i]
						#a-=1
						#b-=1
						positionMatrixPetrol[a,b,i] += 1
					for key in positions[2]:
						a, b= positions[2][key][i]
						#a-=1
						#b-=1
						positionMatrixDiesel[a,b,i] += 1

					
				'''
				positionMatrix = positionMatrixEV + positionMatrixPetrol + positionMatrixDiesel
				def compare_matrices(A,B):
					if np.shape(A)!=np.shape(B):
						print('The two matrices have different sizes')
						return
					print('Suma primera matriz', np.sum(A))
					print('Suma segunda matriz', np.sum(B))
					A=A/np.sum(A)
					B=B/np.sum(B)
					meanA=(0,0)
					meanB=(0,0)
					m,n=np.shape(A)
					C=A-B
					dif=0
					for i in range(m):
						for j in range(n):
							meanA=(meanA[0]+i*A[i,j],meanA[1]+j*A[i,j])
							meanB=(meanB[0]+i*B[i,j],meanB[1]+j*B[i,j])
							dif+=abs(C[i,j])
					meanA = (meanA[1],meanA[0])
					meanB = (meanB[1],meanB[0])
					print('Posición promedio primera matriz: ', meanA)
					print('Posición promedio segunda matriz: ', meanB)
					print('Diferencia entre ambas matrices: ', dif)
					return
				print('Total Pollution emitted:')
				for pollutant in G:
					print(pollutant, ': ', sum([np.sum(G[cartype][pollutant]) for evType in G[pollutant]]))
				compare_matrices(positionMatrix[:,:,-2],Psum[:,:,-2])
				compare_matrices(P['PMexhaust'][CarType.Petrol][:,:,-2],P['PMexhaustprueba'][CarType.Petrol][:,:,-2])
				compare_matrices(P['PMexhaust'][CarType.Diesel][:,:,-2],P['PMexhaustprueba'][CarType.Diesel][:,:,-2])
				'''
				if True:#Plot?
					def update_plot(frame_number, zarray, plot, fig, ax):
						for collection in ax.collections:
							collection.remove()
						#ax.collections.clear()
						total_pollution = np.sum(zarray[:, :, frame_number])
						plot.set_data(np.transpose(zarray[:, :, frame_number]))
						ax.set_title(f"$t={frame_number}$, $P_{{tot}}={total_pollution:.6f}$")

						xx = []
						yy = []
						for x in range(width):
							for y in range(height):
								if positionMatrixEV[x,y,frame_number]:
									xx.append(x)
									yy.append(y)
						ax.scatter(xx, yy, color='green', s=3)

						xx = []
						yy = []
						for x in range(width):
							for y in range(height):
								if positionMatrixPetrol[x,y,frame_number]:
									xx.append(x)
									yy.append(y)
						ax.scatter(xx, yy, color='black', s=3)

						xx = []
						yy = []
						for x in range(width):
							for y in range(height):
								if positionMatrixDiesel[x,y,frame_number]:
									xx.append(x)
									yy.append(y)
						ax.scatter(xx, yy, color='red', s=3)

						#x_vals = [a[frame_number][0]-1 for a in positions.values()]
						#y_vals = [a[frame_number][1]-1 for a in positions.values()]
						#x_vals_EV = [a[frame_number][0]-1 for a in positionsEV.values()]
						#y_vals_EV = [a[frame_number][1]-1 for a in positionsEV.values()]
						#ax.scatter(x_vals, y_vals, color='green', s=3)
						#ax.scatter(x_vals_EV, y_vals_EV, color='blue', s=1)

					# Create figure and axes
					fig, ax = plt.subplots()
					
					#for i in range(len(acc)):
					#    for j in range(len(acc[0])):
					#        if acc[i, j] == 0:
					#            ax.text(j, i, 'X', ha='center', va='center', color='red')
					
					# Choose colorbar scale: 'linear' or 'logarithmic'
					for cs in self.indiv.stations:
						i,j = cs.coordinates
						#ax.text(i-1, j-1, 'X', ha='center', va='center', color='blue')
						ax.text(i, j, 'X', ha='center', va='center', color='blue')
					colorbar_scale = 'logarithmic'  # 'logarithmic' or 'linear'

					# Initial plot
					if colorbar_scale == 'logarithmic':
						norm = mcolors.LogNorm(vmin=0.01, vmax=max_color_value) # Avoid zero in log scale
					else:
						norm = mcolors.Normalize(vmin=0.01, vmax=max_color_value)

					# Define the colors for the custom colormap
					colors = ["green", "yellow", "red"]  # Define a list of colors
					n_bins = 1000  # Increase this number to make the transitions smoother
					cmap_name = "custom1"

					# Create the colormap
					cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

					#plot = ax.imshow(Psum[1:-1, 1:-1, 0], cmap=cm, interpolation="nearest", norm=norm)
					plot = ax.imshow(Psum[1:-1, 1:-1, 0], cmap=cm, interpolation="nearest", norm=norm)
					colorbar = fig.colorbar(plot, ax=ax, format='%.2f')

					# Creating the animation
					#ani = animation.FuncAnimation(fig, update_plot, times, fargs=(Psum[1:-1, 1:-1, :], plot, fig, ax), interval=200)
					ani = animation.FuncAnimation(fig, update_plot, times, fargs=(Psum[1:-1,1:-1,:], plot, fig, ax), interval=200)

					# Save the animation as a MP4
					#FFwriter = animation.FFMpegWriter(fps=5)
					#ani.save(f"pollution_genetic_gamma_{gamma}.mp4", writer = FFwriter)
					#ani.save(f"pollution_genetic_gamma_{gamma}.gif", writer='imagemagick', fps=5)
					ani.save("pollution2.gif", writer='imagemagick', fps=5)
					#plt.show()
					plt.close()
					'''
			#filtered_dict = {key: original_dict[key] for key in keys_to_keep if key in original_dict}
			cars = {
				CarType.EV: {'position': positionsEV, 'velocity': velocitiesEV, 'acceleration': accelerationsEV},
				CarType.Petrol: {'position': {key:positions[key] for key in  positions if key not in diesel}, 'velocity': {key:velocities[key] for key in  positions if key not in diesel}, 'acceleration': {key:accelerations[key] for key in  positions if key not in diesel}},
				CarType.Diesel: {'position': {key:positions[key] for key in diesel}, 'velocity': {key:velocities[key] for key in diesel}, 'acceleration': {key:accelerations[key] for key in diesel}}
				}
			
			filename = f"P_delta_{delta}_gamma_{gamma}.pkl"
			with open(filename, 'wb') as file:
				pickle.dump(P, file)
			filename = f"G_delta_{delta}_gamma_{gamma}.pkl"
			with open(filename, 'wb') as file:
				pickle.dump(G, file)
			filename = f"cars_delta_{delta}_gamma_{gamma}.pkl"
			with open(filename, 'wb') as file:
				pickle.dump(cars, file)

			with h5py.File('P_delta_{delta}_gamma_{gamma}.h5', 'w') as f:
				for pollutant, vehicles in P.items():
					for vehicle, data in vehicles.items():
						f.create_dataset(f"{pollutant}/{vehicle}", data=data, compression='gzip')
			with h5py.File('G_delta_{delta}_gamma_{gamma}.h5', 'w') as f:
				for pollutant, vehicles in G.items():
					for vehicle, data in vehicles.items():
						f.create_dataset(f"{pollutant}/{vehicle}", data=data, compression='gzip')
			with h5py.File('cars_delta_{delta}_gamma_{gamma}.h5', 'w') as f:
				for vehicle, properties in cars.items():
					for property, data in properties.items():
						f.create_dataset(f"{vehicle}/{property}", data=data, compression='gzip')
			

			for cartype in P:
				for pollutant in P[cartype]:
					P[cartype][pollutant] = P[cartype][pollutant].astype(np.float32)
			#P = {str(key): value for key, value in P.items()}

			saved_times = list(range(0, times+1, max(times//40,2)))

			flattened_P = {}
			averages = {}
			for key1, subdict in P.items():
				for key2, array in subdict.items():
					flattened_P[f"{key1}_{key2}"] = array[1:-1,1:-1,saved_times].astype(np.float32)
					averages[f"P_{key1}_{key2}"] = np.mean(array[1:-1, 1:-1, times//4:], axis=2).astype(np.float32)

			flattened_G = {}
			for key1, subdict in G.items():
				for key2, array in subdict.items():
					flattened_G[f"{key1}_{key2}"] = array[:,:,saved_times].astype(np.float32)
					averages[f"G_{key1}_{key2}"] = np.mean(array[:, :, times//4:], axis=2).astype(np.float32)

			np.savez_compressed(os.path.join(dataSaveDir, f'P_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **flattened_P)
			np.savez_compressed(os.path.join(dataSaveDir, f'G_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **flattened_G)
			#p.seed,p.buildings,p.distributionCS,p.densityCars,p.densityEV,p.densityDiesel,p.windV,p.pollutionRouting
			#np.savez_compressed(f'G_delta_{delta}_gamma_{gamma}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz', **flattened_G)
			
			del P
			del G
			del flattened_P
			del flattened_G
			gc.collect()


			map_cars =  {cartype : {'positions': np.zeros((width, height, times+1)), 'velocities': np.zeros((width, height, times+1)), 'accelerations': np.zeros((width, height, times+1)), 'braking': np.zeros((width, height, times+1))} for cartype in CAR_PROPERTIES}
			for cartype in map_cars:
				for car in positions[cartype]:
					for t in range(times+1):
						a,b=positions[cartype][car][t]
						map_cars[cartype]['positions'][a,b,t]+=1
						map_cars[cartype]['velocities'][a,b,t]+=tuple_norm(velocities[cartype][car][t])
						aux = accelerations[cartype][car][t]
						if aux > 0:
							map_cars[cartype]['accelerations'][a,b,t]+=aux
						else:
							map_cars[cartype]['braking'][a,b,t]-=aux
			flattened_map_cars={}
			for key1, subdict in map_cars.items():
				for key2, array in subdict.items():
					flattened_map_cars[f"{key1}_{key2}"] = array[:,:,saved_times].astype(np.float32)
					averages[f"cars_{key1}_{key2}"] = np.mean(array[:,:,times//4:], axis=2).astype(np.float32)		

			np.savez_compressed(os.path.join(dataSaveDir, f'C_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **flattened_map_cars)
			#np.savez_compressed(f'cars_delta_{delta}_gamma_{gamma}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz', **flattened_map_cars)#p.seed,p.buildings,p.distributionCS,p.densityCars,p.densityEV,p.densityDiesel,p.windV,p.pollutionRouting
			np.savez_compressed(os.path.join(dataSaveDir, f'A_delta_{0.1}_gamma_{0.01}_times_{times}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz'), **averages)
			#np.savez_compressed(f'averages_delta_{delta}_gamma_{gamma}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz', **averages)
			del flattened_map_cars
			del averages
			gc.collect()
			#print('Experimento terminado')

			for key1, subdict1 in cars.items():
				for key2, subdict2 in subdict1.items():
					for key3, array in subdict2.items():
						averages[f"averagecars{key1}_{key2}_{key3}"] = np.mean(array[:, :, 90:], axis=2).astype(np.float32)
			'''
			#averages = {'average_P': np.mean(P[:, :, 90:], axis=2), 'average_G': np.mean(G[:, :, 90:], axis=2), 'average_positions': np.mean(P[:, :, 90:], axis=2), 'average_velocities': np.mean(P[:, :, 90:], axis=2), 'average_accelerations': np.mean(P[:, :, 90:], axis=2)}
			#np.savez_compressed(f'averages_delta_{delta}_gamma_{gamma}_seed_{names[0]}_buildings_{names[1]}_distributionCS_{names[2]}_densityCars_{names[3]}_densityEV_{names[4]}_densityDiesel_{names[5]}_windV_{names[6]}_pollutionRouting_{names[7]}.npz', **averages)
			'''
			def compute_norms(velocity_dict):
				norms = []
				for velocities in velocity_dict.values():
					norms.extend([math.sqrt(x**2 + y**2)*3.6 for x, y in velocities])
				return norms
			
			all_norms = compute_norms(velocities) + compute_norms(velocitiesEV)
			#print('maximum', np.max(all_norms))

			frequency = Counter(all_norms)

			# Prepare data for plotting
			values = list(frequency.keys())
			counts = list(frequency.values())

			# Create the plot
			plt.bar(values, counts, width=0.01, align='center', color='b', alpha=0.7)  # Small width to mimic lines
			plt.xlabel('Value')
			plt.ylabel('Frequency')
			plt.title('Histogram of Discrete Values')
			plt.xticks(values)  # Ensure all unique values are marked on x-axis
			plt.grid(axis='y', linestyle='--', alpha=0.6)  # Optional grid for better visibility
			plt.show()
			'''

			#new_acc = np.ones((width+2, height+2))
			#new_acc[1:-1,1:-1] = acc
			if returnFits:
				return #-carsAtDestination/times+Psum[new_acc==1,:].mean()+Psum[new_acc==1,:].std(), loc_fits
			else:
				return
	
		#print() 
		#self.stats.close()
		
	
	def update(self,frameNum, img, grid, heigh, width):
		try:
			#for i in range(100):
			next(self.city_generator)
		except StopIteration:
			pass
		except Exception as e:
			print(e)
			pass

		# for i in range(heigh):
		#     for j in range(width):
		#         grid[i, j].set_next_state()

		newGrid = np.empty((heigh, width))
		for i in range(heigh):
			for j in range(width):
				#grid[i, j].update_state()
				newGrid[i, j] = grid[i, j].color(self)
				# if newGrid[i, j] == Cell.STREET:
				# 	print(i,j)
				
		# initial_states = np.random.choice([Cell.STREET, Cell.FREE], grid.shape[0]*grid.shape[1] , p=[0.2, 0.8])
		# newGrid = np.array([state for state in initial_states]).reshape(grid.shape[0], grid.shape[1])
		img.set_data(newGrid)
		return img,

	def introduceCarsInCSToStacionaryState(self):
		if not self.p.introduceCarsInCSToStacionaryState:
			return
		
		# count cars in cs
		carInCS=0
		for car in self.cars:
			if car.isCharging():
				carInCS+=1
		# calculate equilibrium of cars in cs in energy terms
		percentageCarsInCS=1/(self.p.carRechargePerTic/self.p.mediumVelocity+1)
		#numberCarsCanCharge = sum([CAR_PROPERTIES[cartype].number for cartype in CAR_PROPERTIES if CAR_PROPERTIES[cartype].needsCharging])
		carsCanCharge = [car for car in self.cars if CAR_PROPERTIES[car.p.type].needsCharging]
		numberCarsCanCharge = len(carsCanCharge)
		equilibrium=numberCarsCanCharge*percentageCarsInCS


		queryCS = random.sample(carsCanCharge, int(self.p.aStarCSQueueQuery*numberCarsCanCharge))
		reserve = random.sample(queryCS, int(self.p.aStarCSReserve*len(queryCS)))
		for car in queryCS:
			car.csqueuequery=True
		for car in reserve:
			car.csreserve=True


		# move cars to equilibrium number
		#while carInCS<equilibrium:
			# get random car
		if carInCS>equilibrium:
			print('Error, should not be as many cars in CS')
		if carInCS<equilibrium:
			n=int(equilibrium-carInCS)
			if n<0:
				print('There are more cars in CS than there should be initially, so equilibrium - carInCS = {n}.')
				n=0
			selectableCars = [car for car in self.cars if CAR_PROPERTIES[car.p.type].needsCharging and not car.isCharging()]
			if len(selectableCars)<n:
				print(f"Not enough cars meet the criteria, only {len(selectableCars)} cars available, but {n} more are needed as {carInCS} are already charging. ", 'Density of EV: ', self.p.densityEV)
				cars=selectableCars
			else:
				cars=random.sample(selectableCars,n)
			for car in cars:
				# move car to cs
				cell=car.cell
				_,_,cs=car.localizeCS(cell,self.t)	
				car.target2=cs.cell
				cell.car=None
				car.cell=cs.cell
				car.enterOnCS()
				carInCS+=1
			'''
			car=random.choice(self.cars)#self.cars[random.randint(0,numberEVCars-1)]
			# if car is not in cs
			if car.p.type == CarType.EV and not car.isCharging():
				# move car to cs
				cell=car.cell
				_,_,cs=car.localizeCS(cell,self.t)	
				car.target2=cs.cell
				cell.car=None
				car.cell=cs.cell
				car.enterOnCS()
				carInCS+=1
			'''
		#if self.p.viewWarning: Hacked 
			#if self.p.numberStations<equilibrium:
			#print("Number of chargers:",self.p.numberChargingPerStation)

	def nextCarType(self):
		total=sum(CAR_PROPERTIES[c].number for c in CAR_PROPERTIES)
		if total==0:
			density = [(c,CAR_PROPERTIES[c].density) for c in CAR_PROPERTIES]
		else:
			density = [(c,CAR_PROPERTIES[c].density-CAR_PROPERTIES[c].number/total) for c in CAR_PROPERTIES]
		r = max(density, key=lambda x: x[1])[0]
		CAR_PROPERTIES[r].number+=1
		return r


	def generator(self):
		#try:
			# Build city streets
		yieldCada=1000
		if self.p.viewDrawCity:
			yieldCada=1
		yieldI=0
		for i in range(self.p.verticalBlocks):
			for j in range(self.p.horizontalBlocks):
				for _ in self.block.draw(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2):
					if self.p.viewDrawCity:
						if yieldCada<=yieldI:
							yieldI=0
							yield
						yieldI+=1
				self.block.semaphore(self.grid,i*self.block.height+self.block.height//2,j*self.block.width+self.block.width//2)
		#aqui anillo
		#'''
		if self.p.ring:
			#self.p.ring=False
			xmax = self.p.horizontalBlocks * self.block.width-1
			ymax = self.p.verticalBlocks * self.block.height-1
			y=ymax
			for x in range(xmax+2): # CHANGED +1 -> +2
				if x==0:
					for _ in self.block.draw2(self.grid, 0, y-1, x, y, 5, True):
						pass
				else:
					for _ in self.block.draw2(self.grid, x-1, y, x, y, 5, True):
						pass
			#print('here4')
			x=xmax
			for y in reversed(range(ymax+2)): # CHANGED +1 -> +2
				if y==ymax:
					for _ in self.block.draw2(self.grid, x-1, ymax, x, y, 5, True):
						pass
				else:
					for _ in self.block.draw2(self.grid, x, y+1, x, y, 5, True):
						pass
			#print('here4')
			y=0
			for x in reversed(range(xmax+2)):
				if x==xmax:
					for _ in self.block.draw2(self.grid, xmax, y+1, x, y, 5, True):
						pass
				else:
					for _ in self.block.draw2(self.grid, x+1, y, x, y, 5, True):
						pass
			#print('here4')
			x=0
			for y in range(ymax+2):
				if y==0:
					for _ in self.block.draw2(self.grid, x+1, 0, x, y, 5, True):
						pass
				else:
					for _ in self.block.draw2(self.grid, x, y-1, x, y, 5, True):
						pass
		self.p.ring=True
		#'''
		# Parameters that depend on the number of streets is calculated here
		p=self.p
		if hasattr(self.p, 'listgenerator'):
			self.listgenerator= [(cellX,cellY) for (cellX,cellY) in itertools.product(range(len(self.grid.grid[0])),range(len(self.grid.grid))) if self.grid.grid[cellY][cellX].state != self.grid.grid[cellY][cellX].FREE]
			self.sizes = (len(self.grid.grid[0]),len(self.grid.grid))
			return
		#print('Got here!!!')

			
		p.numberCars=int(p.densityCars*self.grid.streets)
		# medium velocity of cells in the city
		p.mediumVelocity=self.grid.totalVelocity/self.grid.streets
		p.numberChargers=p.numberChargersPerBlock*p.numberBlocks

		#p.numberChargers=p.numberCars*p.densityEV*p.mediumVelocity/p.carRechargePerTic*p.energy
		if p.numberCars*p.densityEV*p.mediumVelocity/p.carRechargePerTic==0:
			p.energy = np.inf
		else:
			p.energy=p.numberChargers/(p.numberCars*p.densityEV*p.mediumVelocity/p.carRechargePerTic)
		#print("energy",p.energy)
		
		#p.numberChargersPerBlock=p.numberChargers/p.numberBlocks
		p.numberChargingPerStation=p.numberChargers//p.numberStations


		#numberBlocks=self.p.verticalBlocks*self.p.horizontalBlocks
		numberCars=self.p.numberCars
		numberStations=self.p.numberStations
		numberChargingPerStation=self.p.numberChargingPerStation

		self.stats=Stats(self.p)


		if self.indiv != None:       
			for CS in self.indiv.stations:
				if CS.num_chargers == 0:
					print('There should not be any CS with 0 chargers')
					continue
				cell = self.grid.grid[CS.coordinates[1]][CS.coordinates[0]] # WHERE IS NUMBER OF CHARGERS BEING TAKEN INTO ACCOUNT
				currentCS=ChargingStation(p,self.grid,cell,CS.num_chargers)
				self.grid.cs.append(currentCS)

			# Put cs (Charge Stations)
		# self.cs=[]
		# for _ in range(numberStations): #*self.verticalBlocks*self.horizontalBlocks): # number of cs
		# 	self.cs.append(ChargingStation(self.p,self.grid,self.grid.randomStreet(),numberChargingPerStation))
		# 	#if p.viewDrawCity:
		# 	yieldCada=1
		# 	if yieldCada<=yieldI:
		# 		yieldI=0
		# 		yield
		# 	yieldI+=1
		for cs in self.grid.cs:
			cs.createChargins()
			cs.insertInRouteMap()

		# Orden and filter cs by p.opmitimizeCSSearch
		for cell in self.grid.grid.flatten():
			if 0<len(cell.h2cs):
				cell.h2cs.sort(key=lambda x: x.distance)
				cell.h2cs=cell.h2cs[:self.p.opmitimizeCSSearch]

		# Put cars
		self.cars=[]
		CAR_PROPERTIES[CarType.EV].density = self.p.densityEV
		#print(self.p.densityEV)
		#print(CAR_PROPERTIES[CarType.EV].density)
		CAR_PROPERTIES[CarType.Diesel].density = self.p.densityDiesel * (1-self.p.densityEV)
		#self.p.densityPetrol = 1 - self.p.densityDiesel - self.p.densityEV
		CAR_PROPERTIES[CarType.Petrol].density = (1-self.p.densityDiesel) * (1-self.p.densityEV)
		#print('densidades: ', {cartype: CAR_PROPERTIES[cartype].density for cartype in CAR_PROPERTIES})
		if any(CAR_PROPERTIES[cartype].density < 0 for cartype in CAR_PROPERTIES):
			print('Error: densities should sum 1.')
			return
		def assignCarTypes(num_cars):
			# Calculate the exact number of cars for each type based on density
			car_counts = {c: int(CAR_PROPERTIES[c].density * num_cars) for c in CAR_PROPERTIES}
			
			# Adjust counts to ensure the total is correct due to rounding
			total_assigned = sum(car_counts.values())
			difference = num_cars - total_assigned
			if difference > 0:
				# Sort car types by density in descending order
				sorted_types = sorted(CAR_PROPERTIES.keys(), key=lambda c: CAR_PROPERTIES[c].density, reverse=True)
				
				# Distribute the remaining cars to the types with the highest densities first
				for i in range(difference):
					car_counts[sorted_types[i % len(sorted_types)]] += 1

			# Create the list of car types
			car_types = []
			for c, count in car_counts.items():
				car_types.extend([c] * count)

			# Shuffle to randomize the order of car types
			random.shuffle(car_types)
			
			return car_types
		allcartypes = assignCarTypes(numberCars)
		for id in range(numberCars): # number of cars
			p2=self.p.clone()
			#p2.type = self.nextCarType()
			p2.type = allcartypes[id]
			CAR_PROPERTIES[p2.type].number += 1

			'''
			if (id+1)/numberCars<self.p.densityEV:
				p2.type=CarType.EV
			else:
				p2.type=CarType.ICEV
			'''

			self.cars.append(Car(p2,id,self.grid,self.grid.randomStreet(),self.grid.randomStreet(),self.t))
			if self.p.viewDrawCity:
				if yieldCada<=yieldI:
					yieldI=0
					yield
				yieldI+=1

		#print('Density of EV: ', CAR_PROPERTIES[CarType.EV].number/numberCars)
		#print('Density of Diesel: ', CAR_PROPERTIES[CarType.Diesel].number/numberCars)
		#print('Density of Petrol', CAR_PROPERTIES[CarType.Petrol].number/numberCars)
		
		
		self.introduceCarsInCSToStacionaryState()

		# Simulation
		while True:
			#sleep(1)
			self.t+=1
			self.stats.setT(self.t) # INTERCEPTAR
			firstTime=True
			while True:
				if firstTime:
					goPriority=0
				else:
					goPriority=minPriority
				maxPriority=-math.inf
				minPriority=math.inf
				for numcar,car in enumerate(self.cars):
					self.stats.addCarState(numcar,car.state) # Aquí guardar
					if car.isCharging():
						continue
					if firstTime:
						if car.priority<0:
							car.priority=-car.priority
						car.submove=car.cell.velocity

					if car.priority==goPriority:
						car.moveCar(self.t)
					if car.priority<minPriority and 0<car.priority:
						minPriority=car.priority
					if maxPriority<car.priority:
						maxPriority=car.priority
						
				firstTime=False
				if maxPriority<=0:
					break

			for numcs,cs in enumerate(self.grid.cs):
				cs.moveCS(self.t)
				self.stats.addCSQueue(numcs,len(cs.queue))

			yieldCada=1
			if yieldCada<=yieldI:
				yieldI=0
				yield
			yieldI+=1

		#except Exception as e:  # Esto captura cualquier excepción derivada de la clase base Exception
		#    print(traceback.format_exc())  # Esto imprime la traza completa del error

class Grid:
	"""
	Grid is a class that represents the grid of the city. It is a matrix of cells.
	It stores the intersections of the city to make a semaphore. 
	Also coinains several utility functions to calculate the distance between two cells, to get a random street, and 
	to link two cells.
	"""
	def __init__(self, p, heigh, width): #HACKED: añadido grid
		self.p = p
		self.width = width
		self.heigh = heigh
		self.streets=0
		self.totalVelocity=0
		self.intersections = []
		initial_states = np.random.choice([Cell.FREE], width*heigh, p=[1])
		self.grid = np.array([Cell(state) for state in initial_states]).reshape(heigh, width)
		for i in range(heigh):
			for j in range(width):
				self.grid[i, j].x=j
				self.grid[i, j].y=i
				for di in [-1, 0, 1]:
					for dj in [-1, 0, 1]:
						self.grid[i, j].neighbors[di+1][dj+1]=self.grid[(i+di)%heigh, (j+dj)%width]
		self.cs=[]
	def distance(self,x0,y0,x1,y1):
		# latince distance
		dx = abs(x1 - x0)
		dy = abs(y1 - y0)
		if dx > self.width / 2:
			dx = self.width - dx
		if dy > self.heigh / 2:
			dy = self.heigh - dy
		return dx + dy
	
	def randomStreet(self):
		# If the random is fixed and introduced on cars we can reproduce the same simulation
		while True:
			x = random.randint(0,self.width-1)
			y = random.randint(0,self.heigh-1)
			cell=self.grid[y][x]
			if cell.state!=Cell.FREE and cell.car==None:
				return cell
	def link(self,origin,target,velocity):
		if origin!=None:
			if (len(target.origin)==0 and len(target.destination)==0) or (len(origin.origin)==0 and len(origin.destination)==0):
				self.streets+=1
				self.totalVelocity+=velocity
			target.origin.append(origin)
			origin.destination.append(target)
			target.updateColor()
			origin.updateColor()
			target.velocity=velocity
			origin.velocity=velocity
			if len(target.origin)>1:
				self.intersections.append(target)
			
			#autosemaphore
			if self.p.yellowBox:
				for d in target.destination:
					d.semaphore.append(origin)
			else:
				if len(target.origin)>1:
				#target.origin[0].semaphore.append(origin)
				#target.origin[0].origin[0].semaphore.append(origin.origin[0])
					
					for d in target.destination:
					#d.semaphore.append(origin)
						d.semaphore.append(target)
				if len(origin.origin)>1:
					target.semaphore.append(origin)

class Buscador:
	def __init__(self):#,profundidad):
		self.cell=None
		self.father=None
		self.heuristico=0
		self.tiempo=0
		self.time=0
		self.open=False
		self.numberChildren=0 # number of open children



class Car:
	"""
	The car class represents a car of the simulation. The moveCar function is the main function. 
	The car moves from one cell to another. Sometimes it is only one cell, but sometimes 
	there are more than one cell (bifurcation). In this case, the car uses the A* algorithm to find the best path.
	If the car has not enough moves to reach the target, it will try to reach the nearest CS to recharge.
	"""
	def __init__(self,p,id, grid: Grid, cell,target,t):
		self.id=id
		self.state:CarState=CarState.Driving

		self.p=p
		self.grid = grid
		self.cell=cell
		cell.car=self
		self.priority=CarPriority.NoAsigned

		self.target=target
		self.target2=None # if need to recharge

		self.submove=0
		# Change V2 to V3. Why use normal? A normal is a sum of uniform distributions. The normal is not limited to [0,1] but the uniform is. The normal by intervals.
		self.moves=p.carMovesFullDeposity*random.random() 

		# A percentage of cars have CS Queue Query
		self.csqueuequery=False
		self.csreserve=False
		#if self.id<p.aStarCSQueueQuery*p.numberCars:
		#	self.csqueuequery=True
		#	if self.id<p.aStarCSQueueQuery*p.aStarCSReserve*p.numberCars:
		#		self.csreserve=True

		# initial moves must be enough to reach the CS at least
		if CAR_PROPERTIES[self.p.type].needsCharging:
			dis,_,cs=self.localizeCS(cell,t)	
			if self.moves<dis:
				self.target2=cs.cell
				cell.car=None
				self.cell=cs.cell
				self.enterOnCS()
				#self.moves=dis
		
		self.toCell=[]

		self.targets=[]
		last=self.cell
		for i in range(p.carSizeTarget):
			while True:
				cand=self.grid.randomStreet()
				if cand!=last:
					break
			self.targets.append(cand)
		self.goTargets=0


	def inTarget(self,target):
		return self.cell==target
	
	def localizeCS(self,cell,t,distance=0): 
		# return distance, timeToAttend, cs
		if self.p.aStarRandomCS:
			# select random CS
			cs=random.choice(self.grid.cs) 
			return (distance,0,cs)
		if cell.cs!=None:
			if self.csqueuequery and self.p.aStarMethod=="Time":
				# Calculate the ticks (time) to attend the CS
				return (distance,cell.cs.TicksStepsToAttend(),cell.cs)
			else:
				return (distance,0,cell.cs)
		if len(cell.h2cs)==0:
			if len(cell.destination)==1:
				return self.localizeCS(cell.destination[0],t,distance+1)
			else:
				print("Error: in data structure of CS")
		tupla=None

		if 0<self.p.aStarTimeOutCacheBifulcation:
			getFromCache=True
			if cell.timeCache==None:
				cell.timeCache=[0]*len(cell.h2cs)
				getFromCache=False
			time=t-cell.tCache
			if self.p.aStarTimeOutCacheBifulcation<time:
				getFromCache=False
		else:
			getFromCache=False
				
		for ind,aux in enumerate(cell.h2cs):
			cand=distance+aux.distance
			heuristic=cand
			if self.p.aStarMethod=="Time":
				if getFromCache:
					heuristic=cell.timeCache[ind]
				else:
					heuristic,_=self.aStar(cell,aux.cs.cell,t)					
					if 0<self.p.aStarTimeOutCacheBifulcation:
						cell.timeCache[ind]=cand
						cell.tCache=t
				
				if self.csqueuequery:
					heuristic+=aux.cs.TicksStepsToAttend()

			#  add tickssteps to attended?
			if tupla==None or heuristic<tupla[1]:
				tupla=(cand,heuristic,aux.cs)
		return tupla
	
		# si es por distancia, solo rellena 1
		# aux=cell.h2cs[0]
		# 
		# return (distance+aux.distance,aux.cs)
	
	def isCharging(self):
		return self.state==CarState.Queueing or self.state==CarState.Charging

	def checkLegalMove(self,cell,toCell):
		dif=abs(self.cell.x-toCell.x)+abs(self.cell.y-toCell.y)
		if dif!=1:
			print("(",cell.x,",",cell.y,") -> (",toCell.x,",",toCell.y,")")
			print("Error in move, no neighbor")

	def enterOnCS(self):
		if not self.isCharging() and CAR_PROPERTIES[self.p.type].needsCharging:
			# enter on CS
			self.cell.car=None
			self.cell=None
			cs=self.target2.cs
			cs.queue.append(self)
			cs.carsInCS+=1
			self.target2.car=None
			self.state=CarState.Queueing
			# remove from reserve
			if self.csreserve:
				try:
					cs.reserve.remove(self)
				except:
					pass
			return True
		return False

	def calculateRoute(self,cell,t):
		dis,ire=self.aStar(cell,self.target,t)

		if self.p.type !=CarType.EV:
			# If the car is ICEV, it will not need to recharge
			self.toCell=ire#[ire[0]] para calcular en cada movimiento
			return 

		if len(ire)==0:
			print('dis=', dis)
			p=self.p
			file_name = f'P_delta_{0.1}_gamma_{0.01}_times_{2000}_seed_{p.seed}_buildings_{p.buildings}_distributionCS_{p.distributionCS}_densityCars_{p.densityCars}_densityEV_{p.densityEV}_densityDiesel_{p.densityDiesel}_windV_{p.windV}_pollutionRouting_{p.pollutionRouting}.npz'
			print("\t\t-Error: in data structure of A*")
			print(f"\t\t\t--{t}-",file_name)
			print(f"\t\t\t--{t}-",cell.x, " - ", cell.y)
			print(p.id)
		dis2,_,_=self.localizeCS(self.target,t)
		if self.moves<dis+dis2:
			self.state=CarState.ToCharging
			# There are not enough moves, need to recharge in CS first
			dis3,_,cs=self.localizeCS(cell,t)
			if self.moves<dis3:
				# There are not enough moves, event with recharge in CS
				# In this version we allow negative moves (energy)
				# We don't have studied the case of cars withou energy and how to recharge them
				pass
			ire=self.aStar(cell,cs.cell,t)[1]
			self.target2=cs.cell
			if self.csreserve:
				cs.reserve.append(self)
		self.toCell=ire

	def moveCar(self,t):
		if self.inTarget(self.target):
			if self.state==CarState.Destination:
				self.target=self.targets[self.goTargets]
				self.goTargets=(self.goTargets+1)%len(self.targets)
			else:
				self.state=CarState.Destination
				self.cell.t=t
				return 
		if self.inTarget(self.target2):
			# There is no error if pass over target2, because target2 is only set when there is need to recharge	
			if self.enterOnCS():
				return
		
		cell=self.cell

		def calculateNext(toCell):
			self.calculateRoute(toCell,t)

		if 1==len(cell.destination):
			toCell=cell.destination[0]
		elif 0<len(self.toCell):
			toCell=self.toCell.pop(0)
		else:
			calculateNext(cell)
			toCell=self.toCell.pop(0)

		#if toCell.t==t or toCell.car!=None:
		#	if self.p.yellowBox:
		#		for s in cell.semaphore:
		#			s.t=t
		isStop=toCell.t==t or toCell.car!=None
		if not isStop and (len(toCell.destination)>1 or len(toCell.origin)>1):
			if 1==len(toCell.destination):
				toCell2=toCell.destination[0]
			elif 0<len(self.toCell):
				toCell2=self.toCell[0]
			else:
				calculateNext(toCell)
				toCell2=self.toCell[0]
			isStop=toCell2.t==t or toCell2.car!=None

			if isStop:
				# Ten cicles waiting, move to the other cell if possible
				if t-toCell2.t0>self.p.timeStop:
					for other in toCell.destination:
						if other!=toCell2 and other.car==None:
							isStop=False
							self.toCell=[other]
							break

		if isStop: 
			for s in cell.semaphore:
				s.t=t+1
			
			cell.occupation+=1
			if 0<self.p.aStarUseCellExponentialWeight:
				a=math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)
				b=(1-self.p.aStarUseCellExponentialWeight)
				c=1-a-b
				cell.exponentialOccupation=cell.exponentialOccupation*a+c*1/cell.velocity+b*1
				#cell.exponentialOccupation=cell.exponentialOccupation*math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)+(1-self.p.aStarUseCellExponentialWeight)
				cell.exponentialLastT=t
			if 1<len(cell.destination):
				self.toCell.insert(0,toCell)
			self.state=CarState.Waiting
			self.priority=-abs( self.priority)
			return
		
		# Execute move
		# identifica si es ilegal, no join
		# self.checkLegalMove(cell,toCell)
		#print("(",cell.x,",",cell.y,") -> (",toCell.x,",",toCell.y,")")
		#if self.p.yellowBox:
		#	for s in toCell.semaphore:
		#		s.t=t
		self.cell = toCell
		cell.car = None
		toCell.car = self
		cell.t=t
		toCell.t0=t
		cell.occupation+=1/cell.velocity
		if 0<self.p.aStarUseCellExponentialWeight:
			a=math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)
			# b=(1-self.p.aStarUseCellExponentialWeight)
			# c=1-a-b
			# cell.exponentialOccupation=cell.exponentialOccupation*a+c*1/cell.velocity+b*1/cell.velocity
			d=1-a
			cell.exponentialOccupation=cell.exponentialOccupation*a+d*1/cell.velocity
			#cell.exponentialOccupation=cell.exponentialOccupation*math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)+(1-self.p.aStarUseCellExponentialWeight)*1/cell.velocity
			cell.exponentialLastT=t
		#if not self.p.yellowBox:
		#	for s in cell.semaphore:
		#		s.t=t


		# Calculate priority
		if len(toCell.destination)==1:
			toCell=toCell.destination[0]
			if self.target2!=None:
				self.state=CarState.ToCharging
			else:
				self.state=CarState.Driving
		else:
			if len(self.toCell)==0:
				calculateNext(toCell)
			toCell=self.toCell[0]
	
		if not self.cell in toCell.origin:
			self.checkLegalMove(self.cell,toCell)
			print("Next error")
		self.priority=toCell.origin.index(self.cell)+1

		# Update energy (moves) and sub-moves (velocity)
		self.moves-=1
		self.submove-=1
		if self.submove==0:
			#print("End")
			# negative priority is used to indicate that the car finished the submove
			self.priority=-abs(self.priority)

	def aStar(self,cell:Cell,target:Cell,t):
		if not hasattr(self.grid,"aStarPerMillisecond"):
			self.grid.aStarTotal=0
			self.grid.aStarPerMillisecond=0
		since=datetime.datetime.now()
		aux= getattr(self,"aStar"+self.p.aStarMethod)(cell,target,t)
		now=datetime.datetime.now()
		millis=(now-since).microseconds/1000
		self.grid.aStarTotal+=1
		self.grid.aStarPerMillisecond+=millis
		#print("millis per aStar:",self.grid.aStarPerMillisecond/self.grid.aStarTotal)
		return aux
	'''
	def aStar2(self,cell:Cell,target:Cell,t):
		# Reserva espacio fijo
		buscadores=500

		buscador=[Buscador() for _ in range(buscadores)]

		buscador[0].b[0].cell=cell

		# Localiza mejor buscador no abierto
		imejor=-1
		for i in range(1,buscadores):
			if not buscador[i].open:
				if imejor==-1 or buscador[i].time<buscador[imejor].time:
					imejor=i

		# abre mejor
		buscador[imejor].open=True
		current=buscador[imejor].cell
		#buscar próxima celda con bifurcación, sumando heurístico
		time=buscador[imejor].time
		while len(current.destination)==1:		
			current=current.destination[0]
			#time+=current.		
		#mira si esta en ...
		for b in buscador:
			if b.cell==current:
				pass

	def aStar(self,cell:Cell,target:Cell,t):
		return getattr(self,"aStar"+self.p.aStarMethod)(cell,target,t)
	'''
	 
	def aStarDistance(self,cell:Cell,target:Cell,t):
		# Distance version
		# only mark visited if it has more than one destination
		visited=set()
		visited.add(cell)
		opened={}
		for d in cell.destination:
			if len(cell.destination)==1:
				opened[d]=[]
			else:
				opened[d]=[d]
		opened2={}
		distancia=1

		while True:
			# solo se añaden los visited con mas de uno
			for (o,r) in opened.items():
				if len(o.destination)==1:
					opened2[o.destination[0]]=r
				else:
					if o not in visited:
						visited.add(o)
						for d in o.destination:
							r2=r.copy()
							r2.append(d)
							opened2[d]=r2
				if o==target:
					return (distancia,opened[o])
			opened=opened2
			opened2={}
			distancia+=1

	def aplicarHijos(self,buscador,father, mejora):
		for b in buscador:
			if b.father==father:
				b.heuristico+=mejora
				b.tiempo+=mejora
				if b.tiempo<0:
					print("Error: negative time")
				self.aplicarHijos(buscador,b,mejora)

	def boorarHijos(self,buscador,padre):
		for b in buscador:
			if b.father==padre:
				b.cell=None
				b.father=None
				b.heuristico=0
				b.tiempo=0
				b.open=False
				b.numberChildren=0
				self.boorarHijos(buscador,b)

	def aplicarPadre(self,hijo,padre,mejora):
		if padre==None:
			return
		if padre.mejorHijo==hijo:
			padre.heuristico+=mejora
			self.aplicarPadre(padre,padre.father,mejora)

	def path(self,buscador,lista):
		if buscador.father==None:
			return
		self.path(buscador.father,lista)
		lista.append(buscador.cell)




	def aStarTimeV2(self,cell:Cell,target:Cell,t,buscadores=100):
		buscador=[Buscador() for _ in range(buscadores)]

		buscador[0].cell=cell

		def tiempoDe(cell):
			currentTime=0
			if self.p.aStarUseCellAverageVelocity and 0<cell.t:
				if 0<self.p.aStarUseCellExponentialWeight:
					a=math.pow(self.p.aStarUseCellExponentialWeight,t-cell.exponentialLastT)
					b=1-a
					currentTime=cell.exponentialOccupation*a+b*1/cell.velocity
				else:
					'''
					if self.p.pollutionRouting:
						currentTime=cell.pollutionLevel/cell.t
					else:
						currentTime=cell.occupation/cell.t
					'''
					currentTime=cell.occupation/cell.t
			else:
				currentTime=1/cell.velocity
			return currentTime

		totalTime=0
		totalDistance=0

		while True:
			# Localiza mejor buscador no abierto
			father=None
			for cand in buscador:
				if not cand.open and cand.cell!=None:
					if father==None or father.heuristico>cand.heuristico:
						father=cand

			if father==None:
				try:
					return self.aStarTime(cell,target,t,buscadores*10)
				except RecursionError as e:
					print("Error: profundidad máxima de recursión excedida. V21")

			father.open=True
			current=father.cell

			#buscar próxima celda con bifurcación, sumando heurístico
			currentTimeSegment=father.tiempo+tiempoDe(current)
			while len(current.destination)==1:		
				if current==target:
					# me falta marcar el mejor hijo, para reemplazo
					path=[]
					try:
						self.path(father,path)
					except RecursionError as e:
						print("Error: profundidad máxima de recursión excedida. V22")
					return (currentTimeSegment,path)
					# timeV1,pathV1=self.aStarTimeV1(cell,target,t)
					# # compare path with pathV1 if are different
					# # if len(path)!=len(pathV1):
					# # 	print("Error: path different")
					# # for i in range(len(path)):
					# # 	if path[i]!=pathV1[i]:
					# # 		print("Error: path different")

					# return (timeV1,pathV1)
				current=current.destination[0]
				# comprueba si es target
				currentTimeSegment+=tiempoDe(current)


			for d in current.destination:
				dOrigin=abs(d.x-cell.x)+abs(d.y-cell.y)
				totalDistance+=dOrigin
				totalTime+=currentTimeSegment		
				#print("totalDistance",totalDistance)

				distancia=abs(d.x-target.x)+abs(d.y-target.y)

				time2=distancia*totalTime/totalDistance
				heuristico=time2+currentTimeSegment

				esta=None
				for b in buscador:
					if b.cell==d:
						esta=b
						break
				if esta!=None:
					if esta.tiempo>currentTimeSegment:
							# Mueve la cuenta.
							father.numberChildren+=1
							esta.father.numberChildren-=1

							# Encuentra una mejor solución procede a su reemplazo
							#mejora=heuristico-esta.heuristico
							esta.heuristico=heuristico
							esta.tiempo=currentTimeSegment
							esta.father=father
							esta.open=False
							esta.numberChildren=0
							try:
								self.boorarHijos(buscador,esta)
							except RecursionError as e:
								print("Error: profundidad máxima de recursión excedida. V23")
							# if esta.open:
							# 	self.aplicarHijos(buscador,esta,mejora)
				else:
					# si no está inserta, la cabeza y los hijos operas...
					# Localiza candidato a reemplazar
					minimo=None
					for i,b in enumerate(buscador):
						if b.numberChildren==0:
							if minimo==None or b.cell==None or b.heuristico>minimo.heuristico:
								minimo=b
								if b.cell==None:
									break
					if minimo.cell==None or minimo.heuristico>heuristico:
						father.numberChildren+=1
						minimo.cell=d
						if minimo.father!=None:
							minimo.father.numberChildren-=1
						minimo.father=father
						minimo.open=False
						minimo.heuristico=heuristico
						minimo.tiempo=currentTimeSegment
						minimo.numberChildren=0

	def aStarTime(self,cell:Cell,target:Cell,t):
		# Time version
		visited={}
		opening=[TimeNode(self.p,i) for i in range(self.p.aStarDeep)]
		opening[0].setCell(self.grid,cell,target,0)
		opening[0].decision=[]

		def selectBest():
			'''
			If reached target we can't select
			Only best of best is super-momorized, the others in target are destroyed
			'''
			bestTarget=None
			best=None
			for i in range(0,len(opening)):
				if opening[i].cell==None:
					continue
				if opening[i].cell==target:
					if bestTarget==None:
						bestTarget=opening[i]
					else:
						if bestTarget.heuristic()>opening[i].heuristic():
							bestTarget.cell=None
							bestTarget=opening[i]
					continue
				elif best==None or opening[i].heuristic()<best.heuristic():
					best=opening[i]

			if best==None:
				#print('Error tipo 1')
				return bestTarget
			# is valid?
			if bestTarget!=None and bestTarget.heuristic()<best.heuristic():
				return bestTarget
			if best==None:
				print("best is None")
			return best

		def selectWorst():
			worst=None
			for i in range(0,len(opening)):
				if opening[i].cell==None:
					return opening[i]
				if worst==None or opening[i].heuristic()>worst.heuristic():
					worst=opening[i]
			return worst
		
		best=selectBest()
		# movesToStop=0
		#print("Target: (",target.x,",",target.y,")")
		while True:
			# Go through all the targets of the best
			yet=visited.get(best.cell)
			if yet==None or best.time<yet:
				visited[best.cell]=best.time
				for d in best.cell.destination:
					# When the target is reached, it is not necessarily finished				
					worst=selectWorst()
					worst.backup() 

					worst.setCell(self.grid,d,target,best.distance+1)
					if self.p.pollutionRouting:
						worst.time=best.time+cell.pollutionLevel#/cell.t
					else:
						if self.p.aStarUseCellAverageVelocity and 0<cell.t:
							if 0<self.p.aStarUseCellExponentialWeight:
								a=math.pow(self.p.aStarUseCellExponentialWeight,t-best.cell.exponentialLastT)
								b=1-a
								worst.time=best.time+best.cell.exponentialOccupation*a+b*1/best.cell.velocity							
								#worst.time=best.time+best.cell.exponentialOccupation*math.pow(self.p.aStarUseCellExponentialWeight,t-best.cell.exponentialLastT)
							else:
								worst.time=best.time+cell.occupation/cell.t
								#worst.time=best.time+cell.occupation/cell.t
						else:
							worst.time=best.time+1/best.cell.velocity
						if 0<self.p.aStarAddRoadCarAsTimeSteps and best.cell.car!=None:
							worst.time+=self.p.aStarAddRoadCarAsTimeSteps

						

					# clone copy of decision list
					worst.decision=best.decision.copy()
					# if d comes from a bifurcation add

					if len(best.cell.destination)>1 and len(worst.decision)<self.p.aStarCalculateEach:#HACKED > changed to >=
						worst.decision.append(d)
					
					worst.undoBackupIfWorst()
			# free best
			best.cell=None

			# if reached target we can't select
			# only best of best is super-momorized
			best=selectBest()

			#print("(",best.cell.x,",",best.cell.y,")",best.remainder)

			# stops criterias:
			# if reached target and others are worst

			#HACKED
			if best==None:
				file_name = os.path.join(dataSaveDir, f'P_delta_{0.1}_gamma_{0.01}_times_{2000}_seed_{p.seed}_buildings_{p.buildings}_distributionCS_{p.distributionCS}_densityCars_{p.densityCars}_densityEV_{p.densityEV}_densityDiesel_{p.densityDiesel}_windV_{p.windV}_pollutionRouting_{p.pollutionRouting}.npz')
				print('Here is the error: ', file_name)

			if best.cell==target:
				stopCriteria=True
				for d in opening:
					if d.heuristic()<best.heuristic():
						stopCriteria=False
						break
				if stopCriteria:
					return best.distance,best.decision
			# by number of moves
			# movesToStop+=1
			# if self.p.aStarStepsPerCar<movesToStop:
			#  	return best.distance,best.decision
				
class TimeNode:
	'''
	Used by A* algorithm. It is a node of the A* tree.
	'''
	def __init__(self,p,id):
		self.p=p
		self.id=id
		self.cell=None
		self.time=0
		self.decision=None
		self.remainder=0
		self.distance=0
	def backup(self):
		self.backupCell=self.cell
		self.backupTime=self.time
		self.backupDecision=self.decision
		self.backupRemainder=self.remainder
		self.backupHeuristic=self.heuristic()
		self.backupDistance=self.distance
	def undoBackupIfWorst(self):
		if self.backupCell!=None and self.heuristic()>self.backupHeuristic:
			self.cell=self.backupCell
			self.time=self.backupTime
			self.decision=self.backupDecision
			self.remainder=self.backupRemainder
			self.distance=self.backupDistance
	def heuristic(self):
		if self.cell==None:
			return math.inf
		return self.time+self.remainder*self.p.aStarRemainderWeight
	def setCell(self,grid:Grid,cell:Cell,target:Cell,distance):
		self.cell=cell
		self.remainder=grid.distance(cell.x,cell.y,target.x,target.y)
		self.distance=distance

class TimeNode2:
	'''
	Used by A* algorithm. It is a node of the A* tree.
	This is the version 2, in this version the visited is not unloaded.
	'''
	def __init__(self,p,id):
		self.p=p
		self.id=id
		self.cell=None
		self.time=0
		self.parent=None
		self.childs=0
		self.remainder=0
		self.distance=0
	def backup(self):
		self.backupCell=self.cell
		self.backupTime=self.time
		self.backupDecision=self.decision
		self.backupRemainder=self.remainder
		self.backupHeuristic=self.heuristic()
		self.backupDistance=self.distance
	def undoBackupIfWorst(self):
		if self.backupCell!=None and self.heuristic()>self.backupHeuristic:
			self.cell=self.backupCell
			self.time=self.backupTime
			self.decision=self.backupDecision
			self.remainder=self.backupRemainder
			self.distance=self.backupDistance
	def heuristic(self):
		if self.cell==None:
			return math.inf
		return self.time+self.remainder*self.p.aStarRemainderWeight
	def setCell(self,grid:Grid,cell:Cell,target:Cell,distance):
		self.cell=cell
		self.remainder=grid.distance(cell.x,cell.y,target.x,target.y)
		self.distance=distance

class Stats:
	def __init__(self,p):
		self.p=p
		self.statsFileName=p.statsFileName+p.fileName
		if p.statsFileName=="":
			return
		self.carStateFile=None

	def setT(self,t):
		if self.statsFileName=="":
			return
		if self.carStateFile==None:
			self.t=0		

			self.carStateFile=open(self.statsFileName+"_carstate.json","w")
			self.car=[]
			
			self.csQueueFile=open(self.statsFileName+"_csqueue.json","w")
			self.cs=[]
		else:
			json.dump(self.car,self.carStateFile)
			self.carStateFile.write("\n") 

			json.dump(self.cs,self.csQueueFile)
			self.csQueueFile.write("\n")  
		self.t=t

	def close(self):
		if self.statsFileName=="":
			return
		self.carStateFile.close()
		self.csQueueFile.close()

	def addCarState(self,num,carState):
		if self.statsFileName=="":
			return
		while len(self.car)<=num:
			self.car.append({})
		self.car[num]=carState	

	def addCSQueue(self,num,queue):
		if self.statsFileName=="":
			return
		while len(self.cs)<=num:
			self.cs.append(0)
		self.cs[num]=queue	

	def plotCS(self,view=True):
		# Load data from file
		data_over_time = []
		with open(self.statsFileName+"_csqueue.json", 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))

		# Calculate standard deviation per line
		std = [np.std(arr) for arr in data_over_time]

		# Plot std
		x = range(len(data_over_time))
		plt.plot(x, std)
		plt.xlabel('Time Steps')
		plt.ylabel('Standard Deviation Queue')

		# Plot using a stacked area plot
		# x = range(len(data_over_time))
		# plt.plot(x, data_over_time)
		# plt.xlabel('Time Steps')
		# plt.ylabel('Number of Vehicles')

		# Put in title verticalBlocks, horizontalBlocks, aStarMethod and aStarCSQueueQuery
		#plt.title(self.p.legendName)
		#plt.xticks(ticks=range(len(data_over_time)), labels=[f'{i+1}' for i in range(len(data_over_time))])
		#plt.legend(loc='lower left')  
		plt.legend(loc='lower right')#, bbox_to_anchor=(1, 1.05))


		#plt.savefig(self.p.statsFileName+self.p.fileName+"_csqueue.eps" , format='eps', dpi=600)
		plt.savefig(self.p.statsFileName+self.p.fileName+"_csqueue.pdf" , format='pdf', dpi=600)
		if view:
			plt.show() 
		else:
			plt.close()
		#print()		

	def plot(self,view=True):
		# Load data from file
		data_over_time = []
		with open(self.statsFileName+"_carstate.json", 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))

		all_values = list(CarState)

		counts_over_time = {value: [arr.count(value.value) for arr in data_over_time] for value in all_values}

		# Plot using a stacked area plot
		x = range(len(data_over_time))
		prev_values = np.zeros(len(data_over_time))
		for value in all_values:
			current_values = prev_values + np.array(counts_over_time[value])
			plt.fill_between(x, prev_values, current_values, label=f'{value.name}')
			prev_values = current_values

		plt.xlabel('Time Steps')
		plt.ylabel('Number of Vehicles')

		# Put in title verticalBlocks, horizontalBlocks, aStarMethod and aStarCSQueueQuery
		#plt.title(self.p.legendName)
		#plt.xticks(ticks=range(len(data_over_time)), labels=[f'{i+1}' for i in range(len(data_over_time))])
		#plt.legend(loc='lower left')  
		plt.legend(loc='lower right')#, bbox_to_anchor=(1, 1.05))

		#plt.savefig(self.p.statsFileName+self.p.fileName +"_carstate.eps" , format='eps', dpi=600)
		plt.savefig(self.p.statsFileName+self.p.fileName+"_carstate.pdf" , format='pdf', dpi=600)
		if view:			
			plt.show()
		else:
			plt.close()

class MetaStats2:
	def __init__(self,labelx,fx,labely,fy,labelz,fz,filter=None,scatter=False):
		# filter return true if the experiment is not included
		self.labelx=labelx
		self.fx=fx
		self.labely=labely
		self.fy=fy
		self.labelz=labelz
		self.fz=fz
		self.filter=filter
		self.scatter=scatter

	def Clone(self):
		return MetaStats2(self.labelx,self.fx,self.labely,self.fy,self.labelz,self.fz,self.filter)

class MetaStats:

	def __init__(self):
		self.ps=cartesianExperiment()

		withOutDistance=lambda p:("Only Time",p.aStarMethod=="Distance")
		withOutTime=lambda p:("Only Distance",p.aStarMethod=="Time")

		mss=[
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Use Cell Exponential Weight",lambda p:str(p.aStarUseCellExponentialWeight),
				withOutDistance),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Density Cars",lambda p:str(int(p.densityCars*100))+"% "+p.aStarMethod),
			# MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
			# 	"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
			# 	"By strategy",lambda p:p.legendName.replace(" aStarCSQueueQuery:"+str(p.aStarCSQueueQuery),"")),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Method Study",lambda p:p.aStarMethod),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Standard Deviation Queue",lambda p:self.stdCSQueue(p),
				"Energy Study per number of Chargers per CS STD",lambda p:str(p.numberChargersPerBlock)+(" chargers" if p.numberChargersPerBlock!=1 else " charger"),
				withOutDistance),
			MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Average Queue",lambda p:self.averageCSQueue(p),
				"Energy Study: per number of Chargers per CS AVE",lambda p:str(p.numberChargersPerBlock)+(" chargers" if p.numberChargersPerBlock!=1 else " charger"),
				withOutDistance),
		]
		# Duplicate al MetaStats2 with Proditivity
		mss2=[]
		for ms in mss:
			mss3=ms.Clone()
			mss3.labely="Productivity (Average of Destinantion)"
			mss3.fy=lambda p:self.carState(p,CarState.Destination)
			mss3.labelz=ms.labelz+" PS"
			mss2.append(mss3)
			mss2.append(ms)

		mss2.append(MetaStats2("% CS Queue Query Penetration",lambda p:int(p.aStarCSQueueQuery*100),
				"Productivity (Average of Destinantion)",lambda p:self.carState(p,CarState.Destination),
				"Method Study",lambda p:p.aStarMethod,
				withOutDistance,True))
				#"Density Cars",lambda p:str(int(p.densityCars*100))+"% "+p.aStarMethod),

		for ms in mss2:
			self.execute(ms)

	def execute(self,ms):
		ps=self.ps
		self.strategy={}
		for id,p in enumerate(ps):
			if ms.filter!=None:
				(nameFilter,filter)=ms.filter(p)
				ms.nameFilter=nameFilter
				if filter:
					continue
			x=ms.fx(p)
			y=ms.fy(p)
			# is is nan
			if y!=y:
				print("nan")
				y=ms.fy(p)
			l=ms.fz(p)
			if l not in self.strategy:
				self.strategy[l]=[]
			self.strategy[l].append((x,y))

		for l in self.strategy:
			x2ys={}
			for (x,y) in self.strategy[l]:
				if x not in x2ys:
					x2ys[x]=[]
				x2ys[x].append(y)
			# calculate average and std
			xs=[]
			ys=[]
			# std=[]
			sortx=sorted(x2ys.keys())
			for x in sortx:
				xs.append(x)
				ys.append(np.average(x2ys[x]))
				# std.append(np.std(x2ys[x]))
			self.strategy[l]=(xs,ys)
		self.plot(ms,False)

	def plot(self,ms,view=True):
		plt.figure(figsize=(12,7))
		plt.rcParams.update({'font.size': 18})
		for s in self.strategy:
			(xs,ys)=self.strategy[s]
			#plt.plot(xs, ys, '-o', label=s)

			plt.scatter(xs, ys, color='red', s=50)  # s es el tamaño de los puntos
			coeficientes = np.polyfit(xs, ys, 1)
			recta = np.poly1d(coeficientes)
			plt.plot(xs, recta(xs), "--", color="gray")
			
			yhat = recta(xs)
			ybar = np.mean(ys)
			ssreg = np.sum((yhat - ybar)**2)
			sstot = np.sum((ys - ybar)**2)
			r2 = ssreg / sstot
			plt.text(0.05, 0.95, f"$R^2 = {r2:.2f}$", transform=plt.gca().transAxes, ha="left", va="top")



		plt.ylabel(ms.labely)
		plt.xlabel(ms.labelx)
		if ms.filter!=None:
			title=ms.labelz+" ("+ms.nameFilter+")"
		else:
			title=ms.labelz

		#plt.title(title)
		plt.legend()
		plt.grid(True)
		plt.tight_layout()

		#plt.savefig("metastats/"+title+".eps" , format='eps', dpi=600)
		# if not exists directory, create it
		dir=self.ps[0].metastatsFileName
		if not os.path.exists(dir):
			os.makedirs(dir)
		plt.savefig(dir+title+".pdf" , format='pdf', dpi=600)
		#plt.savefig("metastats/"+title+".pdf" , format='pdf', dpi=600)
		if view:
			plt.show()
		else:
			plt.close()

	def averageCSQueue(self,p):
		# Load data from file
		data_over_time = []
		with open(p.statsFileName+p.fileName+"_csqueue.json", 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))
		average = [np.average(arr) for arr in data_over_time]
		average2 = np.average(average)
		return average2
	
	def stdCSQueue(self,p):
		# Load data from file
		data_over_time = []
		with open(p.statsFileName+p.fileName+"_csqueue.json", 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))
		std = [np.std(arr,ddof=1) for arr in data_over_time]
		average2 = np.average(std)
		return average2
	
	def carState(self,p,state):
		# Load data from file
		data_over_time = []
		fileName=p.statsFileName+p.fileName+"_carstate.json"
		if not os.path.isfile(fileName):
			print("Not exists file: "+fileName)
		with open(fileName, 'r') as file:
			for line in file:
				data_over_time.append(json.loads(line))
		all_values = list(CarState)
		# Number of state over time
		counts_over_time = {value: [arr.count(state) for arr in data_over_time] for value in all_values}
		average = [np.average(arr) for arr in counts_over_time.values()]
		average2 = np.average(average)
		return average2



def cartesianExperiment():
	p=Parameters()
	ps=p.metaExperiment(
		#energy=[0.15,0.3,0.45,0.6,0.75,0.9],
		seed=[12,34,1,78],#,90],
		buildings=[True,False],
		distributionCS=[0,1,2],
		#numberChargersPerBlock=[1,5,10],#cambiar por las 3 config
		#aStarMethod=["Time"],#"Distance"],#solo tiempo, o también evitar contaminación?
		#aStarCSQueueQuery=[0,0.25,0.5,0.75,1], 
		#aStarCSQueueQuery=[0.5],#poner el óptimo (ver paper)
		#aStarCSReserve=[0.5],#<= que el anterior, x primeros. ¿de los que consultan, qué porcentaje reservan? comprobar
		densityCars=[0.05, 0.15, 0.25, 0.35],#[0.05,0.1,0.15,0.25,0.5],#,#añadir porcentaje de tipos
		densityEV=[0.05, 0.35, 0.65, 0.95],#[0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1],#[10,20,30,40,50,75,100],
		densityDiesel=[0, 0.25, 0.5, 0.75, 1],#[0, 0.05, 0.10, 0.25, 0.4, 0.5, 0.6],
		windV = [
			(0, 0),  # Deja los ceros como enteros
			(np.float32(0.1), 0),
			(np.float32(0.3), 0),
			(np.float32(0.5), 0),
			(np.float32(0.2), np.float32(0.2)),
			(np.float32(0.5), np.float32(0.3)),
			(np.float32(1), np.float32(0.5))
		],
		pollutionRouting=[False]#,True]#,True]
		#(WE,WN)
		#[0,5,10,20,50]
		#aStarUseCellExponentialWeight=[0.5],#mirar cuál daba mejores resultados y usar solo ese ('0.95??)
		#reserveCS=[False], # it has been removed because legend is too long
	)
	ps2=[]
	for p in ps:
		ok=True
		'''
		if p.aStarMethod=="Distance":
			if p.aStarCSQueueQuery!=0:
				ok=False
			if p.aStarUseCellExponentialWeight!=0.95:
				ok=False
		
		if p.densityEV+p.densityDiesel>1:
			ok=False
		#if p.densityCars>0.25 and (p.seed,p.buildings,p.distributionCS,p.windV,p.densityEV,p.densityDiesel,p.pollutionRouting)!=(12,True,1,(0.1,0),0.25,0.35,False):
		#	ok=False
		
		if p.densityEV+p.densityDiesel==1 and (p.seed,p.buildings,p.distributionCS,p.windV,p.densityCars,p.pollutionRouting)!=(12,True,1,(0.1,0),0.15,False):
			ok=False
		if p.densityEV+p.densityDiesel==0 and (p.seed,p.buildings,p.distributionCS,p.windV,p.densityCars,p.pollutionRouting)!=(12,True,1,(0.1,0),0.15,False):
			ok=False
		if (p.densityEV,p.densityDiesel) in [(0,0.05),(0,0.65),(0.05,0),(0.45,0)]:
			ok=False
		if (p.densityEV,p.densityDiesel) in [(0,0.35),(0.25,0),(0,0)] and (p.seed,p.buildings,p.distributionCS,p.windV,p.densityCars,p.pollutionRouting)!=(12,True,1,(0.1,0),0.15,False):
			ok=False
		'''
		if ok:
			p.id=len(ps2)
			ps2.append(p)
	return ps2

def experiment(i, numSave=0,run_all=True,view=False,cache=False,indiv=None, cnnExpDataSave=None, returnFits = False,
	 contaminationExp = False, numTimesteps = 2000, delta = 0.1, corner_factor = 1, gamma = 0.01, acc = None, dif_matrix = None, 
	 wind = None, timeTest = None):
	t_start = time.time()
	p=cartesianExperiment()[i]

	for cartype in CAR_PROPERTIES:
		CAR_PROPERTIES[cartype].number=0

	#if exists file of experiment, skip
	if cache:
		if not view and os.path.isfile(p.statsFileName+p.fileName+"_carstate.json") and os.path.isfile(p.statsFileName+p.fileName+"_csqueue.json"):
			#if size is 0
			if os.path.getsize(p.statsFileName+p.fileName+"_carstate.json")>0 and os.path.getsize(p.statsFileName+p.fileName+"_csqueue.json")>0:
				return

	random.seed(p.seed)
	#p.densityCars=0.05 #HACKED
	p.aStarMethod="Time"#"Distance"],#solo tiempo, o también evitar contaminación?
	p.aStarCSQueueQuery=0.#poner el óptimo (ver paper)
	p.aStarCSReserve=0.#<= que el anterior, x primeros. ¿de los que consultan, qué porcentaje reservan? comprobar
	#densityCars=[0.05,0.1],#añadir porcentaje de tipos
	p.aStarUseCellExponentialWeight=0.5

	# p.densityCars=0.35 # HACK
	# p.densityEV = 0.6 # HACK
	# p.densityDiesel=0.2 # HACK
	# p.densityPetrol=1-p.densityEV-p.densityDiesel # HACK

	if timeTest is not None:

		p.aStarCSQueueQuery=0.5#poner el óptimo (ver paper)
		p.aStarCSReserve=0.5#<= que el anterior, x primeros. ¿de los que consultan, qué porcentaje reservan? comprobar
		
		p.densityCars = timeTest[0]
		p.densityEV = timeTest[1]

		p.densityDiesel = 0.5
		p.densityPetrol = 0.5

	if view:
		city=City(p,indiv)
		#print("Running experiment: "+p.legendName)
		city.plot(True)
	else:
		#print("Running experiment: "+p.legendName)
		if run_all:
			cont = ContaminationExperiment(numExperiment=i,buildings=p.buildings,distributionCS=p.distributionCS,windV=p.windV)
			indiv = cont.individual
			city=City(p,indiv)
			city.runWithoutPlot(times=cont.num_timesteps,returnFits=False,contaminationExp=True,delta=cont.delta,corner_factor=cont.corner_factor,gamma=cont.gamma,acc=cont.acc,dif_matrix=cont.dif_matrix,wind=cont.wind,names=(p.seed,p.buildings,p.distributionCS,p.densityCars,p.densityEV,p.densityDiesel,p.windV,p.pollutionRouting))
		else:
			names=(p.seed,p.buildings,p.distributionCS,p.densityCars,p.densityEV,p.densityDiesel,p.windV,p.pollutionRouting)
			city=City(p,indiv)
			if returnFits:
				if cnnExpDataSave is not None:
					if not os.path.exists(cnnExpDataSave):
							os.makedirs(cnnExpDataSave)
					dataName = f"CNNExp_num-{numSave}_seed-{i}"
					np.save(os.path.join(contaminationExp, dataName), city.grid.grid)
				global_fit, local_fit = city.runWithoutPlot(times=numTimesteps, returnFits=returnFits, delta=delta, corner_factor=corner_factor, gamma=gamma, acc=acc, dif_matrix=dif_matrix, wind=wind, names=names)
				return global_fit, local_fit
			elif contaminationExp:
				# print("\tInit the contaminationEXP")
				city.runWithoutPlot(numTimesteps, returnFits, contaminationExp, delta, corner_factor, gamma, acc, dif_matrix, wind, names=names)
			else:
				city.runWithoutPlot(numTimesteps) #HACKed antes valía 1000
		t_end = time.time()
		if timeTest is not None:
			t_final = t_end - t_start

			return t_final
		#stats=Stats(p)
		#stats.plotCS(False)
		#stats.plot(False)


SimulationResult = namedtuple('SimulationResult', ['global_fit', 'local_fit'])


def run_sim(a):
	sim_cache = a[0]
	indiv = a[1]
	numExperiment = a[2]
	aux=Genetic(numExperiment,simulation_cache=sim_cache).run_simulation(indiv)
	return aux

def initialize_individual(valid_coordinates, num_chargers, distance, lim_distance):
	some_coordinates = [random.choice(valid_coordinates)]
	M = random.randint(1,num_chargers)
	j=0
	while len(some_coordinates)<M:
		aux_coord = random.choice(valid_coordinates)
		if all([distance(aux_coord,y)>=lim_distance and distance(y,aux_coord)>lim_distance for y in some_coordinates]):
			some_coordinates.append(aux_coord)
		j+=1
		if j>10000:
			print('Error: No se pueden obtener las CS.')
			exit()
	limits = sorted(random.choices(range(0, num_chargers + 1), k=M - 1))
	limits = [0] + limits + [num_chargers]
	chargers_per_station = [limits[i] - limits[i - 1] for i in range(1, len(limits))]
	stations = [GChargingStation(some_coordinates[k], chargers_per_station[k]) for k in range(M) if chargers_per_station[k]>0]
	return Individual(stations)

def wrappedGetIndividual(args):
	def aStarDistance(cell:Cell,target:Cell, t=0):
		# Distance version
		# only mark visited if it has more than one destination
		visited=set()
		visited.add(cell)
		opened={}
		for d in cell.destination:
			if len(cell.destination)==1:
				opened[d]=[]
			else:
				opened[d]=[d]
		opened2={}
		distancia=1

		while True:
			# solo se añaden los visited con mas de uno
			for (o,r) in opened.items():
				if len(o.destination)==1:
					opened2[o.destination[0]]=r
				else:
					if o not in visited:
						visited.add(o)
						for d in o.destination:
							r2=r.copy()
							r2.append(d)
							opened2[d]=r2
				if o==target:
					return (distancia,opened[o])
			opened=opened2
			opened2={}
			distancia+=1
			
	valid_coordinates, num_chargers, lim_distance, numExperiment = args

	p=cartesianExperiment()[numExperiment]
	p.listgenerator=True
	p.numberChargingPerStation=0
	city1 = City(p)
	g = city1.generator()
	for k in g:
		print(k)
		break
	distance = lambda x,y: aStarDistance(city1.grid.grid[x[1],x[0]], city1.grid.grid[y[1],y[0]] )[0]
	return initialize_individual(valid_coordinates, num_chargers, distance, lim_distance)
# if __name__ == '__main__':
# 	# list of all experiments
# 	ps = cartesianExperiment()
# 	for (i,p) in enumerate(ps):
# 		print(i,p.legendName)

# 	# view an particular experiment
# 	experiment(406)

# 	# execute in background all experiments
# 	start_time = time.time()  
# 	num_processors = multiprocessing.cpu_count()

# 	ps2=[]
# 	for i in range(0,len(ps),50):
# 		ps2.append(ps[i:i+50])

# 	for i,ps in enumerate(ps2):
# 		with multiprocessing.Pool(num_processors) as pool:
# 			pool.map(experiment, range(len(ps)))
# 		print(f'Finished {i+1}/{len(ps2)}')
		
# 	end_time = time.time()  
# 	duration = end_time - start_time 
# 	print(f'Total time: {duration:.2f} seconds')

# 	# generate the metastats
# 	ms=MetaStats()
		
class GChargingStation:
	def __init__(self, coordinates: Tuple[int, int], num_chargers: int):
		self.coordinates: Tuple[int, int] = coordinates
		self.num_chargers: int = num_chargers

	def __repr__(self) -> str:
		return f"GChargingStation(coordinates={self.coordinates}, num_chargers={self.num_chargers})"
	
class Individual:
	def __init__(self, stations: List[GChargingStation]):
		self.stations = stations  # List of ChargingStation objects

	def __repr__(self) -> str:
		return f"Individual(stations={self.stations})"
	
	def __eq__(self, other):
		if not isinstance(other, Individual):
			return NotImplemented
		return sorted(self.stations, key=lambda x: (x.coordinates, x.num_chargers)) == sorted(other.stations, key=lambda x: (x.coordinates, x.num_chargers))

class Genetic:
	def __init__(self,numExperiment, simulation_cache={}, args_gen={}) -> None:

		############ Extract data from args_gen ############
		self.path_gen_data = args_gen.get('path_gen_data', os.path.join('data', 'genetic'))
		self.is_save_data = args_gen.get('is_save_data', False)
		####################################################

		self.numExperiment = numExperiment
		self.population_size = (multiprocessing.cpu_count()-1)*2
		#self.max_num_stations = 5
		self.num_chargers = 72
		self.num_timesteps = 2000
		#self.distance = lambda x,y : np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
		self.lim_distance = 10
		self.num_generations: int = 50
		self.mutation_rate: float = 0.9
		p=cartesianExperiment()[numExperiment]
		p.listgenerator=True
		p.numberChargingPerStation=0
		city1 = City(p)
		g = city1.generator()
		for k in g:
			print(k)
			break
		self.city1 = city1
		self.distance = lambda x,y: self.aStarDistance(city1.grid.grid[x[1],x[0]],city1.grid.grid[y[1],y[0]])[0]
		self.valid_coordinates = city1.listgenerator
		(n_rows, n_cols) = city1.sizes
		self.delta = np.float32(0.2)  # Diffusion parameter
		self.corner_factor = 1#/np.sqrt(2)
		self.gamma = np.float32(0.1) # Lost to the atmosphere
		self.buildings=True
		if self.buildings:
			acc = np.zeros((n_rows+2, n_cols+2))
			for x,y in self.valid_coordinates:
				acc[x,y] = 1

			sidewalk = True
			
			if sidewalk:
				acc[45:52,:]=1
				acc[:,45:52]=1
				acc[141:148,:]=1
				acc[:,141:148]=1
				acc[237:244,:]=1
				acc[:,237:244]=1

				l=list(range(0,10))+list(range(85,106))+list(range(181,202))+list(range(277,287))
				for i in l:
					for j in l:
						acc[i+1,j+1]=1
		

			acc[0,:]=1
			acc[-1,:]=1
			acc[:,0]=1
			acc[:,-1]=1

			#acc[:,:]=1

			#print(acc[50,146])
			acc_neig_edge = (
				acc[0:-2, 1:-1] + acc[2:, 1:-1] +
				acc[1:-1, 0:-2] + acc[1:-1, 2:]
			)
			acc_neig_corner = (
				acc[0:-2, 0:-2] + acc[2:, 2:] +
				acc[2:, 0:-2] + acc[0:-2, 2:]
			)

			if sidewalk: #Aceras
				for i in range(1,n_rows+1):
					for j in range(1,n_cols+1):
						if acc_neig_corner[i-1,j-1]+acc_neig_edge[i-1,j-1]:
							if (i in range(2,n_rows) and j in range(2,n_cols)) or acc_neig_corner[i-1,j-1]+acc_neig_edge[i-1,j-1]>3:
							#if 1<i<n_rows and 1<j<n_cols:
								acc[i,j]=1
		else:
			acc = np.ones((n_rows+2, n_cols+2))

		acc_neig_edge = (
			acc[0:-2, 1:-1] + acc[2:, 1:-1] +
			acc[1:-1, 0:-2] + acc[1:-1, 2:]
		)
		acc_neig_corner = (
			acc[0:-2, 0:-2] + acc[2:, 2:] +
			acc[2:, 0:-2] + acc[0:-2, 2:]
		)

		dif_matrix = 1 - (acc_neig_edge + acc_neig_corner * self.corner_factor) * self.delta / (4 + 4 * self.corner_factor)

		self.acc = acc[1:-1, 1:-1]
		self.dif_matrix = dif_matrix
		
		self.WN = 0# * np.ones((n_rows+2, n_cols+2, self.num_timesteps+1))
		self.WE = 0.1# * np.ones((n_rows+2, n_cols+2, self.num_timesteps+1))
		sign_WN = np.sign(self.WN).astype(int)
		sign_WE = np.sign(self.WE).astype(int)
		displ_N = np.zeros((n_rows+2, n_cols+2))#np.zeros_like(self.WN)
		displ_S = np.zeros_like(displ_N)
		displ_E = np.zeros_like(displ_N)
		displ_W = np.zeros_like(displ_N)
		displ_NW = np.zeros_like(displ_N)
		displ_NE = np.zeros_like(displ_N)
		displ_SW = np.zeros_like(displ_N)
		displ_SE = np.zeros_like(displ_N)
		stays = np.ones_like(displ_N)
		for p in range(1, n_rows+1):
			for q in range(1, n_cols+1):
					displ_N[p,q] = acc[p,q] * np.maximum(self.WN, 0) * (1 - np.maximum(acc[p + sign_WE, q - 1], acc[p + sign_WE, q]) * abs(self.WE)) * acc[p, q - 1]
					displ_S[p,q] = acc[p,q] * np.maximum(-self.WN,0) * (1 - np.maximum(acc[p + sign_WE, q + 1], acc[p + sign_WE, q]) * abs(self.WE)) * acc[p, q + 1]
					displ_E[p,q] = acc[p,q] * np.maximum(self.WE, 0) * (1 - np.maximum(acc[p + 1, q - sign_WN], acc[p, q - sign_WN]) * abs(self.WN)) * acc[p + 1, q]
					displ_W[p,q] = acc[p,q] * np.maximum(-self.WE,0) * (1 - np.maximum(acc[p - 1, q - sign_WN], acc[p, q - sign_WN]) * abs(self.WN)) * acc[p - 1, q]
					displ_NE[p,q] = acc[p,q] * np.maximum(self.WN, 0) * np.maximum(self.WE, 0) * acc[p + 1, q - 1]
					displ_NW[p,q] = acc[p,q] * np.maximum(self.WN, 0) * np.maximum(-self.WE,0) * acc[p - 1, q - 1]
					displ_SE[p,q] = acc[p,q] * np.maximum(-self.WN,0) * np.maximum(self.WE, 0) * acc[p + 1, q + 1]
					displ_SW[p,q] = acc[p,q] * np.maximum(-self.WN,0) * np.maximum(-self.WE,0) * acc[p - 1, q + 1]
		stays += -(displ_N + displ_S + displ_E + displ_W + displ_NE + displ_NW + displ_SE + displ_SW)
		self.wind = (displ_N[1:-1, 2:], displ_S[1:-1, :-2], displ_E[:-2, 1:-1], displ_W[2:, 1:-1], displ_NE[:-2, 2:], displ_NW[2:, 2:], displ_SE[:-2, :-2], displ_SW[2:, :-2], stays[1:-1, 1:-1])
		self.simulation_cache = simulation_cache  # Diccionario para almacenar los resultados de las simulaciones
		#if self.max_num_stations > self.num_chargers:
		#    print("The number of CS cannot exceed the number of chargers.")
		#    exit()
	'''
		def CS_allocators(self,set_valid_coordinates):
			some_coordinates = [random.choice(set_valid_coordinates)]
			M = random.randint(1,self.num_chargers)
			for _ in range(1,M):
				aux_coord = [x for x in set_valid_coordinates if all([self.distance(x,y)>=self.lim_distance and self.distance(y,x)>self.lim_distance for y in some_coordinates])]
				if aux_coord==[]:
						print('Error: No se pueden obtener las CS.')
						exit()
				some_coordinates.append(random.choice(aux_coord))  
			limits = sorted(random.choices(range(0, self.num_chargers + 1), k=M - 1))
			limits = [0] + limits + [self.num_chargers]
			chargers_per_station = [limits[i] - limits[i - 1] for i in range(1, len(limits))]
			return [GChargingStation(some_coordinates[k], chargers_per_station[k]) for k in range(M)]
	'''

	def aStarDistance(self, cell:Cell,target:Cell, t=0):
		# Distance version
		# only mark visited if it has more than one destination
		visited=set()
		visited.add(cell)
		opened={}
		for d in cell.destination:
			if len(cell.destination)==1:
				opened[d]=[]
			else:
				opened[d]=[d]
		opened2={}
		distancia=1

		while True:
			# solo se añaden los visited con mas de uno
			for (o,r) in opened.items():
				if len(o.destination)==1:
					opened2[o.destination[0]]=r
				else:
					if o not in visited:
						visited.add(o)
						for d in o.destination:
							r2=r.copy()
							r2.append(d)
							opened2[d]=r2
				if o==target:
					return (distancia,opened[o])
			opened=opened2
			opened2={}
			distancia+=1

	def initialize_population(self) -> List[Individual]:	
		
		print("Making the init population, size: ", self.population_size)
		processors = multiprocessing.cpu_count()-1

		valid_coordinates = self.valid_coordinates
		num_chargers = self.num_chargers
		lim_distance = self.lim_distance
		population_size = self.population_size
		numExperiment = self.numExperiment
		list_parameters = [(valid_coordinates, num_chargers, lim_distance, numExperiment)]*population_size
		with multiprocessing.Pool(processors) as pool:
			population: List[Individual] = pool.map(wrappedGetIndividual, list_parameters)
		print("Finished maked population, size: ", len(population))
		return population

	'''
		def initialize_population(self) -> List[Individual]:
			population: List[Individual] = []
			for l in range(self.population_size):
				some_coordinates = [random.choice(self.valid_coordinates)]
				M = random.randint(1,self.num_chargers)
				for _ in range(1,M):
					aux_coord = [x for x in self.valid_coordinates if all([self.distance(x,y)>=self.lim_distance and self.distance(y,x)>self.lim_distance for y in some_coordinates])]
					if aux_coord==[]:
						print('Error: No se pueden obtener las CS.')
						exit()
					some_coordinates.append(random.choice(aux_coord))  
				limits = sorted(random.choices(range(0, self.num_chargers + 1), k=M - 1))
				limits = [0] + limits + [self.num_chargers]
				chargers_per_station = [limits[i] - limits[i - 1] for i in range(1, len(limits))]
				stations = [GChargingStation(some_coordinates[k], chargers_per_station[k]) for k in range(M) if chargers_per_station[k]>0]
				population.append(Individual(stations))
				print('Initialized ', l+1)
			return population
	'''
	def calculate_fitness(self, resul_fit) -> float:
		# Maybe study how to implement the Shannon entropy (future work)
		#simulation_result = self.run_simulation(individual)  # This should return a SimulationResult namedtuple
		result = resul_fit.global_fit * (sum(resul_fit.local_fit)/len(resul_fit.local_fit)) # Or some other function of these. Note that we want to maximize the productivity and minimize the energy (multiobjective optimization)
		return result

	def run_simulation(self, individual: Individual) -> namedtuple:
		# Unique key for this individual
		key = tuple(sorted((station.coordinates, station.num_chargers) for station in individual.stations))
		# verify whether the result is already in the cache
		if key in self.simulation_cache:
			return (key,self.simulation_cache[key])
		#SimulationResult = namedtuple('SimulationResult', ['global_fit', 'local_fit'])
		# Here you run your simulation
		exp_result = experiment(self.numExperiment,run_all=False,view=False,cache=False, indiv = individual, \
			returnFits = True, numTimesteps = self.num_timesteps, delta = self.delta, corner_factor = self.corner_factor, \
			gamma = self.gamma, acc = self.acc, dif_matrix = self.dif_matrix, wind = self.wind)
		result = SimulationResult(global_fit = exp_result[0], local_fit = exp_result[1]) #TERMINAR DE ARREGLAR
		#indiv = individual.stations
		#M = len(indiv)
		#coordinates = [indiv[k].coordinates for k in range(M)]
		#number_chargers = [indiv[k].num_chargers for k in range(M)]
		#result = SimulationResult(global_fit=abs(np.mean(number_chargers)-5) + np.std(number_chargers), local_fit=[self.distance(A,(41,11))+self.distance(A,(41,83)) for A in coordinates])
		# Save result in cache
		self.simulation_cache[key] = result
		return (key,result)

	def select_parents(self) -> List[Individual]:
		#fitness_scores = [self.calculate_fitness(individual) for individual in self.population]
		selected = sorted([(a,self.calculate_fitness(b)) for (a,b) in zip(self.population, self.fitness_values)], key=lambda x: x[1])[:self.population_size // 2]
		#selected = sorted(zip(self.population, self.fitness_values), key=lambda x: x[1])[:self.population_size // 2]
		return [x[0] for x in selected]

	def crossover(self, parent1, parent2):
		parent_1 = copy.deepcopy(parent1)
		parent_2 = copy.deepcopy(parent2)
		results = zip(parent_1.stations + parent_2.stations, self.run_simulation(parent_1)[1].local_fit + self.run_simulation(parent_2)[1].local_fit)
		results_ordered = [elemento for elemento, _ in sorted(results, key=lambda k: k[1])]
		child_stations = []
		for _ in range(len(results_ordered)):
			auxnum = random.choices(range(len(results_ordered)), weights=[1 / (k + 1) for k in range(len(results_ordered))], k=1)[0]
			aux = results_ordered[auxnum]
			results_ordered.pop(auxnum)
			if all([self.distance(aux.coordinates, CS.coordinates) >= self.lim_distance or self.distance(CS.coordinates, aux.coordinates) >= self.lim_distance for CS in child_stations]) and sum([CS.num_chargers for CS in child_stations]) < self.num_chargers:
					child_stations.append(aux)
		if child_stations == parent_1.stations or child_stations == parent_2.stations:
			if len(child_stations) != 1:
					child_stations.pop()
		return Individual(child_stations)
	
	def mutate(self, individual):
		indiv = copy.deepcopy(individual.stations)
		M = len(indiv)
		if random.random() < self.mutation_rate:
			mutation_index = random.randint(0, M-1)
			new_num_chargers = random.randint(1, self.num_chargers)
			indiv[mutation_index].num_chargers = new_num_chargers
			coords = [indiv[k].coordinates for k in range(M) if k != mutation_index]
			aux_coord = random.choice(self.valid_coordinates)
			j=0
			while any([self.distance(aux_coord,y)<self.lim_distance or self.distance(y,aux_coord)<self.lim_distance for y in coords]):
				if j>10000:
					print('Error: No se pueden obtener las CS.')
					exit()
				aux_coord = random.choice(self.valid_coordinates)
				j+=1
			indiv[mutation_index].coordinates = aux_coord
		return Individual(indiv)

	def renormalization(self, individual):
		indiv = copy.deepcopy(individual.stations)
		M = len(indiv)
		chargers_per_station = [CS.num_chargers for CS in indiv]
		CS_chosen = [0] * M
		for _ in range(self.num_chargers):
			max_index = max(range(M), key=lambda i: chargers_per_station[i] / (CS_chosen[i] + 1))
			CS_chosen[max_index] += 1
		indiv = [GChargingStation(copy.deepcopy(indiv[k].coordinates),CS_chosen[k]) for k in range(M) if CS_chosen[k]!=0]
		return Individual(indiv)

	def find_best_solution(self) -> Individual:
		#fitness_scores = zip(self.population,self.fitness_values)#[(individual, self.calculate_fitness(individual)) for individual in self.population]
		min_individual, min_fitness, both = sorted([(a,self.calculate_fitness(b), b) for (a,b) in zip(self.population, self.fitness_values)], key=lambda x: x[1])[:5]
		return min_individual, min_fitness, both
	


	def parallel_run_simulations(self, individuals):
		
		# Create a pool of workers
		with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
			# Map the individuals to the worker function
			aux = [(self.simulation_cache,indiv, self.numExperiment) for indiv in individuals]
			results = pool.map(run_sim, aux)
			res_list = []
			for x in results:
				self.simulation_cache[x[0]]=x[1]
				res_list.append(x[1])
		# Return the results
		return res_list

	
	# For example, in calculating fitness for the entire population
	def calculate_population_fitness(self):
		# Use the parallel version instead of individual run_simulation calls
		results = self.parallel_run_simulations(self.population)
		return results

	def make_data(self, ):
		
		pass

	def run(self) -> None:

		def show_top_5(best_five):
			for i, (best_solution, best_value, both) in enumerate(self.find_best_solution()):
				print("Best Solution:", best_solution, "\nBest value:", best_value, '\n Global and Local Fitness:', both)

		self.population: List[Individual] = self.initialize_population()
		self.fitness_values: List[float] = self.calculate_population_fitness()
		best_solution, best_value, both = self.find_best_solution()[0]
		print("Best Solution:", best_solution, "\nBest value:", best_value, '\n Global and Local Fitness:', both)
		values=[0]*(self.num_generations+1)
		values[0]=best_value
		for generation in range(self.num_generations):
			init_time = time.time()
			print('Generation', generation+1)
			parents = self.select_parents()
			new_population = copy.deepcopy(parents)
			while len(new_population) < self.population_size:
					parent1, parent2 = random.sample(parents, 2)
					newindiv = self.crossover(parent1,parent2)
					newindiv = self.mutate(newindiv)
					newindiv = self.renormalization(newindiv)
					if newindiv not in new_population:
						new_population.append(newindiv)
						# print('Generation ', generation+1, ' Individuals ', len(new_population))
			if len(new_population) < self.population_size:
					print(f"Loop terminated at generation: {generation}")
					break
			self.population = new_population
			self.fitness_values = self.calculate_population_fitness()
			best_solution, best_value, both = self.find_best_solution()[0]
			#print(parents[0]==best_solution)
			#print(best_value, '\n', self.calculate_fitness(parents[0]), '\n\n\n')
			#print(sum([CS.num_chargers for CS in parents[0].stations]))
			print("Best Solution:", best_solution, "\nBest value:", best_value, '\n Global and Local Fitness:', both)
			values[generation+1]=best_value
			self.population = new_population
			finish_time = time.time()
			aux = [tuple([(CS.coordinates,CS.num_chargers) for CS in indiv.stations]) for indiv in new_population]
			print(len(aux)-len(set(aux)))
			print('Running time: ', finish_time - init_time, 's')
		best_solution, best_value, both = self.find_best_solution()[0]
		print("Best Solution:", best_solution, "\nBest value:", best_value, '\n Global and Local Fitness:', both)
		print('Total different individials: ', len(self.simulation_cache))
		print('Should be: ', (self.num_generations+1)*self.population_size - self.num_generations * (self.population_size // 2))
		plt.plot(range(self.num_generations+1),values)
		plt.show()

class ContaminationExperiment:
	def __init__(self,numExperiment=43, buildings = True, distributionCS=0, windV=(0.1,0)) -> None:
		#numExperiment=0
		self.numExperiment = numExperiment
		self.buildings = buildings
		self.distributionCS=distributionCS#should be in 0,1,2.
		self.windV = windV
		#self.population_size = multiprocessing.cpu_count()
		#self.max_num_stations = 5
		self.num_chargers = 72
		self.num_timesteps = 2000
		#numExperiment=0
		p=cartesianExperiment()[numExperiment]
		p.listgenerator=True
		p.numberChargingPerStation=0
		city1 = City(p)
		g = city1.generator()
		for k in g:
			#
			pass
		#print('here3')
		#self.distance = lambda x,y: self.aStarDistance(city1.grid.grid[x[1],x[0]],city1.grid.grid[y[1],y[0]])[0]
		self.valid_coordinates = city1.listgenerator
		#print('DONEEE')

		#print((140,144) in self.valid_coordinates)

		#print((51,65) in self.valid_coordinates)
		#print((51,224) in self.valid_coordinates)
		#print((220,51) in self.valid_coordinates)
		#print((220,237) in self.valid_coordinates)
		'''
			print((31,32) in self.valid_coordinates)
			print((31,64) in self.valid_coordinates)
			print((31,128) in self.valid_coordinates)
			print((31,160) in self.valid_coordinates)
			print((31,224) in self.valid_coordinates)
			print((31,256) in self.valid_coordinates)
			print((64,31) in self.valid_coordinates)
			print((64,64) in self.valid_coordinates)
			print((64,127) in self.valid_coordinates)
			print((64,160) in self.valid_coordinates)
			print((64,223) in self.valid_coordinates)
			print((64,256) in self.valid_coordinates)
			print((127,32) in self.valid_coordinates)
			print((127,64) in self.valid_coordinates)
			print((127,128) in self.valid_coordinates)
			print((127,160) in self.valid_coordinates)
			print((127,224) in self.valid_coordinates)
			print((127,256) in self.valid_coordinates)
			print((160,31) in self.valid_coordinates)
			print((160,64) in self.valid_coordinates)
			print((160,127) in self.valid_coordinates)
			print((160,160) in self.valid_coordinates)
			print((160,223) in self.valid_coordinates)
			print((160,256) in self.valid_coordinates)
			print((223,32) in self.valid_coordinates)
			print((223,64) in self.valid_coordinates)
			print((223,128) in self.valid_coordinates)
			print((223,160) in self.valid_coordinates)
			print((223,224) in self.valid_coordinates)
			print((223,256) in self.valid_coordinates)
			print((256,31) in self.valid_coordinates)
			print((256,64) in self.valid_coordinates)
			print((256,127) in self.valid_coordinates)
			print((256,160) in self.valid_coordinates)
			print((256,223) in self.valid_coordinates)
			print((256,256) in self.valid_coordinates)
		'''

		(n_rows, n_cols) = city1.sizes
		'''
			# Separate the list of tuples into two lists: x and y coordinates
			x_coords, y_coords = zip(*self.valid_coordinates)

			# Create a scatter plot
			plt.figure(figsize=(10, 6))
			plt.scatter(x_coords, y_coords, c='blue', marker='o')
			plt.title('Valid Coordinates')
			plt.xlabel('X Coordinate')
			plt.ylabel('Y Coordinate')
			plt.grid(True)
			plt.show()
		'''



		self.delta = np.float32(0.1)  # Diffusion parameter
		self.corner_factor = 1#/np.sqrt(2)
		self.gamma = np.float32(0.01) # Lost to the atmosphere

		if self.buildings:
			acc = np.zeros((n_rows+2, n_cols+2))
			for x,y in self.valid_coordinates:
				acc[x+1,y+1] = 1

			sidewalk = True
			
			if sidewalk:
				acc[45:52,:]=1
				acc[:,45:52]=1
				acc[141:148,:]=1
				acc[:,141:148]=1
				acc[237:244,:]=1
				acc[:,237:244]=1

				l=list(range(0,10))+list(range(85,106))+list(range(181,202))+list(range(277,287))
				for i in l:
					for j in l:
						acc[i+1,j+1]=1
		

			acc[0,:]=1
			acc[-1,:]=1
			acc[:,0]=1
			acc[:,-1]=1

			#acc[:,:]=1

			#print(acc[50,146])
			acc_neig_edge = (
				acc[0:-2, 1:-1] + acc[2:, 1:-1] +
				acc[1:-1, 0:-2] + acc[1:-1, 2:]
			)
			acc_neig_corner = (
				acc[0:-2, 0:-2] + acc[2:, 2:] +
				acc[2:, 0:-2] + acc[0:-2, 2:]
			)

			if sidewalk: #Aceras
				for i in range(1,n_rows+1):
					for j in range(1,n_cols+1):
						if acc_neig_corner[i-1,j-1]+acc_neig_edge[i-1,j-1]:
							if (i in range(2,n_rows) and j in range(2,n_cols)) or acc_neig_corner[i-1,j-1]+acc_neig_edge[i-1,j-1]>3:
							#if 1<i<n_rows and 1<j<n_cols:
								acc[i,j]=1
		else:
			acc = np.ones((n_rows+2, n_cols+2))

		acc_neig_edge = (
			acc[0:-2, 1:-1] + acc[2:, 1:-1] +
			acc[1:-1, 0:-2] + acc[1:-1, 2:]
		)
		acc_neig_corner = (
			acc[0:-2, 0:-2] + acc[2:, 2:] +
			acc[2:, 0:-2] + acc[0:-2, 2:]
		)

		dif_matrix = (1 - (acc_neig_edge + acc_neig_corner * self.corner_factor) * self.delta / (4 + 4 * self.corner_factor))
		
		

		self.acc = acc[1:-1, 1:-1]
		self.dif_matrix = dif_matrix
		self.WN = self.windV[1]# * np.ones((n_rows+2, n_cols+2, self.num_timesteps+1))
		self.WE = self.windV[0]# * np.ones((n_rows+2, n_cols+2, self.num_timesteps+1))
		sign_WN = np.sign(self.WN).astype(int)
		sign_WE = np.sign(self.WE).astype(int)
		displ_N = np.zeros((n_rows+2, n_cols+2))#np.zeros_like(self.WN)
		displ_S = np.zeros_like(displ_N)
		displ_E = np.zeros_like(displ_N)
		displ_W = np.zeros_like(displ_N)
		displ_NW = np.zeros_like(displ_N)
		displ_NE = np.zeros_like(displ_N)
		displ_SW = np.zeros_like(displ_N)
		displ_SE = np.zeros_like(displ_N)
		stays = np.ones_like(displ_N)
		for p in range(1, n_rows+1):
			for q in range(1, n_cols+1):
					displ_N[p,q] = acc[p,q] * np.maximum(self.WN, 0) * (1 - np.maximum(acc[p + sign_WE, q - 1], acc[p + sign_WE, q]) * abs(self.WE)) * acc[p, q - 1]
					displ_S[p,q] = acc[p,q] * np.maximum(-self.WN,0) * (1 - np.maximum(acc[p + sign_WE, q + 1], acc[p + sign_WE, q]) * abs(self.WE)) * acc[p, q + 1]
					displ_E[p,q] = acc[p,q] * np.maximum(self.WE, 0) * (1 - np.maximum(acc[p + 1, q - sign_WN], acc[p, q - sign_WN]) * abs(self.WN)) * acc[p + 1, q]
					displ_W[p,q] = acc[p,q] * np.maximum(-self.WE,0) * (1 - np.maximum(acc[p - 1, q - sign_WN], acc[p, q - sign_WN]) * abs(self.WN)) * acc[p - 1, q]
					displ_NE[p,q] = acc[p,q] * np.maximum(self.WN, 0) * np.maximum(self.WE, 0) * acc[p + 1, q - 1]
					displ_NW[p,q] = acc[p,q] * np.maximum(self.WN, 0) * np.maximum(-self.WE,0) * acc[p - 1, q - 1]
					displ_SE[p,q] = acc[p,q] * np.maximum(-self.WN,0) * np.maximum(self.WE, 0) * acc[p + 1, q + 1]
					displ_SW[p,q] = acc[p,q] * np.maximum(-self.WN,0) * np.maximum(-self.WE,0) * acc[p - 1, q + 1]
		stays += -(displ_N + displ_S + displ_E + displ_W + displ_NE + displ_NW + displ_SE + displ_SW)
		self.wind = (displ_N[1:-1, 2:], displ_S[1:-1, :-2], displ_E[:-2, 1:-1], displ_W[2:, 1:-1], displ_NE[:-2, 2:], displ_NW[2:, 2:], displ_SE[:-2, :-2], displ_SW[2:, :-2], stays[1:-1, 1:-1])

		individual1 = Individual([GChargingStation((137,142),self.num_chargers)])#Individual([GChargingStation((140,144),self.num_chargers)])
		#individual2 = Individual([GChargingStation((51,65),self.num_chargers//4), GChargingStation((51,224),self.num_chargers//4), GChargingStation((220,51),self.num_chargers//4), GChargingStation((220,237),self.num_chargers//4)])
		individual2 = Individual([GChargingStation((213,45),self.num_chargers//4), GChargingStation((45,74),self.num_chargers//4), GChargingStation((74,243),self.num_chargers//4), GChargingStation((243,213),self.num_chargers//4)])
		#individual3 = Individual([GChargingStation((31,32),self.num_chargers//36), GChargingStation((31,64),self.num_chargers//36), GChargingStation((31,128),self.num_chargers//36), GChargingStation((31,160),self.num_chargers//36), GChargingStation((31,224),self.num_chargers//36), GChargingStation((31,256),self.num_chargers//36),
		#				   GChargingStation((64,31),self.num_chargers//36), GChargingStation((64,64),self.num_chargers//36), GChargingStation((64,127),self.num_chargers//36), GChargingStation((64,160),self.num_chargers//36), GChargingStation((64,223),self.num_chargers//36), GChargingStation((64,256),self.num_chargers//36),
		#				   GChargingStation((127,32),self.num_chargers//36), GChargingStation((127,64),self.num_chargers//36), GChargingStation((127,128),self.num_chargers//36), GChargingStation((127,160),self.num_chargers//36), GChargingStation((127,224),self.num_chargers//36), GChargingStation((127,256),self.num_chargers//36),
		#				   GChargingStation((160,31),self.num_chargers//36), GChargingStation((160,64),self.num_chargers//36), GChargingStation((160,127),self.num_chargers//36), GChargingStation((160,160),self.num_chargers//36), GChargingStation((160,223),self.num_chargers//36), GChargingStation((160,256),self.num_chargers//36),
		#				   GChargingStation((223,32),self.num_chargers//36), GChargingStation((223,64),self.num_chargers//36), GChargingStation((223,128),self.num_chargers//36), GChargingStation((223,160),self.num_chargers//36), GChargingStation((223,224),self.num_chargers//36), GChargingStation((223,256),self.num_chargers//36),
		#				   GChargingStation((256,31),self.num_chargers//36), GChargingStation((256,64),self.num_chargers//36), GChargingStation((256,127),self.num_chargers//36), GChargingStation((256,160),self.num_chargers//36), GChargingStation((256,223),self.num_chargers//36), GChargingStation((256,256),self.num_chargers//36)])
		individual3 = Individual([GChargingStation((45,21),self.num_chargers//36), GChargingStation((45,117),self.num_chargers//36), GChargingStation((45,213),self.num_chargers//36), GChargingStation((51,74),self.num_chargers//36), GChargingStation((51,170),self.num_chargers//36), GChargingStation((51,266),self.num_chargers//36),
						GChargingStation((141,21),self.num_chargers//36), GChargingStation((141,117),self.num_chargers//36), GChargingStation((141,213),self.num_chargers//36), GChargingStation((147,74),self.num_chargers//36), GChargingStation((147,170),self.num_chargers//36), GChargingStation((147,266),self.num_chargers//36),
						GChargingStation((237,21),self.num_chargers//36), GChargingStation((237,117),self.num_chargers//36), GChargingStation((237,213),self.num_chargers//36), GChargingStation((243,74),self.num_chargers//36), GChargingStation((243,170),self.num_chargers//36), GChargingStation((243,266),self.num_chargers//36),
						GChargingStation((21,51),self.num_chargers//36), GChargingStation((74,45),self.num_chargers//36), GChargingStation((117,51),self.num_chargers//36), GChargingStation((170,45),self.num_chargers//36), GChargingStation((213,51),self.num_chargers//36), GChargingStation((266,45),self.num_chargers//36),
						GChargingStation((21,147),self.num_chargers//36), GChargingStation((74,141),self.num_chargers//36), GChargingStation((117,147),self.num_chargers//36), GChargingStation((170,141),self.num_chargers//36), GChargingStation((213,147),self.num_chargers//36), GChargingStation((266,141),self.num_chargers//36),
						GChargingStation((21,243),self.num_chargers//36), GChargingStation((74,237),self.num_chargers//36), GChargingStation((117,243),self.num_chargers//36), GChargingStation((170,237),self.num_chargers//36), GChargingStation((213,243),self.num_chargers//36), GChargingStation((266,237),self.num_chargers//36)])
		
		self.individuals = [individual1,individual2,individual3]
		self.individual = self.individuals[self.distributionCS]

	def run(self):
		#[self.distributionCS]
		# returnFits must be false so that contaminationExp is used.
		experiment(self.numExperiment,run_all=False,view=False,cache=False, indiv = self.individual, returnFits = False, contaminationExp = True, numTimesteps = self.num_timesteps, delta = self.delta, corner_factor = self.corner_factor, gamma = self.gamma, acc = self.acc, dif_matrix = self.dif_matrix, wind = self.wind)

	def run2(self, timeTest):
		return experiment(self.numExperiment,run_all=False,view=False,cache=False, indiv = self.individual, returnFits = False, contaminationExp = True, numTimesteps = self.num_timesteps, delta = self.delta, corner_factor = self.corner_factor, gamma = self.gamma, acc = self.acc, dif_matrix = self.dif_matrix, wind = self.wind, timeTest = timeTest)

def wrappedTimeTest(args):
	dCars, dEv, g = args
	return g.run2([dCars, dEv])

def fTimeTest(g, processors = 1):
	lProcessors = [None, 1]#[255, 127, 1]
	rho = [0.05, 0.15, 0.25, 0.35, 0.45]
	rhoEV = [0.05, 0.35, 0.65, 0.95]

	combinaciones = list(itertools.product(rho, rhoEV, [g])) # G is the class contaminationExperiment
	results = []
	
	for processors in lProcessors:
		print(f"{'#'*15} Iniciando test con {processors} procesos {'#'*15}\n")
		for combinacion in combinaciones:
			t = time.time()
			print((combinacion[0], combinacion[1]), "len task: ", processors)

			if processors == 1:
				res = [wrappedTimeTest(combinacion)]

			elif processors is None:
				args = [combinacion] * 256
				with multiprocessing.Pool() as pool:
					res = pool.map(wrappedTimeTest, args)
			else:
				args = [combinacion] * processors
				with multiprocessing.Pool(processors) as pool:
					res = pool.map(wrappedTimeTest, args)
			results.append((combinacion, res))
			print('\t----> time: ', time.time()-t, f"\tmean: {np.mean(res)}\n\t\tmeadian: {np.median(res)}\n\t\tstd: {np.std(res)}\n\t\tmax/min: {max(res), min(res)}\n\t\t{res}")

		print("Los tiempos de ejecucion de cada experimento han sido:")
		for combinacion, resTimes in results:
			print(f"\t{combinacion[0], combinacion[1]}:\n\t\tmean: {np.mean(resTimes)}\n\t\tmeadian: {np.median(resTimes)}\n\t\tstd: {np.std(resTimes)}\n\t\tmax/min: {max(resTimes), min(resTimes)}\n\t\t{resTimes}\n")
		print()

def wrappedGetDataCNN(idxIndv):
        numExperiment, idxI, individual = idxIndv
        cnnExpDataSave = "dataCNN/"#poner fecha
        return experiment(i=numExperiment, indiv=individual, cnnExpDataSave=cnnExpDataSave, returnFits=True, run_all=False, view=False) # Devuelve fit_globa, fit_local, name.
        
def getData2CNNExp(args, numExp = 10, numProcessors = 0) -> None:
        """Función para obtener los datos de entranamiento de la CNN.
        Se realizará de forma paralela.
        Args:
                args (dict): argumentos de entrada del archivo
                numExp (int): número de datos que se van a obtener.
        """
        def aStarDistance(cell:Cell,target:Cell):
                # Distance version
                # only mark visited if it has more than one destination
                visited=set()
                visited.add(cell)
                opened={}
                for d in cell.destination:
                        if len(cell.destination)==1:
                                opened[d]=[]
                        else:
                                opened[d]=[d]
                opened2={}
                distancia=1
                while True:
                        # solo se añaden los visited con mas de uno
                        for (o,r) in opened.items():
                                if len(o.destination)==1:
                                        opened2[o.destination[0]]=r
                                else:
                                        if o not in visited:
                                                visited.add(o)
                                                for d in o.destination:
                                                        r2=r.copy()
                                                        r2.append(d)
                                                        opened2[d]=r2
                                if o==target:
                                        return (distancia,opened[o])
                        opened=opened2
                        opened2={}
                        distancia+=1
        def getIndividuals(numExperiment):
                num_chargers = 72
                lim_distance = 10
                p=cartesianExperiment()[numExperiment]
                p.listgenerator=True
                p.numberChargingPerStation=0
                city1 = City(p)
                g = city1.generator()
                for k in g:
                        print(k)
                        break
                distance = lambda x,y: aStarDistance(city1.grid.grid[x[1],x[0]],city1.grid.grid[y[1],y[0]])[0]
                valid_coordinates = city1.listgenerator
                return initialize_individual(valid_coordinates=valid_coordinates, num_chargers=num_chargers, distance=distance, lim_distance=lim_distance)
        
        numExperiment = args.run
        processors = multiprocessing.cpu_count()-1 if numProcessors == 0 else numProcessors
        individuals = [(numExperiment, i, getIndividuals(numExperiment)) for i in range(numExp)]
        for ind in individuals:
                wrappedGetDataCNN(ind)
        # with multiprocessing.Pool(processors) as pool:
        #         res = pool.map(wrappedGetDataCNN, individuals) # fitG, fitL, name
        
        import pandas as pd
        res2 = [[name, globalF+(sum(localF)/len(localF))] for globalF, localF, name in res]
        return pd.DataFrame(res2, columns=['names', 'labels'])


# Assuming the cartesianExperiment, experiment, and MetaStats functions are defined elsewhere

if __name__ == '__main__':
	# Set default values to None
	default_values = {'list': None, 'view': None, 'run': None, 'all': False, 'stats': False,'contamination':False, 'genetic':False, 'cnnData':False, 'timeout':0, 'processes':0, 'timeTest':False}#timeout de 7200
	parser = argparse.ArgumentParser(description='Selectively run experiments.')
	parser.add_argument('--list', action='store_true', help='List all experiments', default=default_values['list'])
	parser.add_argument('--view', type=int, help='View a specific experiment by index', default=default_values['view'])
	parser.add_argument('--run', type=int, help='Run a specific experiment by index', default=default_values['run'])
	parser.add_argument('--all', action='store_true', help='Run all experiments in the background', default=default_values['all'])
	parser.add_argument('--stats', action='store_true', help='Generate meta statistics', default=default_values['stats'])
	parser.add_argument('--genetic', action='store_true', help='Enable genetic algorithm option', default=default_values['genetic'])
	parser.add_argument('--cnnData', action='store_true', help='Enable genetic algorithm option', default=default_values['cnnData'])
	parser.add_argument('--contamination', action='store_true', help='Perform contamination experiments', default=default_values['contamination'])
	parser.add_argument('--timeout', type=int, help='Timeout to execute a pool of process', default=default_values['timeout'])
	parser.add_argument('--processes', type=int, help='CPUs to use', default=default_values['processes'])
	parser.add_argument('--batch-size', type=int, help='batch size multiplier, default 1 which is equal to the number of processes', default=1)
	parser.add_argument('--data-save-dir', type=str, help='Dir where save the output data', default='simulationData32Last')
	parser.add_argument('--newData', action='store_true', help='Repeat the experiments with a clear run', default=False)
	parser.add_argument('--timeTest', action='store_true', help='Perform contamination experiments', default=default_values['timeTest'])
	#parser.add_argument()

	global dataSaveDir
	# dataSaveDir = args.data_save_dir
	dataSaveDir = "/home/josmorfig1/sanevec/source/pysimtravel3_pollution/data/testTime" # HACK
	# 44

	args = parser.parse_args()

	if args.genetic and args.contamination:
		print('Cannot use genetic and contamination experiment modules simultaneously')
		exit()

	if args.genetic or args.contamination:
		if args.run is None:
			print("If the genetic or the contamination experiment modules are used, use the --run option to specify the experiment.\n")
			exit()
		

	if not any(vars(args).values()):
		parser.print_help()
	else:
		ps = cartesianExperiment()

		if args.list:
			for (i, p) in enumerate(ps):
				print(i, p.legendName)

		if args.view is not None:
			experiment(args.view,True)

		if args.run is not None:
			if args.genetic:
				print("Iniciando el algoritmo genetico a fecha y hora: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
				g=Genetic(args.run)
				g.run()

			elif args.contamination:
				print("Init contamination experiment:")
				g = ContaminationExperiment(numExperiment=args.run)
				g.run()

			elif args.timeTest:
				print("Init the testTime experiment.")

				g = ContaminationExperiment(numExperiment=args.run, windV=(0.5, 0.3), distributionCS=1)
				processors = multiprocessing.cpu_count() - 1 if args.processes == 0 else args.processes

				print(f"###### Se van a utilizar batches de {processors} con la misma configuracion ######")
				fTimeTest(g, processors)

			elif args.cnnData:
				getData2CNNExp(args=args)
			else:
				experiment(args.run,True)#True)



		if args.all:
			import sys
			print("Python version: ", sys.version)
			start_time = time.time()
			numberExperiments = len(ps)
			#print('Hay ', numberExperiments, ' experimentos.')
			#num_processors = multiprocessing.cpu_count()-1
			if args.processes == 0:
				num_processors = multiprocessing.cpu_count()-1
			else:
				num_processors = args.processes

			print(f"Inicio de experimento el día y horas: {datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y %H:%M:%S')}")
			print('Se van a usar un total de ', num_processors, 'cores.')
			idx = 0
			if not os.path.exists(dataSaveDir):
				os.mkdir(dataSaveDir)
				print(f"Se ha creado el directorio {dataSaveDir}")
			
			while os.path.exists(dataSaveDir) and args.newData:
				_dataSaveDir = dataSaveDir+str(idx)
				if not os.path.exists(_dataSaveDir):
					dataSaveDir = _dataSaveDir
					os.mkdir(dataSaveDir)
					print(f"Se ha generado la carpeta {dataSaveDir}.")

					break
				idx += 1

			print(f"Se van a almacenar los datos en {dataSaveDir}")

			timeout = args.timeout #* args.batch_size habría que hacer esto? el batch es el doble de grande es ciert, y la pool tendrá que hacer más trabajo con el mismo tiempo, pero como irá en bloques no debería de afectarle 

			path_experiments = lambda pm: os.path.join(dataSaveDir, f'P_delta_{0.1}_gamma_{0.01}_times_{2000}_seed_{pm.seed}_buildings_{pm.buildings}_distributionCS_{pm.distributionCS}_densityCars_{pm.densityCars}_densityEV_{pm.densityEV}_densityDiesel_{pm.densityDiesel}_windV_{pm.windV}_pollutionRouting_{pm.pollutionRouting}.npz')
			 #os.path.join("simulationData32", f'P_delta_{0.1}_gamma_{0.01}_times_{2000}_seed_{p.seed}_buildings_{p.buildings}_distributionCS_{p.distributionCS}_densityCars_{p.densityCars}_densityEV_{p.densityEV}_densityDiesel_{p.densityDiesel}_windV_{p.windV}_pollutionRouting_{p.pollutionRouting}.npz')
			#bool_experiments = lambda pm: not pm.pollutionRouting# pm.densityCars <= 0.25 and (not pm.buildings and)
			#bool_experiments = lambda pm: pm.seed==34 and pm.buildings and pm.distributionCS==1 and pm.densityCars==0.05 and pm.densityEV == 0.05 and pm.densityDiesel==0.25 and pm.windV == (np.float32(0.1), 0) and not pm.pollutionRouting
			bool_experiments = lambda pm: not pm.pollutionRouting
			def transform_time(t):
				return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
			timeout = args.timeout
			print('Hay ', num_processors, 'cores.')
			'''
			ps2 = []
			for i in range(0, len(ps), 50):
				ps2.append(ps[i:i+50])

			for i, ps in enumerate(ps2):
				with multiprocessing.Pool(num_processors) as pool:
					pool.map(experiment, range(len(ps)))
				print(f'Finished {i+1}/{len(ps2)}')
			'''

			experiments_to_run = []
			print(numberExperiments)
			pattern = re.compile(r'np\.float32\((.*?)\)')
			for i in range(numberExperiments):
				p = ps[i]
				file_name = path_experiments(p)
				if bool_experiments(p) and not os.path.exists(file_name):
					if 'np.float32' in file_name:
						new_filename = pattern.sub(r'\1', file_name)
						if not os.path.exists(new_filename):
							experiments_to_run.append(i)
						else:
							print("Experimento ", p, "ya hecho.")

				else:
					print("Experimento ", p, "ya hecho.")
			print('Number of experiments: ', len(experiments_to_run))
			#experiment(experiments_to_run[0])
			#exit()
			ps2 = []
			assert args.batch_size > 0, "Error, el argumento de entrada batch-size tiene que ser mayor a 1"
			batchSize = num_processors * args.batch_size#//2
			for i in range(0, len(experiments_to_run), batchSize):
				ps2.append(experiments_to_run[i:i+batchSize])


			# Función que desenvuelve los argumentos pasados como entrada para que pueda leerlos la función
			def worker_wrapper(args):
				# finished = False
				try:
					# Desempaquetamos los argumentos y pasamos únicamente el idExperimento
					i, queue = args
					# finished = experiment(i)
					experiment(i)
					# assert finished, "Error: no se ha acabado el experimento por algún fallo interno."
					# Solo se añadiran aquellos idExperimentos que hayan acabado la ejecución
					queue.put(i)	# Como es un envoltorio si da error sigue saliendo, tendré que hacer un try: except o algo así.
				except Exception as e:
					print("\tError in: ", i, ";\n\t\t-", e, "\n\t\tName exp: ", path_experiments(cartesianExperiment()[i]))

			print("Numero de batchs totales: ", len(ps2))

			if timeout == 0:
				print("No se va a hacer uso de timeout")
			else:
				print("Se va a utilizar un timeout de: ", timeout, 'segundos.')
			experiments_finished_set = set()
			for i, _ in enumerate(ps2):
				print(f"{transform_time(time.time())} - Iniciando el batch {i}/{len(ps2)}:")
				with multiprocessing.Manager() as manager:
					experiments_finished_queue = manager.Queue()

					start_time2 = time.time()
					with multiprocessing.Pool(num_processors) as pool:
						# creamos los batchs pero añadiendo la Queue que controlará aquellos experimentos que finalicen
						batch_with_queue = [(id_experiment, experiments_finished_queue) for id_experiment in ps2[i]]
						# Ejecutamos la pool de forma asíncrona, accediendo primero a la función intermedia para desenvolver y controlar los resultados
						async_results = pool.map_async(worker_wrapper, batch_with_queue)	
						if timeout == 0:
							async_results.get()
							print("Pool finalizada")
							pool.close()
							pool.join()
						else:
							async_results.wait(timeout=timeout) # Lanzamos la función con timeout
							if async_results.ready():
								print("\tTodos los experimentos se han finalizado en tiempo, todos de forma correcta: ", async_results.successful())
								pool.close()
								pool.join()
							else:
								pool.terminate()
								pool.join()
								print("\tNo se han finalizado todos los experimentos en tiempo.")

							_experiments_finished_set = set()
							while not experiments_finished_queue.empty():
								# Añadimos los experimentos finalizados a un conjunto
								experiment_finished = experiments_finished_queue.get()
								_experiments_finished_set.add(experiment_finished)
								experiments_finished_set.add(experiment_finished)

							_experiments_not_finished = [id_experiment for id_experiment in ps2[i] if id_experiment not in _experiments_finished_set]
							print(f"\tEl número de experimentos no finalizados son: {len(_experiments_not_finished)}/{batchSize};\n\t\t y son: {_experiments_not_finished}")


						print(f'\t- Finished {i+1}/{len(ps2)} -')
						print("\t- ",(time.time()-start_time2)/60, 'minutes -')

					# Forzamos la liberación de memoria
					del pool
					gc.collect()  # Forzamos la recolección de basura
					print("\t--Queue finalizada y cerrada--")

			experiments_not_finished = [id_experiment for id_experiment in ps2[i] if id_experiment not in experiments_finished_set]
			print(f"Los siguientes experimentos ({len(experiments_not_finished)}) no han sido finalizados: {experiments_not_finished}")	
	
	
			'''
			for i, exp2run in enumerate(ps2):
				print(f"Init the {i+1}/{len(ps2)}; Hora -> {transform_time(time.time())}", len(exp2run))
				with multiprocessing.Pool(num_processors) as pool:
						
					pool.map(experiment, ps2[i])
					print(f'Finished {i+1}/{len(ps2)}')
			'''
			
			end_time = time.time()
			duration = end_time - start_time
			print(f'Total time: {duration/3600:.2f} hours')

		if args.stats:
			ms = MetaStats()
