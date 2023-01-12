"""
04-11-201

@author: Nicolas Moreno Chaparro
         nmoreno@bcamath.org

This script generates a lammps data file with the coordinates for different structures. Currently customized for viral
models. 
"""
import sys, os, re
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

class dataFile():

   def __init__(self, oFile, dim,lbox, R,RP, m, rhoN, collID, ellips, ro,rop,nspikes,lp,ps,deform,eps1,eps2,xi,alpharange, nColloids, eFact, nSolv, solvFrac, patchType, shellOnly=False,location=-1,ks=0,simtype='ring',distribution = 0, blender=False, patchFrac=1.0,lp2=0):
      #Variable location is used to place initially all the blocks in one side of the box.
      self.ex, self.ey, self.ez = 1.0, 0.9, 0.7
      self.distribution = distribution   ###value of 0 corresponds to homogeneous patches around the core
      self.dummyvar = ellips
      self.simtype = simtype
      self.f = oFile+'.data'
      self.fxyz = oFile
      self.lbox = lbox
      self.R = R
      self.m = m.split(',')
      self.rho = rhoN
      self.dh = (1/rhoN)**(1./dim)
      self.ro = ro
      self.nColloids = nColloids
      self.dimension=dim
      self.eFact = eFact
      self.nSolv = nSolv
      print (nSolv, "number of solvent types")
      self.solvFrac = np.array(solvFrac.split(','), dtype=float)
      self.location = location
      self.collID = collID
      self.nPart = int(self.lbox**3*self.eFact*self.rho)
      #Adicional condicion slip 
      self.slip_c = xi
      self.aplha1 = alpharange
      
      

      self.patchType = patchType

      if self.patchType==7: #only for mix cases, otherwise will no use the value in the input.
        self.patchFrac = patchFrac  ##Fraction of patches for mixed path test
      #else: self.patchFrac=1

      self.shellOnly = shellOnly    
      #self.RSolvent=self.RP+self.lp*self.ro+(self.ps)*self.ro
      self.V, self.F = [],[]
      self.E = np.zeros([10000,10000])  
      #self.a, self.x, self.y, self.z = self.readDataFile(self.bandID).T
      if self.shellOnly==0 or self.shellOnly==2:
          print('gen core')
          self.xs, self.ys, self.zs,self.ro = self.genIcoSurf(self.R,self.ro,self.V,self.F,self.E)
          #since iso surface, distance between points ro is different from input, self.ro updated
          print('shell with %s particles' %len(self.xs))
      elif self.shellOnly==4:
          self.patchType = 0  #by default if core is tetra then no spikes are created
          print('gen tetra core')
          self.xs, self.ys, self.zs,self.ro = self.genTetra(self.R,self.ro,self.V,self.F,self.E,True)

          #rotBands = np.vstack([self.xs,self.ys,self.zs]).T
          #aR = self.rotation_matrix(np.array([0,1,1]),np.array([0.3,0.1,0.7]))
          #self.xs,self.ys,self.zs = aR.apply(rotBands).T
          print('shell with %s particles' %len(self.xs))
      else: self.xs=[]

      self.RP = self.R+self.ro

      self.sideN = np.round((((self.R+self.ro/2)*ps)/(3/8)**0.5)/self.ro,0)-1  ##number of particles per side of tetra
      if self.sideN==0: self.sideN=1

      self.ps = ((self.sideN*self.ro)*(3/8)**0.5)/self.ro ##divided by ro to be consistent with other method that multiply by ro later ## ##int((self.R*ps)/self.ro)
      self.ps1 =  int((self.R*ps)/self.ro)
      print('ps input %s, recomputed %s , sideN%s'%(self.ps1,self.ps,self.sideN))
      self.lp = int((self.R*lp)/self.ro) ##levels to grow popmers
      self.lp2 = int((self.R*lp2)/self.ro) ##levels to grow popmers
      self.rop = self.ro*rop
      self.RSolvent=self.RP

      self.nspikes=nspikes 
      self.deform = deform

      self.posRan = 0
      self.xp, self.yp, self.zp = np.array([]), np.array([]), np.array([])
      if self.patchType>0:
          print('gen patch sites')
          #self.ps = int((self.R*ps)/self.ro)
          self.lp = int((self.R*lp)/self.ro) ##levels to grow popmers
          self.lp2 = int((self.R*lp2)/self.ro) ##levels to grow popmers

          self.RSolvent+= self.lp*self.ro+(2*self.ps)*self.ro
          self.Vp, self.Fp = [],[]
          self.Ep = np.zeros([10000,10000])  
          if self.distribution == 0:
            self.xp0, self.yp0, self.zp0,self.rop = self.genIcoSurf(self.RP,self.rop,self.Vp,self.Fp,self.Ep)
            meshsize = len(self.xp0)

          if self.distribution == 1:
            self.xp0, self.yp0, self.zp0 = [], [], []
            self.nspikes = nspikes
            tempxp, tempyp, tempzp, ropEstimate = self.genIcoSurf(self.RP,self.rop,self.Vp,self.Fp,self.Ep) #generate mesh to pick pos of patches
            meshsize = len(tempxp)
            if meshsize<nspikes:
              print('Submesh used to generate random position of patch is too coarse to allocate the target spikes, use a smaller rop')
              tempxp, tempyp, tempzp, self.ropEstimate = self.genIcoSurf(self.RP,self.ro,self.Vp,self.Fp,self.Ep) #generate mesh to pick pos of patches
              meshsize = len(tempxp)

              if meshsize<nspikes: raise Exception('Not possible to allocate required patches with the current resolution ro')

            unique = 0
            while unique==0: ##Checks that the random positions of the patches is unique
              posRan = np.array([],dtype=int)
              ind = np.arange(meshsize)
              unique,self.posRan  = self.getRandomint(ind, self.nspikes,posRan)

            self.xp0 = tempxp[self.posRan]
            self.yp0 = tempyp[self.posRan]
            self.zp0 = tempzp[self.posRan]

#            for i in range(self.nspikes):
#                aFlag = 0
#                while aFlag==0:
#                  posRan = np.random.randint(0,meshsize)
#                 if tempxp[posRan]!= 0:
#                    aFlag = 1
#                self.xp0.append(tempxp[posRan])
#                self.yp0.append(tempyp[posRan])
#                self.zp0.append(tempzp[posRan])
#                tempxp[posRan], tempyp[posRan], tempzp[posRan] = 0,0,0 
#            self.xp0, self.yp0, self.zp0 = np.array(self.xp0), np.array(self.yp0), np.array(self.zp0)

          if self.distribution == 2:
            self.xp0, self.yp0, self.zp0 = self.randomPositionAtRo(self.RP,nspikes) ##This method showed some issues to distributed properly
            meshsize = len(self.xp0)
            print('creating %s patches, from %s available sites' % (len(self.xp0), meshsize))
      else: self.xp0, self.yp0, self.zp0 = [],[],[]
      if self.distribution == 3:  ### This distribution is given by the angle range with slip contidion.
          self.xp0, self.yp0, self.zp0 = self.alpharange(self.RP) ##This method showed some issues to distributed properly
          meshsize = len(self.xp0)
          print('creating %alpha range, from %s available sites' % (len(self.xp0), meshsize))
################################################################################################################################
      
     
      if self.dummyvar: ##Scale sphere as an ellipsoid, points distance will not be preserved 
            self.xs*=self.ex
            self.ys*=self.ey
            self.zs*=self.ez
           # if len(self.xp)>0:
           #   self.xp*=self.ex
           #   self.yp*=self.ey
           #   self.zp*=self.ez
            #if len(self.xc)>0:
            #  self.xc*=self.ex
            #  self.yc*=self.ey
            #  self.zc*=self.ez

      if self.patchType==1:
        print('gen linear')
        self.growLinear(0,self.nspikes)
      elif self.patchType==2:
        print('gen tetraSolid on rod')
        self.growTetra(ro,0,self.nspikes)
      elif self.patchType==3:
        print('gen sphere on rod')
        self.growSphere(ro)
      elif self.patchType==4:
        print('gen sphere solid patch')
        self.growSpherePatch(ro)
      elif self.patchType==5:
        print('gen tetra patch')
        self.growtetraPatch(ro)
      elif self.patchType==6:
        print('gen coarser tetra')
        self.growtetraPatchCoarse(self.ps*self.ro)
      elif self.patchType==7:
        print('gen mix patch 1 and 2')
        npat = int(len(self.xp0)*patchFrac)

        
        self.growTetra(ro,npat)
        self.lp=self.lp2
        self.growLinear(0,npat)

      
      if self.shellOnly==0:
        print('gen inner particles in colloid')
        self.xc,self.yc,self.zc = self.createSphereIter(self.R-self.ro/2,self.ro,len(self.xs))
      elif self.shellOnly==1:
        self.xc,self.yc,self.zc = self.createSphereIter(self.R,self.ro,0)
      elif self.shellOnly==3:
        self.xc,self.yc, self.zc = self.createSphereBin(self.R,self.ro,0)
      else: self.xc, self.yc, self.zc = [],[],[]

      if self.deform:
        self.deformIcos(eps1,eps2)
      #if self.shellOnly==2:
      #  self.m[0] = 0.012#self.ro**(self.dimension-1)
      #else:
      #  self.m[0] = 0.012#self.dh**self.dimension

      #self.m[1] = 0.012#self.dh**self.dimension
      #self.m[2] = 0.#self.dh**self.dimension
      
      print(self.ro,self.m)
      #self.x = np.hstack([self.xs,self.xp])
      #self.y = np.hstack([self.ys,self.yp])
      #self.z = np.hstack([self.zs,self.zp])
      ##Scale the band size so the average distance between particles is consistent with the interparticle distance I ususlly use.
      #scal = self.ro/dummyvar
      #self.x = scal*self.x
      #self.y = scal*self.y
      #self.z = scal*self.z

      self.parPerColloid = len(self.xs)+(1)*len(self.xp)+len(self.xc)
      self.colloidBeads  = self.parPerColloid*self.nColloids


      self.ks=ks #Spring constant of the bonds, if 0 bonds are not created
      self.bondPCol = 0 ###later can be input
      if self.ks>0:
        print ('ks values used:%s' % self.ks)
        self.nBonds        = self.bondPCol*self.nColloids
      else:
          self.nBonds = 0
          
      self.volColloid = 4/3.*np.pi*self.R**self.dimension

      self.nAngles      = 0 #self.nChains*(self.polyL-2)#+self.addNAngles
      self.solventBeads = int(self.nPart - self.volColloid*self.rho) #self.colloidBeads

      self.nPart = self.solventBeads+self.colloidBeads

      self.partPerSolvent = self.solventBeads*self.solvFrac

      self.xlo, self.xhi = -self.lbox/2., self.lbox/2.
      self.ylo, self.yhi = -self.lbox/2., self.lbox/2.
      self.zlo, self.zhi = -self.lbox/2.*self.eFact, self.lbox/2.*self.eFact
    
      self.p3 = (self.location)*(self.zhi-self.zlo)+self.zlo  ##initial vertical positions of the band
   #   self.z += self.p3

      self.aType  = self.nSolv + 3 ##Only one type of particle for the colloid
      self.bType  = 1 #+ len(self.blockFracs) - 1   ##Assuming that not consecutive block definition are not connected (e.g: for types [1,2,3] 1-1, 2-2, 3-3, 1-2, 2-3)
      self.anType = 1 # len(self.blockFracs) + len(self.addBlockFracs) #+ len(self.blockFracs) - 1   ##Same previous assumption

      if self.aType != len(self.m):
         print ("Stypes:%s, Colloid:%s" % (self.nSolv ,1))
         print ('ERROR ATOM TYPES AND MASS DO NOT MATCH: %d, %d' %(self.aType, len(self.m)))
         exit()

      ##Initialize global counters for atom and molecule ID
      self.currentA = 1
      self.currentM = 1
      self.currentBond = 1
      self.currentColloid = 0
      self.fblend=oFile+'.xyz'
      self.blender = blender

 ## Creating Header
      if self.simtype=='xyz':
        f = open(self.fxyz, 'w') 
        f.write(self.colloidBeads.__str__()+"\n")
        f.close()
      if self.blender==True:
        
        f = open(self.fblend, 'w') 
        f.write(self.colloidBeads.__str__()+"\n")
        f.write("xyz file to render in blende\n")
        f.close()
      self.header()

   def header(self):
      f = open(self.f, 'w')
      f.write("# Lammps data file generator \n")
      f.write("# @Author: Nicolas Moreno \n")
      #f.writeself.setup
      
      f.write("  \n")
      f.write(self.nPart.__str__()+" atoms \n")
      f.write((self.nBonds).__str__()+" bonds \n")
      f.write((self.nAngles).__str__()+" angles \n")
      f.write("0 dihedrals \n")
      f.write("0 impropers \n")
          
      f.write("  \n")
      f.write(self.aType.__str__()+" atom types \n")
      f.write(self.bType.__str__()+" bond types \n")
      f.write(self.anType.__str__()+" angle types \n")
      f.write("0 dihedral types \n")
      f.write("0 improper types \n")

      #Box limits
      f.write("  \n")
      f.write(self.xlo.__str__()+" "+self.xhi.__str__()+" xlo xhi \n")
      f.write(self.ylo.__str__()+" "+self.yhi.__str__()+" ylo yhi \n")
      
      if self.dimension==2:
        f.write("0.0 0.2 zlo zhi \n")
      else:
        f.write(self.zlo.__str__()+" "+self.zhi.__str__()+" zlo zhi \n")
      
      #Masses
      f.write("  \n")
      f.write("Masses \n \n")
      for i in range(len(self.m)):
         f.write((i+1).__str__()+" "+self.m[i].__str__()+" \n")
         
      f.write("\n Atoms")
      f.write("  \n \n")
      f.close()

   def wBond(self, f, idbond,p1,p2):
        IDshift = self.currentColloid*self.parPerColloid ##shifting in the ID according to the current saved bond data for colloids
        #print IDshift, self.currentColloid
        f.write("%s 1 %s %s \n"% (idbond, p1+IDshift,p2+IDshift))

   def atoms(self):
       f = open(self.f, 'a')
       if self.simtype=='xyz':
          fxyz = open(self.fxyz, 'a')
       else:
       	  fxyz = ''
       if self.blender:
          fblend = open(self.fblend,'a')
       else:
          fblend = ''

       # WRITTING ON FILE POLYMER BEADS COORDS
       for nC in range(self.nColloids):
           #self.createMobius(f)
           if self.shellOnly==0 or self.shellOnly==2 or self.shellOnly==4:
             # self.createISOColloid(f,self.xs,self.ys,self.zs,1,fxyz=fxyz,fblend = fblend)
                self.createISOColloid(f,self.xs,self.ys,self.zs,1,fxyz=fxyz,fblend = fblend)
           
           if self.patchType>0:
                self.createISOColloid(f,self.xp,self.yp,self.zp,2,fxyz=fxyz,fblend = fblend)
           if self.shellOnly==1 or self.shellOnly==0 or self.shellOnly==3:
                self.createISOColloid(f,self.xc,self.yc,self.zc,3,fxyz=fxyz,fblend = fblend)
               # self.createISOColloid(f,self.xp2,self.yp2,self.zp2,3)

          
       # WRITTING ON FILE SOLVENT BEADS COORDS
       if self.location>2:
#          xc, yc, zc = self.randomXYZarrayAtLocation(self.solventBeads)
          xc, yc, zc = self.randomXYZarrayAtLevels(self.solventBeads)
       else:
          xc, yc, zc = self.randomXYZAroundSphere(self.RSolvent)
       a = self.currentA  ##Takes the last indezx used in polymer bead definition
       t = 4
       m = self.currentM

       k=0 #starting counter
       flagS = 0 ## this is a flag to check when the type of solvent must be changed. I prefer to do this in this way rather that define an array with the type of all the solvents since this data is ussualy the larger, so its better not to stored.
       pps = self.partPerSolvent[flagS]
       
       if self.simtype=='colloid' or self.simtype=='xyz':
           for i in range(self.solventBeads):
              f.write("%d %d %d %f %f %f\n" % (a+k, m+k, t+flagS, xc[i],yc[i], zc[i]))
              if pps <= k:
                 flagS+=1
                 pps += self.partPerSolvent[flagS]
              k+=1

       elif self.simtype=='sdpd':
            for i in range(self.solventBeads):
              f.write("%d %d %f %f %f %d 1.0 0.0 125.0\n" % (a+k, t+flagS, xc[i],yc[i], zc[i],m+k))
              if pps <= k:
                 flagS+=1
                 pps += self.partPerSolvent[flagS]
              k+=1

       if self.nBonds>0:
            # WRITTING ON FILE BONDS
            k=0
            print >>f,"\n Bonds"
            print >>f, " "
            for i in range(self.nColloids):
                self.defineBonds(f)

        # WRITTING ON FILE ANGLES
       if self.nAngles>0:
            k=0
            print >>f,"\n Angles"
            print >>f, " "
            for i in range(self.nChains):
                for j in range(1,self.polyL-1):
                    f.write("%d %d %d %d %d\n" % (allia[k],allta[k], j+(i*self.polyL), j+1+(i*self.polyL), j+2+(i*self.polyL)))
                    k+=1
        
       f.close()

   def createISOColloid(self,f,x,y,z,ptype,pos=[0.0,0.0,0.0],fxyz='', fblend='',normal_v,slip_c):

        rotBands = np.vstack([x,y,z]).T
        #rotBands = np.round(rotBands*self.rotate(0.,2)*self.rotate(np.pi/2,0)*self.rotate(0,1),4)   ###Pi/2 changend for the DK bands and lim surfaces because they are not align in the source file        
#        for i,j,k in zip(self.x,self.y,self.z):
#            f.write("%d %d %d %f %f %f\n" % (self.currentA, self.currentM, 1, i,j,k))
#            self.currentA+=1
#        self.currentM+=1

        if self.simtype=='colloid' or self.simtype=='xyz':
            for i in rotBands:
                f.write("%d %d %d %f %f %f\n" % (self.currentA, self.currentM, ptype, i[0]+pos[0],i[1]+pos[1], i[2]+self.p3+pos[2]),normal_v,slip_c)
                self.currentA+=1
            self.currentM+=1
        elif self.simtype=='sdpd':
            for i in rotBands:
                f.write("%d %d %f %f %f %d 1.0 0.0 125.0\n" % (self.currentA, ptype, i[0]+pos[0],i[1]+pos[1], i[2]+self.p3+pos[2], self.currentM))  ###Swap molecule to the end for hybrid
                self.currentA+=1
            self.currentM+=1
        if self.simtype=='xyz':
            for i in rotBands:
                fxyz.write("%f %f %f\n" % (i[0]+pos[0],i[1]+pos[1], i[2]+self.p3+pos[2]))  ###Write data file in xyz format no atom nor molec id
        if self.blender:    
            for i in rotBands:
                fblend.write("%d %f %f %f\n" % (ptype, i[0]+pos[0],i[1]+pos[1], i[2]+self.p3+pos[2]))  ###Write data file in xyz format no atom nor molec id

   def deformIcos(self,eps1,eps2):
        phi = np.arctan2(self.ys,self.xs)
        theta = np.arccos(self.zs/(self.xs**2+self.ys**2+self.zs**2)**0.5)

        self.xs = (self.R+(self.ro*eps1)*np.sin(phi*eps2)*np.sin(theta*eps2))*np.sin(theta)*np.cos(phi)
        self.ys = (self.R+(self.ro*eps1)*np.sin(phi*eps2)*np.sin(theta*eps2))*np.sin(theta)*np.sin(phi)
        self.zs = (self.R+(self.ro*eps1)*np.sin(phi*eps2)*np.sin(theta*eps2))*np.cos(theta)
        #return xg,yg,zg

   def growLinear(self, minPatch=0, maxPatch=-1):
        #a1, a2 = ((np.random.rand(1)-0.5)*np.pi/2.), ((np.random.rand(1)-0.5)*np.pi/2.)
        a1,a2=0,0
        print('min and max patch %s %s'% (minPatch, maxPatch))
        phi = np.arctan2(self.yp0[minPatch:maxPatch],self.xp0[minPatch:maxPatch])
        theta = np.arccos(self.zp0[minPatch:maxPatch]/(self.xp0[minPatch:maxPatch]**2+self.yp0[minPatch:maxPatch]**2+self.zp0[minPatch:maxPatch]**2)**0.5)
        

        if self.dummyvar==False:
           self.ex = 1.
           self.ey = 1.
           self.ez = 1.

        for i in range(0,self.lp):
            self.xp = np.append(self.xp,(self.RP+i*self.ro)*np.sin(theta+a1)*np.cos(phi+a2)*self.ex)
            self.yp = np.append(self.yp,(self.RP+i*self.ro)*np.sin(theta+a1)*np.sin(phi+a2)*self.ey)
            self.zp = np.append(self.zp,(self.RP+i*self.ro)*np.cos(theta+a1)*self.ez)

        #if self.dummyvar: ##Scale sphere as an ellipsoid, points distance will not be preserved 
        #    if len(self.xp0)>0:
        #      self.xp*=self.ex
        #      self.yp*=self.ey
        #      self.zp*=self.ez
            #if len(self.xc)>0:
            #  self.xc*=self.ex
            #  self.yc*=self.ey
            #  self.zc*=self.ez
        #return xg,yg,zg

   def growSpherePatch(self,ro):
      for i,j,k in zip(self.xp0,self.yp0,self.zp0):
        self.V,self.F = [],[]
        self.E = np.zeros([10000,10000])  
        x, y, z,rt= self.genIcoSurf(self.ps*self.ro,ro,self.V,self.F,self.E)
        self.xp = np.append(self.xp,x+i)
        self.yp = np.append(self.yp,y+j)
        self.zp = np.append(self.zp,z+k)

   def growSphere(self,ro):
        phi = np.arctan2(self.yp0,self.xp0)
        theta = np.arccos(self.zp0/(self.xp0**2+self.yp0**2+self.zp0**2)**0.5)
        px0= (self.RP+(self.lp+self.ps)*self.ro)*np.sin(theta)*np.cos(phi)
        py0 =(self.RP+(self.lp+self.ps)*self.ro)*np.sin(theta)*np.sin(phi)
        pz0 =(self.RP+(self.lp+self.ps)*self.ro)*np.cos(theta)
        
       
        self.growLinear(0,self.nspikes)
        if self.dummyvar: ##Scale sphere as an ellipsoid, points distance will not be preserved 
              px0*=self.ex
              py0*=self.ey
              pz0*=self.ez

        for i,j,k in zip(px0,py0,pz0):
            self.V,self.F = [],[]
            self.E = np.zeros([10000,10000])  
            x, y, z,rt= self.genIcoSurf(self.ps*self.ro,ro,self.V,self.F,self.E)
            #rotBands = np.vstack([x,y,z]).T
            #aR = self.rotation_matrix(np.array([0,0,1]),np.array([i,j,k]))
            #x,y,z = aR.apply(rotBands).T
            self.xp = np.append(self.xp,x+i)
            self.yp = np.append(self.yp,y+j)
            self.zp = np.append(self.zp,z+k)

   def growTetra(self,ro, minPatch=0, maxPatch=-1):

        phi = np.arctan2(self.yp0[minPatch:maxPatch],self.xp0[minPatch:maxPatch])
        theta = np.arccos(self.zp0[minPatch:maxPatch]/(self.xp0[minPatch:maxPatch]**2+self.yp0[minPatch:maxPatch]**2+self.zp0[minPatch:maxPatch]**2)**0.5)
        px0= (self.RP+(self.lp)*self.ro)*np.sin(theta)*np.cos(phi)
        py0 =(self.RP+(self.lp)*self.ro)*np.sin(theta)*np.sin(phi)
        pz0 =(self.RP+(self.lp)*self.ro)*np.cos(theta)
        if self.patchType!=7:
          self.growLinear(0,self.nspikes)
        else: self.growLinear(minPatch,maxPatch)

        if self.dummyvar: ##Scale sphere as an ellipsoid, points distance will not be preserved 
              px0*=self.ex
              py0*=self.ey
              pz0*=self.ez

        for i,j,k in zip(px0,py0,pz0):
            self.V,self.F = [],[]
            self.E = np.zeros([10000,10000])  
            x, y, z,rt= self.genTetra(self.ps*self.ro,self.ro,self.V,self.F,self.E)
            rotBands = np.vstack([x,y,z]).T
            aR = self.rotation_matrix(np.array([0,0,1]),np.array([i,j,k]))
            x,y,z = aR.apply(rotBands).T
            self.xp = np.append(self.xp,x+i)
            self.yp = np.append(self.yp,y+j)
            self.zp = np.append(self.zp,z+k)


   def getRandomint(self,freeInd, spikes,posRan):   
   ##Iteratively select random non-repeated integers from a set of index        
        pR = np.random.randint(0,len(freeInd),spikes)
        uniqueVec = np.unique(pR)
        ind = np.arange(len(freeInd))
        posRan = np.append(posRan,freeInd[uniqueVec])
        #print('posran',posRan, len(uniqueVec))
        if len(uniqueVec)==spikes: 
               unique=1
        else:  
            availMesh = np.setdiff1d(freeInd,freeInd[uniqueVec])  # corrected to freeInd[uniqueVec] since unique vec is an array of indes and should not be comapred directly.
            #print('av mesh',availMesh, len(availMesh))
            unique,posRan = self.getRandomint(availMesh, spikes-len(uniqueVec),posRan)

        return unique,posRan 


   def growtetraPatch(self,ro):
        phi = np.arctan2(self.yp0,self.xp0)
        theta = np.arccos(self.zp0/(self.xp0**2+self.yp0**2+self.zp0**2)**0.5)
        px0= (self.RP)*np.sin(theta)*np.cos(phi)
        py0 =(self.RP)*np.sin(theta)*np.sin(phi)
        pz0 =(self.RP)*np.cos(theta)
        
        if self.dummyvar: ##Scale sphere as an ellipsoid, points distance will not be preserved 
              px0*=self.ex
              py0*=self.ey
              pz0*=self.ez

        for i,j,k in zip(px0,py0,pz0):
            self.V,self.F = [],[]
            self.E = np.zeros([10000,10000])  
            x, y, z,rt= self.genTetra(self.ps*self.ro,self.ro,self.V,self.F,self.E)
            rotBands = np.vstack([x,y,z]).T
            aR = self.rotation_matrix(np.array([0,0,1]),np.array([i,j,k]))
            x,y,z = aR.apply(rotBands).T
            self.xp = np.append(self.xp,x+i)
            self.yp = np.append(self.yp,y+j)
            self.zp = np.append(self.zp,z+k)
        print('distance for patch %s' % rt)

   def growtetraPatchCoarse(self,ro):
        phi = np.arctan2(self.yp0,self.xp0)
        theta = np.arccos(self.zp0/(self.xp0**2+self.yp0**2+self.zp0**2)**0.5)
        px0= (self.RP)*np.sin(theta)*np.cos(phi)
        py0 =(self.RP)*np.sin(theta)*np.sin(phi)
        pz0 =(self.RP)*np.cos(theta)
        if self.dummyvar: ##Scale sphere as an ellipsoid, points distance will not be preserved 
              px0*=self.ex
              py0*=self.ey
              pz0*=self.ez
        
        for i,j,k in zip(px0,py0,pz0):
            x, y, z = self.tethraedra(i,j,k,ro)
            #print (x[3],y[3],z[3])
            self.xp = np.append(self.xp,x+i)
            self.yp = np.append(self.yp,y+j)
            self.zp = np.append(self.zp,z+k)


   def tethraedra(self,i,j,k,ro):
        x1 = np.array([(8/9)**0.5,-(2/9.)**0.5,-(2/9.)**0.5,0.0])   
        y1 = np.array([0.0,(2./3)**0.5,-(2./3)**0.5,0.0])
        z1 = np.array([1+1/3.,1+1/3.,1+1/3.,0.0])
        aR = self.rotation_matrix(np.array([0,0,1]),np.array([i,j,k]))
        r = (8/3.)**0.5
        #print (x1[3],y1[3],z1[3])
        rotBands = np.vstack([x1,y1,z1]).T
        rotBands = aR.apply(rotBands)#np.dot(rotBands,aR)
        #rotBands = np.round(rotBands*self.rotate(0,0)*self.rotate(psi,2)*self.rotate(theta,1),4)   ###Pi/2 changend for the DK bands and lim surfaces because they are not align in the source file
        
        return rotBands.T/r*ro#*1.2


   def rotate(self,theta,axis):
        c, s = np.cos(theta), np.sin(theta)
        if axis ==0:
            R = np.matrix([[1.,0.,0.], [0,c, -s], [0,s, c]])
        elif axis == 2:
            R = np.matrix([[c, -s,0], [s, c,0],[0.,0.,1.]])
        elif axis == 1:
            R = np.matrix([[c, 0.,s], [0., 1.,0.],[-s,0.,c]])
        return R.T

   def rotation_matrix(self,f,t):
    # a and b are in the form of numpy array
    ##fromo https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
       ax = f[0]
       ay = f[1]
       az = f[2]

       bx = t[0]
       by = t[1]
       bz = t[2]

       au = f/(np.sqrt(ax*ax + ay*ay + az*az))
       bu = t/(np.sqrt(bx*bx + by*by + bz*bz))

       rot=np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])
       rot = Rot.from_matrix(rot)
       return(rot)

   def randomPositionAtRoBiased(self, ro):
        x = np.random.rand(1)*(ro-ro/2)
        y = np.sqrt(np.random.rand(1)*(ro**2-x**2))
        z = np.sqrt(ro**2-x**2-y**2)
        return x,y,z

   def randomPositionAtRo(self, ro,Npoints=1):
       alpha = np.random.rand(Npoints)*2*np.pi
       beta = np.random.rand(Npoints)*2*np.pi
       x = ro*np.sin(alpha)*np.cos(beta)
       y = ro*np.sin(alpha)*np.sin(beta)
       z = ro*np.cos(alpha)
       return x,y,z
   
   def alpharange(self, ro,alpha1):
       alpha = alpha1*2*np.pi
       beta = np.pi   # We fix the beta angle in 180 degrees 
       x = ro*np.sin(alpha)*np.cos(beta)
       y = ro*np.sin(alpha)*np.sin(beta)
       z = ro*np.cos(alpha)
       return x,y,z   
    

   def randomXYZarray(self, randomPoints):
        print (randomPoints, 'randomPoints')
        x = np.random.rand(randomPoints)*(self.xhi-self.xlo)+self.xlo
        y = np.random.rand(randomPoints)*(self.yhi-self.ylo)+self.ylo
        z = np.random.rand(randomPoints)*(self.zhi-self.zlo)+self.zlo

        return x, y, z

   def randomXYZarrayAtLocation(self, randomPoints):
        print (randomPoints, 'randomPoints')
        z = self.polyLimits +np.random.rand(randomPoints)*(self.zhi-self.polyLimits)#(1-self.pFrac+self.addFrac)-self.polyLimits
                                                                                    #        x = np.random.rand(randomPoints)*(self.lbox*(1-self.pFrac+self.addFrac))-self.lbox/2
        print (self.polyLimits)
        y = np.random.rand(randomPoints)*(self.yhi-self.ylo)+self.ylo
        x = np.random.rand(randomPoints)*(self.xhi-self.xlo)+self.xlo

        return x, y, z  


   def randomXYZarrayAtLevels(self, randomPoints):
        print (randomPoints, 'randomPoints')
        beadS1 = int(randomPoints*self.solvFrac[0])
        beadS2 = randomPoints-beadS1

        x = np.random.rand(randomPoints)*(self.xhi-self.xlo)+self.xlo
        y = np.random.rand(randomPoints)*(self.yhi-self.ylo)+self.ylo

        frOfBox = self.pFrac+self.addFrac+self.solvFrac[0]*(1-self.pFrac+self.addFrac)
        z1 = np.random.rand(beadS1)*(self.lbox*self.eFact*frOfBox)-self.zhi
        z2 = self.zhi-np.random.rand(beadS2)*(self.lbox*self.eFact*(1-frOfBox))#(1-self.pFrac+self.addFrac)-self.polyLimits
                                                                                    #        x = np.random.rand(randomPoints)*(self.lbox*(1-self.pFrac+self.addFrac))-self.lbox/2
        print (self.polyLimits)
        #print x.shape, z2.shape, z1.shape, self.solvFrac[0], beadS1, beadS2,randomPoints-beadS1,'shaoes'
        if len(z2)>0:
            z = np.append(z1,z2,1)
            print ("%s Particles for second solvent included" % len(z2))
        else:
            z = z1
        #print x.shape, z.shape, z2.shape, z1.shape, self.solvFrac[0], beadS1, beadS2,randomPoints-beadS1,'shaoes'
        return x, y, z

   def randomXYZAroundSphere(self,R):
        #print (randomPoints, 'randomPoints')
         #self.solventBeads = self.nPart - self.colloidBeads
        
        x = np.arange(self.xlo,self.xhi,self.dh)
        y = np.arange(self.ylo,self.yhi,self.dh)
        z = np.arange(self.zlo,self.zhi,self.dh)
        #print(x)
        xx,yy,zz =np.meshgrid(x, y,z)
        xx,yy,zz =  xx.flatten(),yy.flatten(),zz.flatten()
        posR = (xx*xx+yy*yy+zz*zz)**0.5
        x,y,z = [],[],[]
        todel = len(xx)-self.solventBeads
        print("solvent to delete %s, with dh %s" %(todel,self.dh))
        for i in range(len(xx)):
            if posR[i]<=R:
                if todel==0:          
                    alpha = np.random.rand(1)*2*np.pi
                    beta = np.random.rand(1)*2*np.pi
                    ro = np.random.rand(1)*(((self.xhi-self.xlo)-R)/2)+R ####THIS IS ASSUMING CUBIC BOX
                    x.append(ro*np.sin(alpha)*np.cos(beta))
                    y.append(ro*np.sin(alpha)*np.sin(beta))
                    z.append(ro*np.cos(alpha))
                else:
                    todel-=1  ##deleted from grid
            else:
                x.append(xx[i]), y.append(yy[i]), z.append(zz[i])

        if todel>0:
            print('More solvent particles that required')
        else:
            print(self.solventBeads, todel, len(x))

        return x[0:self.solventBeads], y[0:self.solventBeads], z[0:self.solventBeads]
   
   def createSphere(self, R,dh,npshell,d=3):
        vol  = 4/3.*np.pi*(R)**d
        rho  = 1/dh**d
        print (np.round(vol*rho,0), npshell,'part for rho and currently in shell')
        part = int(np.round(vol*rho,0))-npshell
        x = (R-dh/2)*(2*np.random.uniform(size=part)-1.)
        y = ((R-dh/2)**2-x**2)**0.5*(2*np.random.uniform(size=part)-1.)
        z = ((R-dh/2)**2-x**2-y**2)**0.5*(2*np.random.uniform(size=part)-1.)
        print('n of part in core created %s '%len(x))
        return x,y,z

   def createSphereBin(self,R,dh,npshell,d=3):
        numlayer = np.ceil(R/dh)
        x,y,z =[],[],[]
        rads = R-np.arange(1,numlayer+1)*dh+dh/2
        for rad in rads:
            x.append(rad)
            y.append(0.0)
            z.append(0.0)
            nphi = int(np.pi*rad/dh)
            dphi = np.pi/nphi
            print(dphi,nphi,rad)
            phi=0
            for i in range(1,(nphi)):
                phi+=dphi
                ntheta = int(2*np.pi*rad*np.sin(phi)/dh)
                #print(ntheta)
                dtheta = np.pi*2/ntheta
                #print(dtheta,ntheta)

                theta=0
                for j in range(1,(ntheta)+1):
                    
                    x.append(rad*np.cos(phi))
                    y.append(rad*np.sin(phi)*np.sin(theta))
                    z.append(rad*np.sin(phi)*np.cos(theta))
                    theta+=dtheta

            x.append(-rad)
            y.append(0.0)
            z.append(0.0)
        return x,y,z

   def createSquare(self,R,dh,d=3):
    
        vol  = (2*R)**d
        rho  = 1/dh**d
        part = int(np.round(vol*rho,0))
        x = (R)*(2*np.random.uniform(size=part)-1.)
        y = (R)*(2*np.random.uniform(size=part)-1.)
        z = (R)*(2*np.random.uniform(size=part)-1.)
        return x,y,z

   def createSphereIter(self,R,dh,npshell,d=3):
        xc,yc,zc = self.createSquare(5*R,dh,d=3)
        x,y,z = [],[],[]
        vol  = 4/3.*np.pi*(R)**d
        rho  = 1/dh**d

        part = int(np.round(vol*rho,0))-npshell
        
        for i in range(len(xc)):
            if xc[i]**2+yc[i]**2+zc[i]**2<R**2 and len(x)<part:
                x.append(xc[i])
                y.append(yc[i])
                z.append(zc[i])

        #z = np.random.uniform(size=part)
        print (np.round(vol*(1/self.dh**d),0),np.round(vol*rho,0), npshell,len(x),'part for rhoOrig, rho, currently in shell and core')

        return x,y,z
   
   def createSphereShell(self, R,dh,npshell,d=3):
        vol  = 4/3.*np.pi*(R)**d
        rho  = 1/dh**d
        part = int(np.round(vol*rho,0))-npshell
        theta = 2*np.pi*np.random.uniform(size=part)
        #phi = np.pi*np.random.rand(part)
        phi = np.arccos(2*np.random.uniform(size=part)-1.0);
        r = R
        x= r*np.cos(theta)*np.sin(phi)
        y= r*np.sin(theta)*np.sin(phi)
        z= r*np.cos(phi)
        
        return x,y,z

   def genTetra(self,R,dh,V,F,E, cent=False):
        zpos=0
        sig=1
        if (cent==True): 
          zpos = 1
          sig=-1

        ro = dh  ##If different that 2. the value of the bond distance is fixed but R is and output

        self.addVertexTetra(V,R*(8/9)**0.5,  R*0.0, R*(1+sig*1/3.-zpos))  #0
        self.addVertexTetra(V, R*-(2/9.)**0.5, R*(2./3)**0.5, R*(1+sig*1/3.-zpos)) #1
        self.addVertexTetra(V,R*-(2/9.)**0.5, R*-(2./3)**0.5,  R*(1+sig*1/3.-zpos)) #2
        self.addVertexTetra(V,R* 0.0, R*0.0,  R*zpos) #3


        ##   // 5 faces around point 0
        self.addFaces(F,0, 1,2)
        self.addFaces(F,0, 1, 3)
        self.addFaces(F,0, 2, 3)
        self.addFaces(F,1, 2, 3)


           # print (np.array(V).shape)
        #F = np.array(F)
        e1 = np.linalg.norm(np.array(V)[F[0][0]]-np.array(V)[F[0][2]])
        #print("Current ro %s" % e1)
        it = 0
        print ("Current ro tetra %s- Iteration %s" % (e1,it))

        while e1>ro:
            F = self.refine(F,V,E,R,False)
            it+=1
            e1 = np.linalg.norm(np.array(V)[F[0][0]]-np.array(V)[F[0][2]])
            print ("Current ro tetra %s - Iteration %s" % (e1,it))
        V = np.array(V)
        e1 = np.linalg.norm(V[F[0][0]]-V[F[0][1]])
        print ("Final ro tetra %s and R %s" % (e1,R))
    
        return V.T[0],V.T[1],V.T[2], e1

   def genIcoSurf(self,R,dh,V,F,E):

        ro = dh  ##If different that 2. the value of the bond distance is fixed but R is and output
        ##if R is defined the radius of the icosphere is prescribed and the ro is output
        
        t = ro*(1.0 + np.sqrt(5.0)) / 2.0  ##bondDistance *golden ratio (ro*(1+sqrt(5))/2.0)  
        self.addVertex(V,-ro,  t,  0,R)
        self.addVertex(V, ro,  t,  0,R)
        self.addVertex(V,-ro, -t,  0,R)
        self.addVertex(V, ro, -t,  0,R)

        self.addVertex(V, 0, -ro,  t,R)
        self.addVertex(V, 0,  ro,  t,R)
        self.addVertex(V, 0, -ro, -t,R)
        self.addVertex(V, 0,  ro, -t,R)

        self.addVertex(V, t,  0, -ro,R)
        self.addVertex(V, t,  0,  ro,R)
        self.addVertex(V,-t,  0, -ro,R)
        self.addVertex(V,-t,  0,  ro,R)


        ##   // 5 faces around point 0
        self.addFaces(F,0, 11, 5)
        self.addFaces(F,0, 5, 1)
        self.addFaces(F,0, 1, 7)
        self.addFaces(F,0, 7, 10)
        self.addFaces(F,0, 10, 11)

        ##        // 5 adjacent faces 
        self.addFaces(F,1, 5, 9)
        self.addFaces(F,5, 11, 4)
        self.addFaces(F,11,10,2)
        self.addFaces(F,10,7,6)
        self.addFaces(F,7,1,8)

        #        // 5 faces around point 3

        self.addFaces(F,3, 9, 4)
        self.addFaces(F,3, 4, 2)
        self.addFaces(F,3, 2, 6)
        self.addFaces(F,3, 6, 8)
        self.addFaces(F,3, 8, 9)

        ##        // 5 adjacent faces 
        self.addFaces(F,4, 9, 5)
        self.addFaces(F,2,4,11)
        self.addFaces(F,6,2,10)
        self.addFaces(F,8,6,7)
        self.addFaces(F,9,8,1)


       # print (np.array(V).shape)
        #F = np.array(F)
        e1 = np.linalg.norm(np.array(V)[F[0][0]]-np.array(V)[F[0][1]])
        print("Current ro %s" % e1)
        it = 0
        while e1>ro:
            F = self.refine(F,V,E,R)
            it+=1
            e1 = np.linalg.norm(np.array(V)[F[0][0]]-np.array(V)[F[0][1]])
            print ("Current ro %s - Iteration %s, points %s" % (e1,it, len(V)))
        V = np.array(V)
        e1 = np.linalg.norm(V[F[0][0]]-V[F[0][1]])
        #print ("Final ro %s" % e1)
    
        return V.T[0],V.T[1],V.T[2], e1

   def addVertex(self,V, x,y,z,R):
        l = np.linalg.norm([x,y,z])
        V.append([R*x/l,R*y/l,R*z/l])
   
   def addVertexTetra(self,V, x,y,z):
        V.append([x,y,z])
   
   def addFaces(self,F, v1,v2,v3):
        F.append([v1,v2,v3])

   def getMiddlePoint(self,V,vin1,vin2,E,R,sphere=True):
        Vtemp = np.array(V)
        v1, v2 = Vtemp[vin1], Vtemp[vin2]
        if E[vin1,vin2]>0 or E[vin2,vin1]>0:
            return int(E[vin1,vin2])
        else:
            mx = (v1[0]+v2[0])/2.    
            my = (v1[1]+v2[1])/2.   
            mz = (v1[2]+v2[2])/2.  
            if sphere:
                self.addVertex(V,mx,my,mz,R)
            else:
                self.addVertexTetra(V,mx,my,mz)

            E[vin1,vin2] = E[vin2,vin1] = Vtemp.shape[0]
        return Vtemp.shape[0]
        
   def refine(self,F,V,E,R,sphere=True):
        F2 = []
        ##replace triangle by 4 triangles
        for face in F:
            v1,v2,v3 = face
            a = self.getMiddlePoint(V,v1, v2,E,R,sphere)
            b = self.getMiddlePoint(V,v2, v3,E,R,sphere)
            c = self.getMiddlePoint(V,v3, v1,E,R,sphere)

            self.addFaces(F2,v1, a, c)
            self.addFaces(F2,v2, b, a)
            self.addFaces(F2,v3,c,b)
            self.addFaces(F2,a,b,c)

        return F2
    
   def normalVect(self,VectNorm,V,x,y,z,R):
       VectNorm = [] 
       l = np.linalg.norm([x,y,z])
       VectNorm.append([2*x/l,2*y/l,2*z/l])   
       return VectNorm


if sys.argv[0] == 'createVirusPoints.py':
   oFile = sys.argv[1]
   R  = 1.     # Radius of the core/capsid
   RP = R+0.1  # Outer radius to start growing viral spikes
   ro = 0.2    # Distance between points of the structure. If isosurface for shell, ro is recomputed 
               # in general leading to same or smaller ro
   rop = 20    # ratio between spikes distance rspikes and ro. (this define density of spikes). rspikes = ro*rop
   nspikes = 10
   pType = 3   # spike type 0: none, 1: rod, 2: rod-tetra, 3: rod-sphere, 4: sphere, 5: tetra
   ps = 2      # radius of spike. When using spheres or tethraedra 
   shell = 3   # flag to change the type of capsid. Dense or shell only. 
               # 0 or 2 : only shell with equidistant points. 0: also creates randomly filled core
               # 1 or 3 for filled core. 3: equidistant points in core, 1: random distributed filled core 
   loc = 0.5   # location of the center of mass of the virus loc = position cm / size box
   lp=1        # lenght (number of points) of the spike
   eps1 = 2    # wavelength 1 for deformed sphere
   eps2 = 5    # wavelength 2 for deformed sphere
   deform=False   #Deformed core
   dist =0        #Distribution of spikes around core. 0: homogeneous, 1: random
   simType='colloid'  #Type of file - xyz or colloid or sdpd. xyz creates both xyz and dpd file. whereas colloid and sdpd only those

   ## If sdpdm the following parameters are needed 
   lbox = 10   # size of the periodic box for sdpd simulations
   bandid = 0  
   dummvar = False   ##False for spherica envelopes. True for covid-like ellipsoid, using a simple scaling of the axis.
   rhoN = np.round(1./ro**3,0) #particle density of dpd/sdpd simulations 
   m = ro**3                   #mass of the particle for dpd/sdpd simulations
   masses = '%s,%s,%s,%s' %(m,m,m,m)
       
   genBle = False   ###if the xyz file with format to render in blender need to be generated.            
   data = dataFile(oFile, 3,lbox, R,RP,masses, rhoN, bandid, dummvar, ro,rop,nspikes,lp,ps,deform,eps1,eps2, 1, 1., 1, "1.0, 0.0, 0.0", patchType=pType,shellOnly=shell,location=loc,ks=0,simtype=simType,distribution=dist, blender=genBle)
  # def __init__(self, oFile, dim,lbox, R,RP, m, rhoN, collID, dummyvar, ro,rop,lp,ps,deform,eps1,eps2, nColloids, eFact, nSolv, solvFrac, patchType, shellOnly=False,location=-1,ks=0,simtype='ring'):
   data.atoms()
else: print ("createVirus: Running from another script" )
   
