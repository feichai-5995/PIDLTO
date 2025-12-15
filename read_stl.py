import os
import numpy as np

class Read_stl:
    def __init__(self,filename):
        if not os.path.exists(filename):
            raise ValueError(f'{filename} does not exists !!!')

        self.filename = filename

    def stlGetFormat(self):
        fid = open(self.filename,'rb')
        fid.seek(0,2)                
        fidSIZE = fid.tell()         
        if (fidSIZE-84)%50 > 0:
            stlformat = 'ascii'
        else:
            fid.seek(0,0)            
            header  = fid.read(80).decode() 
            isSolid = header[0:5]=='solid'
            fid.seek(-80,2)          
            tail       = fid.read(80)
            isEndSolid = tail.find(b'endsolid')+1

            if isSolid & isEndSolid:
                stlformat = 'ascii'
            else:
                stlformat = 'binary'
        fid.close()
        return stlformat
    
    def __READ_stlascii(self):
        fidIN = open(self.filename,'r')
        fidCONTENTlist = [line.strip() for line in fidIN.readlines() if line.strip()]     #Read all the lines and Remove all blank lines
        fidCONTENT = np.array(fidCONTENTlist)
        fidIN.close()

        line1 = fidCONTENT[0]
        if (len(line1) >= 7):
            stlname = line1[6:]
        else:
            stlname = 'unnamed_object'; 

        stringNORMALS = fidCONTENT[np.char.find(fidCONTENT,'facet normal')+1 > 0]
        coordN  = np.array(np.char.split(stringNORMALS).tolist())[:,2:].astype(float)

        facetTOTAL       = stringNORMALS.size
        stringVERTICES   = fidCONTENT[np.char.find(fidCONTENT,'vertex')+1 > 0]
        coordVall = np.array(np.char.split(stringVERTICES).tolist())[:,1:].astype(float)
        cotemp           = coordVall.reshape((3,facetTOTAL,3),order='F')
        coordV    = cotemp.transpose(1,2,0)

        return [coordV,coordN,stlname]

    def __READ_stlbinary(self): 
        import struct
        fidIN = open(self.filename,'rb')
        fidIN.seek(80,0)                                   
        facetcount = struct.unpack('I',fidIN.read(4))[0]   
        coordN  = np.zeros((facetcount,3))
        coordV = np.zeros((facetcount,3,3))
        for loopF in np.arange(0,facetcount):
            tempIN = struct.unpack(12*'f',fidIN.read(4*12))
            coordN[loopF,:]    = tempIN[0:3]   
            coordV[loopF,:,0] = tempIN[3:6] 
            coordV[loopF,:,1] = tempIN[6:9]   
            coordV[loopF,:,2] = tempIN[9:12]  
            fidIN.read(2); 
        fidIN.close()
        return [coordV,coordN]

    def __call__(self):
        stlformat = self.stlGetFormat()
        if stlformat=='ascii':
            [coordV,coordN,stlname] = self.__READ_stlascii()
        elif stlformat=='binary':
            [coordV,coordN] = self.__READ_stlbinary()
            stlname = 'unnamed_object'
        return [coordV,coordN,stlname]