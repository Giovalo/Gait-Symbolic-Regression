import os

class StridesFromHws:
    
    def __init__(self):
        self.strides_dir = 'data/strides/kalman'
        self.dx = []
        self.sn = []
    

    def readFirstHws(self, myfile):
        time = []
        coordinates = []
        fr = open(myfile, "r")
        for line in range (0, 3):
            if line == 2:
                data = fr.readline().split(" ")
                freq = float(data[1])
            else:
                fr.readline()
        for line in fr:
            data = line.split(" ")
            time.append(float(data[0]))
            point = [int(data[1]), int(data[2])]
            coordinates.append(point)            
        fr.close
        return freq, time, coordinates
    
    
    
    def readHws(self, myfile):
        coordinates = []
        fr = open(myfile, "r")
        for line in range (0, 3):
            fr.readline()
        for line in fr:
            data = line.split(" ")
            point = [int(data[1]), int(data[2])]
            coordinates.append(point)            
        fr.close
        return coordinates
    
    
    
    def getStridesFromHws(self, video):
        list_dirs = os.listdir(self.strides_dir)
        for i in range(0, len(list_dirs)):
            dir_name = list_dirs[i].split('-')
            if dir_name[0] == video:
                myDir = self.strides_dir+'/'+list_dirs[i]+'/hws/'
                freq, time, Nose = self.readFirstHws(myDir+'nose.hws')
                Neck = self.readHws(myDir+'neck.hws')
                RShoulder = self.readHws(myDir+'rshoulder.hws')
                RElbow = self.readHws(myDir+'relbow.hws')
                RWrist = self.readHws(myDir+'rwrist.hws')
                LShoulder = self.readHws(myDir+'lshoulder.hws')
                LElbow = self.readHws(myDir+'lelbow.hws')
                LWrist = self.readHws(myDir+'lwrist.hws')
                RHip = self.readHws(myDir+'rhip.hws')
                RKnee = self.readHws(myDir+'rknee.hws')
                RAnkle = self.readHws(myDir+'rankle.hws')
                LHip = self.readHws(myDir+'lhip.hws')
                LKnee = self.readHws(myDir+'lknee.hws')
                LAnkle = self.readHws(myDir+'lankle.hws')
                strides = []
                for k in range (len(time)):
                    s = [Nose[k], Neck[k], RShoulder[k], RElbow[k], RWrist[k], LShoulder[k], LElbow[k], LWrist[k], RHip[k], RKnee[k], RAnkle[k], LHip[k], LKnee[k], LAnkle[k]]
                    strides.append(s)
                if dir_name[1] == 'dx':                    
                    self.dx.append([strides, time, freq])
                elif dir_name[1] == 'sn':
                    self.sn.append([strides, time, freq])
        return [self.dx, self.sn]
