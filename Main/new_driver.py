import serial
class driver:
    def __init__(self,portx = "/dev/ttyUSB0",bps = 115200):
        self.portx = portx
        self.bps = bps
        timex = 0.01
        self.ser = serial.Serial(portx,bps,timeout = timex,parity='N',stopbits = 1,bytesize=8)

    def forward(self):
        self.ser.write("1")
        
    def set_speed(self,x,y,w):
        self.ser.write(("speed:%d,%d,%d\r\n"%(x,y,w)).encode())
        return self.ser.read(20).decode()

    def set_4_speed(self,v1, v2, v3, v4):
        self.ser.write(("4speed:%d,%d,%d,%d\r\n"%(v1, v2, v3, v4)).encode())
        return self.ser.read(20).decode()





