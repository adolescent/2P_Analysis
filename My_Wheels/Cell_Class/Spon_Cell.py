
from My_Wheels.Cell_Class.Format_Cell import Cell




class Spon_Cell(Cell):
    
    name = r'Cell With Spon Process.'
    
    
    def __init__(self,all_cell_dic,avr_graph = 'Not_Given',stim_frame_align = 'Not_Given',runname = 'Run001'):
        
        super(Spon_Cell, self).__init__()
        
        
        