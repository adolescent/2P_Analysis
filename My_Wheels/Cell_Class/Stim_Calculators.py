
'''
This script include multiple vital functions for cell tuning calculation.
Codes following will generate:
1. Single cell tuning property(t values of OD/Orien/Color etc..)
2. T test maps (in format cell t value matrix)
3. Cell Response Curve. avr/std of each ID response.(for all cells)
4. This tool need to be updatable.
'''
from My_Wheels.Cell_Class.Format_Cell import Cell


class Stim_Cells(Cell):
    
    def __init__(self,*args, **kwargs): # Success all functions & variables in parent class Cell.
        super().__init__(*args, **kwargs)
        # And copy all variables.

    