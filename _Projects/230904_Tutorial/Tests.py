

#%% Test function



swicher = int(input('Input swicher:'))
subject = 2.333333

match swicher:
    case 1:
        print(f'Input is {swicher}, No Return')
    case 2:
        print(f'Input is {swicher}, Return is {subject:.2f}')
    case 3:
        raise IOError('Error Demo')


#%%
import OS_Tools_Kit as ot
acd = ot.Load_Variable(r'E:\Tutorial\230904_L76_slice\Results\All_Series_Dic.pkl')

#%% Range test
class List_Demo(object):

    def __init__(self,input_lists):
        self.lists = input_lists

    def __len__(self):
        return len(self.lists)
        
    def __getitem__(self,key):
        
        return self.lists[key]
    
    def print_All(self):
        for i,c_item in enumerate(self.lists):
            print(c_item)
#%%
demo = List_Demo([1,2,3,4,5])
