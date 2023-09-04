

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
