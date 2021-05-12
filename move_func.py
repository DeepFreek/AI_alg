def find_void(array):
    for row in range(len(array)):
        for col in range(len(array[row])):
            if array[row][col] == 0:
                return [row, col]

def move_down(array):
    [voidX, voidY]=find_void(array)
    A=array[voidX+1][voidY]
    array[voidX+1][voidY]=0
    array[voidX][voidY]=A
    return array

def move_up(array):
    [voidX, voidY]=find_void(array)
    A=array[voidX-1][voidY]
    array[voidX-1][voidY]=0
    array[voidX][voidY]=A
    return array

def move_right(array):
    [voidX, voidY]=find_void(array)
    A=array[voidX][voidY+1]
    array[voidX][voidY+1]=0
    array[voidX][voidY]=A
    return array

def move_left(array):
    [voidX, voidY]=find_void(array)
    A=array[voidX][voidY-1]
    array[voidX][voidY-1]=0
    array[voidX][voidY]=A
    return array