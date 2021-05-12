import copy
from move_func import move_up,move_down,move_left,move_right,find_void
combination={
    'начало': [[2, 8, 3], [1, 6, 4], [7, 0, 5]],
    'конец': [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
}

def print_array(array):
    for row in range(len(array)):
        print(array[row])
        print('\n')

def up_down(array):
    [voidX, voidY]=find_void(array)
    if voidX==0:
        return["Вниз"]
    elif voidX==2:
        return["Вверх"]
    else:
        return ["Вниз", "Вверх"]
    
def left_right(array):
    [voidX, voidY]=find_void(array)
    if voidY==0:
        return["Вправо"]
    elif voidY==2:
        return["Влево"]
    else:
        return ["Вправо", "Влево"]

def searchUsed(array, used):
    for row in range(len(array)):
        for col in range(len(array[row])):
            if array[row][col] != used[row][col]:
                return -1
    return 1

def NotInUsed(array, used):
    for arr in used:
        a=0
        new=list()
        for a in range (3):
            new.append(arr)
        if(searchUsed(array, arr)==1):
            print("old conf")
            return 1
    print("That's new!")
    return -1

def dfs(array, used=None):
    if array==combination['конец']:
        print("end")
        return 1
    
    used = used or list()
    if array not in used:
            used.append(copy.deepcopy(array))
            print(used)
    NotInUsed(array, used)
    print("past combination")
    print(used)
    while array!=combination['конец']:
        if("Вверх" in up_down(array)) and NotInUsed(move_up(copy.deepcopy(array)), used)==-1:
            print("вверх")
            move_up(array)
            print_array(array)
            dfs(array, used)
        elif("Влево" in left_right(array)) and NotInUsed(move_left(copy.deepcopy(array)), used)==-1:
            print("влево")
            move_left(array)
            print_array(array)
            dfs(array, used)
        elif("Вправо" in left_right(array)) and  NotInUsed(move_right(copy.deepcopy(array)), used)==-1:
            print("вправо")
            move_right(array)
            print_array(array)
            dfs(array, used)
        elif("Вниз" in up_down(array)) and  NotInUsed(move_down(copy.deepcopy(array)), used)==-1:
            print("вниз")
            move_down(array)
            print_array(array)
            dfs(array, used)
        else:
            dfs(combination['начало'], used)
            
        
        
    
dfs(combination['начало'], None)

