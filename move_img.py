

def move_img(scanNumber = None, PATH_MOVE = None):
    import os
    import sys
    import shutil
    if scanNumber is None:
        scanNumber = sys.argv[1]
    
    #use assert to check that scanNumber is a string
    if not isinstance(scanNumber, str):
        scanNumber = str(scanNumber)
    
    PATH_FULL = '/mn/stornext/d17/extragalactic/personal/bendikny/'
    PATH_DWNLDS = os.path.join(PATH_FULL, 'dwnlds')
    if PATH_MOVE is None:
        PATH_MOVE = os.path.join(PATH_FULL, 'code', 'PointingScanPlots')

    for path in [PATH_DWNLDS, PATH_MOVE]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    allFiles = os.listdir(PATH_DWNLDS)

    stringMatch = '_2022_' + scanNumber +'_'

    filesMoved = 0
    for fn in allFiles:
        if stringMatch in fn:
            source = os.path.join(PATH_DWNLDS, fn)
            destination = os.path.join(PATH_MOVE, fn)
    
            try:
                shutil.copy(source, destination)
                print("File copied successfully.")
        
            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

    return 



if __name__ == '__main__':
    move_img()