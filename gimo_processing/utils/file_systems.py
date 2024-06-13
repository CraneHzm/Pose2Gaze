import os
import shutil
import time


# remove a directory
def remove_dir(dirName):
	if os.path.exists(dirName):
		shutil.rmtree(dirName)
	else:
		print("Invalid directory path!")


# remake a directory
def remake_dir(dirName):
	if os.path.exists(dirName):
		shutil.rmtree(dirName)
		os.makedirs(dirName)
	else:
		os.makedirs(dirName)
	
    
# calculate the number of lines in a file
def file_lines(fileName):
	if os.path.exists(fileName):
		with open(fileName, 'r') as fr:
			return len(fr.readlines())
	else:
		print("Invalid file path!")
		return 0
	
    
# make a directory if it does not exist.
def make_dir(dirName):
	if os.path.exists(dirName):
		print("Directory "+ dirName + " already exists.")
	else:
		os.makedirs(dirName)
	
	
	
if __name__ == "__main__":
	dirName = "test"
	RemakeDir(dirName)
	time.sleep(3)
	MakeDir(dirName)
	RemoveDir(dirName)
	time.sleep(3)
	MakeDir(dirName)
	#print(FileLines('233.txt'))