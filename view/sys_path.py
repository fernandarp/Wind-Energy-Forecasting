import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
for dir in os.listdir(parentdir):
	if os.path.isfile(os.path.dirname(currentdir)+ '\\' + dir + '\\__init__.py'):
		sys.path.insert(0, os.path.dirname(currentdir)+ '\\' + dir)