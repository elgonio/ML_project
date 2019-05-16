# script that sees the real path of a file because windows sucks and can't find the right paths for things
import os

print( os.path.abspath("dota2Test.csv") )