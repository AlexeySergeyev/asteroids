#!/usr/bin/python
#
# Python script to request Sso from Skybot3D Web service (http://vo.imcce.fr/webservices/skybot3d/)
# (C) 2014-2016 IMCCE, J. Normand, J. Berthier
#

import os
import sys
import getopt
import httplib2

def usage():
   print("s3d-getsso.py is a Python based client of the webservice Skybot3D. It allows to")
   print("query and to get the state vectors of asteroids and comets at a reference epoch")
   print("closed to the requested epoch (arg -d) by less than 5 days. The data are stored")
   print("locally in a file formatted as a JSon structure. Visit the Skybot3D Web site for")
   print("more information.")
   print("")
   print("Usage: " + sys.argv[0] + " [options]")
   print("")
   print("Options:")
   print("  -c, --class <string>   Comma separated list of classes to get the asteroids of the given population classes")
   print("  -d, --date <string>    Set the epoch in julian day or ISO format (default to now)")
   print("  -h, --help             Display this help, and exit")
   print("  -l, --limit <int>      Number of Sso to be returned (default to all, 0 stands for all)")
   print("  -o, --output <string>  Name of the result file (default to sso.json)")
   print("  -s, --sso <string>     Type of solar system object (aster | comet | sso, default to aster)")
   print("  -v, --version          Display version information, and exit")
   print("")
   print("Examples:")
   print(" * Get all solar system objects at epoch = now:")
   print(os.path.basename(sys.argv[0]) + " -s sso -o sso.json")
   print("")
   print(" * Get all asteroids at epoch July, 1 2016:")
   print(os.path.basename(sys.argv[0]) + " -s aster -d 2016-07-01T0:0:0 -o aster.json")
   print("")
   print(" * Get all NEAs and Mars-Crosser at epoch July, 1 2016:")
   print(os.path.basename(sys.argv[0]) + " -s aster -c NEA,Mars-Crosser -d 2016-07-01T0:0:0 -o nea+mc.json")
   print("")
   print(" * Get all short-period comets at epoch July, 1 2016:")
   print(os.path.basename(sys.argv[0]) + " -s comet -c short_period -d 2016-07-01T0:0:0 -o comet_sp.json")
   print("")
   print("More information:")
   print(" Skybot3D: http://vo.imcce.fr/webservices/skybot3d/")
   print(" Skybot classes: http://vo.imcce.fr/webservices/skybot/?documentation")
   print("")

def version():
   print("s3d-getsso.py version 1.0")
   print("Written by J. Normand and J. Berthier")
   print("Copyright (C) 2009-2016, VO-IMCCE")
   print("")

def service_dic(x):
    return {
        'aster': "getAster",
        'comet': "getComet",
        'sso':   "getSso"
    }.get(x, "aster")

def build_uri(sso, coord, date, limit, mime, population):

   if sso != None:
      method = service_dic(sso)

   uri = "/webservices/skybot3d/" + method + "_query.php?-from=Skybot3DSoftware&-mime=" + mime + "&-coord=" + coord

   if date != None:
      uri = uri + "&-ep=" + date

   if limit != None:
      uri = uri + "&-limit=" + limit

   if population != None:
      uri = uri + "&-class=" + population

   return uri
   
def main():
   try:
      opts, args = getopt.getopt(sys.argv[1:], "hc:d:l:o:s:v",
                                 ["help", "class=", "date=", "limit=", "output=", "sso=", "version"])
   except: # getopt.GetoptError, err:
      print(str('Err'))
      usage()
      sys.exit(2)

   sso = "aster"
   coord = "rectangular"
   date = "now"
   limit = None
   mime = "json"
   output = "sso.json"
   population = None

   for o, a in opts:
      if o in ("-h", "--help"):
         usage()
         sys.exit()
      elif o in ("-c", "--class"):
         population = a
      elif o in ("-d", "--date"):
         date = a
      elif o in ("-l", "--limit"):
         limit = a
      elif o in ("-o", "--output"):
         output = a
      elif o in ("-s", "--sso"):
         sso = a
      elif o in ("-v", "--version"):
         version()
         sys.exit()
      else:
         assert False, "Unhandled option,\nTry 's3d-getsso.py --help' for more information"

   if date == None:
      usage()
      sys.exit()
      
   uri = build_uri(sso, coord, date, limit, mime, population)
   
   # query service
   conn = httplib2.HTTPConnectionWithTimeout("vo.imcce.fr")
   conn.request("GET", uri)
   res = conn.getresponse()
   
   if res.status == 200:
      data = res.read()
      with open('data/output.dat', "w") as f:
         print(data)
   else:
      print(os.path.basename(sys.argv[0]) + ": Error: " + str(res.status))

   conn.close()


if __name__ == "__main__":
   main()
