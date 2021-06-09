import glob
namesVideos = glob.glob("/mnt/md1/datasets/toyota_smarthome/rgb/mp4/*.mp4")

actionsList=set()

for name in namesVideos:
    name=name[43:]
    name=name.split('_')[0]
    name=name.split('.')[0]
    name=name.lower()

    actionsList.add(name)

for act in actionsList:
    print(act)
print('Cantidad de videos: ' + str(len(namesVideos)))
print('Cantidad de acciones: ' + str(len(actionsList)))