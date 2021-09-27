from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .Test import TestImage


#def introduction(request):
#    return render(request, 'introduction.html')

def index(request):
    context = {}
    result = []
    if request.method == 'POST':
        up = request.FILES['document']
        print("this --------------------------------  is up "+str(up))
        fs = FileSystemStorage()
        filePathName = fs.save(up.name, up)
        filePathName = fs.url(filePathName)
        testImage = "." + filePathName
        result = TestImage(testImage)
        predict = "the classification is : " + result[0]
        third = "the accurcy is :" + str(result[1]) + "%"
        context = {'filePathName': filePathName, 'predict': predict, 'third': third}
    return render(request, 'index.html',context)

'''
def index(request):
    context = {}
    if request.method == 'POST':
        up = request.FILES['document']
        fs = FileSystemStorage()
        filePathName = fs.save(up.name, up)
        filePathName = fs.url(filePathName)
        testImage = "." + filePathName
        predict = "the classification is " + TestImage(testImage)
        context = {'filePathName': filePathName, 'predict': predict}
    return render(request, 'index.html',context)
'''
